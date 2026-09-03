# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the attention / speculative-decoding metadata factories.

Two contracts here are load-bearing and had no upstream coverage:

* ``set_up_attn_metadata`` asserts that a cached metadata object is still
  bound to the same KV cache manager. That assertion is the reason
  ``py_executor_creator`` nulls ``engine.attn_metadata`` before rebuilding the
  KV cache -- without it, a stale object would be handed to a new manager.
* ``sync_group_all_greedy_sample`` must not touch the collective when its gate
  is off: the gate is pure config, so every rank has to agree on whether the
  all-gather happens at all.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.engine.metadata import (
    resolve_mamba_metadata_cls,
    set_spec_metadata_all_rank_num_tokens,
    set_up_attn_metadata,
    set_up_no_cache_attn_metadata,
    sync_group_all_greedy_sample,
)

pytestmark = pytest.mark.cpu_only


class _RecordingMetadata:
    """Stands in for an ``AttentionMetadata`` subclass.

    The factories only ever construct it and set attributes, so recording the
    construction keywords is enough to tell "built fresh" from "reused".
    """

    instances: list["_RecordingMetadata"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.kv_cache_manager = kwargs.get("kv_cache_manager")
        self.mamba_chunk_size = 0
        _RecordingMetadata.instances.append(self)


def _backend() -> type:
    return SimpleNamespace(Metadata=_RecordingMetadata)


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(num_attention_heads=32, num_key_value_heads=8),
            enable_flash_mla=False,
        )
    )


def _env() -> dict[str, Any]:
    return dict(
        attn_backend=_backend(),
        attn_runtime_features=SimpleNamespace(cache_reuse=False, chunked_prefill=False),
        sparse_attention_config=None,
        cache_indirection_attention=None,
        batch_size=4,
        max_num_tokens=128,
        max_beam_width=1,
        mapping=SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    _RecordingMetadata.instances.clear()


def test_attn_metadata_reuses_cache_bound_to_the_same_kv_manager() -> None:
    kv_manager = object()
    cached = SimpleNamespace(kv_cache_manager=kv_manager)

    result = set_up_attn_metadata(_model(), kv_manager, None, cached, **_env())

    assert result is cached
    assert _RecordingMetadata.instances == []


def test_attn_metadata_rejects_a_cache_bound_to_another_kv_manager() -> None:
    cached = SimpleNamespace(kv_cache_manager=object())

    with pytest.raises(AssertionError):
        set_up_attn_metadata(_model(), object(), None, cached, **_env())


def test_attn_metadata_builds_fresh_when_nothing_is_cached() -> None:
    kv_manager = object()
    draft_kv_manager = object()

    result = set_up_attn_metadata(_model(), kv_manager, draft_kv_manager, None, **_env())

    assert len(_RecordingMetadata.instances) == 1
    assert result.kwargs["kv_cache_manager"] is kv_manager
    assert result.kwargs["draft_kv_cache_manager"] is draft_kv_manager
    # 32 attention heads over 8 KV heads.
    assert result.kwargs["num_heads_per_kv"] == 4
    assert result.mamba_metadata_cls is Mamba2Metadata


def test_no_cache_attn_metadata_clears_both_block_id_fields() -> None:
    result = set_up_no_cache_attn_metadata(_model(), None, **_env())

    assert result.kwargs["kv_cache_manager"] is None
    assert result.block_ids_per_seq is None
    assert result.kv_block_ids_per_seq is None


def test_no_cache_attn_metadata_reuses_the_cache() -> None:
    cached = SimpleNamespace()

    result = set_up_no_cache_attn_metadata(_model(), cached, **_env())

    assert result is cached
    assert _RecordingMetadata.instances == []


def test_resolve_mamba_metadata_cls_prefers_the_model_override() -> None:
    class _Custom(Mamba2Metadata):
        pass

    assert resolve_mamba_metadata_cls(SimpleNamespace(mamba_metadata_cls=_Custom)) is _Custom
    assert resolve_mamba_metadata_cls(SimpleNamespace()) is Mamba2Metadata


def _spec_metadata(*, is_mtp_eagle: bool = False, is_eagle3: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        spec_dec_mode=SimpleNamespace(
            is_mtp_eagle_one_model=lambda: is_mtp_eagle,
            is_eagle3_one_model=lambda: is_eagle3,
        )
    )


def test_all_rank_num_tokens_leaves_gens_alone_when_not_given() -> None:
    spec_metadata = _spec_metadata()

    set_spec_metadata_all_rank_num_tokens(spec_metadata, [4, 5], [1, 2])

    assert spec_metadata.all_rank_num_tokens == [4, 5]
    assert spec_metadata.all_rank_num_seqs == [1, 2]
    assert not hasattr(spec_metadata, "all_rank_num_gens")
    assert not hasattr(spec_metadata, "subseq_all_rank_num_tokens")


@pytest.mark.parametrize("mode", ["is_mtp_eagle", "is_eagle3"])
def test_all_rank_num_tokens_sets_subseq_for_one_model_modes(mode: str) -> None:
    spec_metadata = _spec_metadata(**{mode: True})

    set_spec_metadata_all_rank_num_tokens(spec_metadata, [4, 5], [1, 2], [3, 3])

    assert spec_metadata.all_rank_num_gens == [3, 3]
    assert spec_metadata.subseq_all_rank_num_tokens == [1, 2]


class _RecordingDist:
    def __init__(self, flags: list[bool]) -> None:
        self.flags = flags
        self.calls = 0

    def tp_allgather(self, value: bool, small_payload: bool = False) -> list:
        self.calls += 1
        return self.flags


@pytest.mark.parametrize(
    "enable_lm_head_tp_in_adp,use_rejection_sampling",
    [(False, True), (True, False), (False, False)],
    ids=["no-lm-head-tp", "no-rejection-sampling", "neither"],
)
def test_group_sync_skips_the_collective_when_the_gate_is_off(
    enable_lm_head_tp_in_adp: bool, use_rejection_sampling: bool
) -> None:
    spec_metadata = SimpleNamespace(
        use_rejection_sampling=use_rejection_sampling,
        is_all_greedy_sample=True,
        group_all_greedy_sample=None,
    )
    dist = _RecordingDist([True, False])

    sync_group_all_greedy_sample(
        spec_metadata, SimpleNamespace(enable_lm_head_tp_in_adp=enable_lm_head_tp_in_adp), dist
    )

    assert dist.calls == 0
    assert spec_metadata.group_all_greedy_sample is None
    assert spec_metadata.is_all_greedy_sample is True


def test_group_sync_ands_the_flags_and_overwrites_the_live_one() -> None:
    spec_metadata = SimpleNamespace(
        use_rejection_sampling=True, is_all_greedy_sample=True, group_all_greedy_sample=None
    )
    dist = _RecordingDist([True, False])

    sync_group_all_greedy_sample(
        spec_metadata, SimpleNamespace(enable_lm_head_tp_in_adp=True), dist
    )

    assert dist.calls == 1
    assert spec_metadata.group_all_greedy_sample is False
    # The live flag is overwritten too: the CUDA graph key reads it next, and
    # the stored override would otherwise only apply from the next rescan.
    assert spec_metadata.is_all_greedy_sample is False
