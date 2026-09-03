"""Attention and speculative-decoding metadata factories.

Every function here takes what it reads and returns what it builds. None of
them holds or writes the engine: the three metadata objects are memoized on
``PyTorchModelEngine`` because ``py_executor`` reads them from outside
(``getattr(engine, 'attn_metadata', None)``) and ``py_executor_creator``
writes ``None`` back during the KV-cache-estimation rebuild. The engine hands
the old value in as ``cached`` and stores whatever comes back.
"""

import torch
from torch import nn

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionBackend,
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    get_draft_kv_cache_manager,
    get_spec_metadata,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.mapping import Mapping

from ..config_utils import is_mla
from ..kv_cache_manager_v2 import KVCacheManagerV2
from ..mamba_cache_manager import BaseMambaCacheManager
from ..resource_manager import BaseResourceManager, KVCacheManager, ResourceManager

AnyKVCacheManager = KVCacheManager | KVCacheManagerV2


def resolve_mamba_metadata_cls(model: nn.Module) -> type[Mamba2Metadata]:
    """Resolve the model-specific Mamba metadata class with a default."""
    return getattr(model, "mamba_metadata_cls", None) or Mamba2Metadata


def _shared_attn_metadata_kwargs(
    model: nn.Module,
    attn_backend: type[AttentionBackend],
    attn_runtime_features: AttentionRuntimeFeatures,
    sparse_attention_config,
    cache_indirection_attention: torch.Tensor | None,
    batch_size: int,
    max_num_tokens: int,
    max_beam_width: int,
    mapping: Mapping,
) -> dict:
    """The construction keywords both attention factories pass unchanged."""
    enable_context_mla_with_cached_kv = is_mla(model.model_config.pretrained_config) and (
        attn_runtime_features.cache_reuse or attn_runtime_features.chunked_prefill
    )
    cache_indirection = (
        cache_indirection_attention if attn_backend.Metadata is TrtllmAttentionMetadata else None
    )
    num_attention_heads = getattr(model.model_config.pretrained_config, "num_attention_heads", None)
    config = model.model_config.pretrained_config

    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", None)

    # Calculate the number of attention heads per KV head (GQA ratio)
    if isinstance(num_key_value_heads, (list, tuple)):
        # Filter out invalid KV heads, default to 0 if no valid KV heads are found
        num_key_value_heads = min((kv for kv in num_key_value_heads if kv and kv > 0), default=0)
    if num_attention_heads and num_key_value_heads:
        num_heads_per_kv = num_attention_heads // num_key_value_heads
    else:
        num_heads_per_kv = 1

    sparse_metadata_params = (
        sparse_attention_config.to_sparse_metadata_params(pretrained_config=config)
        if sparse_attention_config is not None
        else None
    )

    return dict(
        max_num_requests=batch_size,
        max_num_tokens=max_num_tokens,
        max_num_sequences=batch_size * max_beam_width,
        mapping=mapping,
        runtime_features=attn_runtime_features,
        enable_flash_mla=model.model_config.enable_flash_mla,
        enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
        cache_indirection=cache_indirection,
        num_heads_per_kv=num_heads_per_kv,
        sparse_metadata_params=sparse_metadata_params,
    )


def set_up_no_cache_attn_metadata(
    model: nn.Module,
    cached: AttentionMetadata | None,
    attn_backend: type[AttentionBackend],
    attn_runtime_features: AttentionRuntimeFeatures,
    sparse_attention_config,
    cache_indirection_attention: torch.Tensor | None,
    batch_size: int,
    max_num_tokens: int,
    max_beam_width: int,
    mapping: Mapping,
) -> AttentionMetadata:
    """Build the attention metadata used when there is no KV cache manager.

    ``no cache`` means no KV cache manager, not "do not memoize" -- the engine
    still memoizes the result, in ``encoder_attn_metadata``.
    """
    if cached is not None:
        return cached

    metadata_cls = attn_backend.Metadata
    attn_metadata = metadata_cls(
        kv_cache_manager=None,
        **_shared_attn_metadata_kwargs(
            model,
            attn_backend,
            attn_runtime_features,
            sparse_attention_config,
            cache_indirection_attention,
            batch_size,
            max_num_tokens,
            max_beam_width,
            mapping,
        ),
    )
    attn_metadata.block_ids_per_seq = None
    attn_metadata.kv_block_ids_per_seq = None
    return attn_metadata


def set_up_attn_metadata(
    model: nn.Module,
    kv_cache_manager: AnyKVCacheManager,
    draft_kv_cache_manager: AnyKVCacheManager | None,
    cached: AttentionMetadata | None,
    attn_backend: type[AttentionBackend],
    attn_runtime_features: AttentionRuntimeFeatures,
    sparse_attention_config,
    cache_indirection_attention: torch.Tensor | None,
    batch_size: int,
    max_num_tokens: int,
    max_beam_width: int,
    mapping: Mapping,
) -> AttentionMetadata:
    """Build the attention metadata bound to ``kv_cache_manager``."""
    if cached is not None:
        # This assertion can be relaxed if needed: just create a new metadata
        # object if it changes.
        assert cached.kv_cache_manager is kv_cache_manager
        return cached

    config = model.model_config.pretrained_config
    metadata_cls = attn_backend.Metadata
    attn_metadata = metadata_cls(
        kv_cache_manager=kv_cache_manager,
        draft_kv_cache_manager=draft_kv_cache_manager,
        **_shared_attn_metadata_kwargs(
            model,
            attn_backend,
            attn_runtime_features,
            sparse_attention_config,
            cache_indirection_attention,
            batch_size,
            max_num_tokens,
            max_beam_width,
            mapping,
        ),
    )
    if isinstance(kv_cache_manager, BaseMambaCacheManager):
        attn_metadata.mamba_chunk_size = getattr(
            config, "chunk_size", attn_metadata.mamba_chunk_size
        )
    attn_metadata.mamba_metadata_cls = resolve_mamba_metadata_cls(model)

    return attn_metadata


def set_up_spec_metadata(
    model: nn.Module,
    spec_resource_manager: BaseResourceManager | None,
    cached: SpecMetadata | None,
    no_cache: bool,
    spec_config,
    enable_spec_decode: bool,
    is_draft_model: bool,
    batch_size: int,
    max_num_tokens: int,
    max_seq_len: int,
    max_num_seq_slots: int,
    enable_disagg_adp_overlap_headroom: bool,
) -> SpecMetadata:
    """Build the speculative-decoding metadata.

    With ``no_cache`` the result is fresh every call and the engine does not
    memoize it; otherwise ``cached`` short-circuits construction.
    """
    spec_config = spec_config if enable_spec_decode else None
    # The disaggregated attention-DP overlap path opts into larger metadata
    # buffers. Passing None preserves the established max_num_requests
    # fallback for other configurations, including PP.
    num_seq_slots = max_num_seq_slots if enable_disagg_adp_overlap_headroom else None
    if no_cache:
        return get_spec_metadata(
            spec_config,
            model.config,
            batch_size,
            max_num_tokens=max_num_tokens,
            spec_resource_manager=spec_resource_manager,
            is_draft_model=is_draft_model,
            max_seq_len=max_seq_len,
            num_seq_slots=num_seq_slots,
        )

    if cached is not None:
        return cached
    return get_spec_metadata(
        spec_config,
        model.config,
        batch_size,
        max_num_tokens=max_num_tokens,
        spec_resource_manager=spec_resource_manager,
        is_draft_model=is_draft_model,
        max_seq_len=max_seq_len,
        num_seq_slots=num_seq_slots,
    )


def sync_group_all_greedy_sample(spec_metadata: SpecMetadata, mapping: Mapping, dist) -> None:
    """All-gather the per-rank greedy flags and store the group AND.

    Why the sampling-path choice must be group-uniform under
    ADP + LM-head TP is documented on the anchor,
    ``SpecMetadata.group_all_greedy_sample``. Local contract: called once
    per iteration, right after ``update_is_all_greedy_sample`` and BEFORE
    the CUDA graph key is built. The gate is pure config (identical on
    every rank), so ranks also agree on whether the exchange happens; the
    gather spans the whole TP group, a superset of any LM-head-TP
    subgroup. A dedicated host all-gather rather than a piggyback on the
    ``all_rank_num_tokens`` exchange, which runs in ``_prepare_inputs`` --
    after the graph key, too late for the key to see the synced value.
    """
    # enable_lm_head_tp_in_adp implies enable_attention_dp (asserted in
    # Mapping.__init__), so ADP needs no separate check here.
    if not (mapping.enable_lm_head_tp_in_adp and spec_metadata.use_rejection_sampling):
        return
    local_flag = bool(spec_metadata.is_all_greedy_sample)
    all_flags = dist.tp_allgather(local_flag, small_payload=True)
    spec_metadata.group_all_greedy_sample = all(all_flags)
    # Also overwrite the live flag directly: this iteration's scan already
    # ran (update_is_all_greedy_sample just returned) and the CUDA graph
    # key reads the flag next -- the stored override only takes effect on
    # the NEXT rescan (populate), which is after key selection.
    spec_metadata.is_all_greedy_sample = spec_metadata.group_all_greedy_sample


def set_spec_metadata_all_rank_num_tokens(
    spec_metadata: SpecMetadata,
    spec_all_rank_num_tokens: list[int],
    all_rank_num_seqs: list[int],
    all_rank_num_gens: list[int] | None = None,
) -> None:
    # Eagle3 / MTP-eagle one-model use subseq_all_rank_num_tokens for
    # draft loop iterations i>0 (per-sequence counts, since each
    # sequence contributes one token per iteration).
    spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
    spec_metadata.all_rank_num_seqs = all_rank_num_seqs
    # DSpark can draft only after the target processes the current bonus token,
    # because it consumes captured target-layer hidden states for that token.
    # Prefill computes hidden states for prompt tokens; the first generated token
    # is sampled from the last prompt logits and has not itself passed through the
    # target layers. Thus context requests seed the rolling window but do not run
    # the draft. On mixed steps, num_seqs therefore over-counts the draft MoE
    # workload; gen-only per-rank counts keep the FUSED_COMM (DeepGEMM MegaMoE)
    # chunk loop identical across EP ranks.
    if all_rank_num_gens is not None:
        spec_metadata.all_rank_num_gens = all_rank_num_gens
    if (
        spec_metadata.spec_dec_mode.is_mtp_eagle_one_model()
        or spec_metadata.spec_dec_mode.is_eagle3_one_model()
    ):
        spec_metadata.subseq_all_rank_num_tokens = all_rank_num_seqs


def make_encoder_attn_metadata(
    model: nn.Module,
    sequence_lengths: list[int],
    request_ids: list[int],
    attn_backend: type[AttentionBackend],
    attn_runtime_features: AttentionRuntimeFeatures,
    sparse_attention_config,
    encoder_batch_size: int,
    encoder_max_num_tokens: int,
    max_beam_width: int,
    max_seq_len: int,
    mapping: Mapping,
) -> AttentionMetadata:
    """Build fresh, no-cache attention metadata for one packed encoder
    batch. ``engine.attn_metadata`` is not reused because that object is
    bound to the decoder's KV-cache manager."""
    if len(sequence_lengths) != len(request_ids):
        raise ValueError("Encoder sequence lengths and request IDs must have the same length.")
    sparse_metadata_params = (
        sparse_attention_config.to_sparse_metadata_params(
            pretrained_config=model.model_config.pretrained_config
        )
        if sparse_attention_config is not None
        else None
    )
    encoder_attn_metadata = attn_backend.Metadata(
        max_num_requests=encoder_batch_size,
        max_num_tokens=encoder_max_num_tokens,
        max_num_sequences=encoder_batch_size * max_beam_width,
        kv_cache_manager=None,
        mapping=mapping,
        runtime_features=attn_runtime_features,
        enable_flash_mla=model.model_config.enable_flash_mla,
        enable_context_mla_with_cached_kv=False,
        cache_indirection=None,
        sparse_metadata_params=sparse_metadata_params,
        num_heads_per_kv=1,
    )
    assert isinstance(encoder_attn_metadata, (VanillaAttentionMetadata, TrtllmAttentionMetadata)), (
        "Only vanilla and trtllm attention metadata are supported for the encoder pass"
    )

    encoder_attn_metadata.seq_lens = torch.tensor(
        sequence_lengths,
        dtype=torch.int,
        pin_memory=prefer_pinned(),
    )
    encoder_attn_metadata.num_contexts = len(sequence_lengths)
    encoder_attn_metadata.max_seq_len = max_seq_len
    encoder_attn_metadata.request_ids = request_ids
    encoder_attn_metadata.prepare_encoder_only()
    return encoder_attn_metadata


def resolve_draft_kv_cache_manager(
    spec_config, resource_manager: ResourceManager
) -> AnyKVCacheManager | None:
    """
    Returns the draft KV cache manager only in one-model speculative decoding
    mode where the target model manages a separate draft KV cache.
    """
    return get_draft_kv_cache_manager(spec_config, resource_manager)
