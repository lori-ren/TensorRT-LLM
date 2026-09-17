# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the decoder family's phases share, named by what kind of thing it is.

The four namespaces are the point. A phase that reaches for ``config.batch_size``
cannot mistake it for something it may write, and one that writes
``state.attn_metadata`` says so. The predicates below are derived from those
namespaces and are shared by more than one phase, so they live here rather than
being reachable through a mixin that happens to be in the MRO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Type, Union

import torch

from tensorrt_llm.logger import logger

from .....attention.backends.trtllm import TrtllmAttentionMetadata
from .....modules.mamba.mamba2_metadata import Mamba2Metadata
from .....speculative import get_draft_kv_cache_manager
from ....kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from ....kv_cache.mamba_cache_manager import BaseMambaCacheManager
from ....resource_manager import KVCacheManager, ResourceManager
from ...metadata import build_attention_metadata
from ..interface import RunnerDeps
from .state import DecoderState


def resolve_mamba_metadata_cls(model: torch.nn.Module) -> Type[Mamba2Metadata]:
    """Resolve the model-specific Mamba metadata class with a default."""
    return getattr(model, "mamba_metadata_cls", None) or Mamba2Metadata


if TYPE_CHECKING:
    from collections.abc import Callable

    from .buffers import DecoderBuffers
    from .config import DecoderRunnerConfig


@dataclass
class DecoderContext:
    """Everything the phases read, split by what it is."""

    config: "DecoderRunnerConfig"
    deps: RunnerDeps
    buffers: "DecoderBuffers"
    state: DecoderState
    warmup_flag: "Callable[[], bool]"

    @property
    def get_runtime_tokens_per_gen_step(self):
        return self.state.get_runtime_tokens_per_gen_step

    @property
    def is_encoder_decoder(self) -> bool:
        return bool(
            getattr(getattr(self.deps.model, "model_config", None), "is_encoder_decoder", False)
        )

    @property
    def is_warmup(self) -> bool:
        return self.warmup_flag()

    @property
    def use_mrope(self) -> bool:
        use_mrope = False
        try:
            use_mrope = (
                self.deps.model.model_config.pretrained_config.rope_scaling["type"] == "mrope"
            )
        except Exception:
            pass
        logger.debug(f"Detected use_mrope: {use_mrope}")
        return use_mrope

    @property
    def use_beam_search(self) -> bool:
        return self.config.max_beam_width > 1

    def set_up_attn_metadata(
        self,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        draft_kv_cache_manager: Optional[Union[KVCacheManager, KVCacheManagerV2]] = None,
    ):
        if self.state.attn_metadata is not None:
            # This assertion can be relaxed if needed: just create a new metadata
            # object if it changes.
            assert self.state.attn_metadata.kv_cache_manager is kv_cache_manager
            return self.state.attn_metadata

        config = self.deps.model.model_config.pretrained_config
        self.state.attn_metadata = build_attention_metadata(
            self.deps.model.model_config,
            max_batch_size=self.config.batch_size,
            max_num_tokens=self.config.decoder_max_num_tokens,
            max_beam_width=self.config.max_beam_width,
            attention_backend=self.config.attention_backend,
            attention_runtime_features=self.config.attention_runtime_features,
            mapping=self.deps.mapping,
            cache_indirection=self.config.cache_indirection_attention
            if self.config.attention_backend.Metadata is TrtllmAttentionMetadata
            else None,
            kv_cache_manager=kv_cache_manager,
            draft_kv_cache_manager=draft_kv_cache_manager,
        )
        if isinstance(kv_cache_manager, BaseMambaCacheManager):
            self.state.attn_metadata.mamba_chunk_size = getattr(
                config, "chunk_size", self.state.attn_metadata.mamba_chunk_size
            )
        self.state.attn_metadata.mamba_metadata_cls = resolve_mamba_metadata_cls(self.deps.model)

        return self.state.attn_metadata

    def get_draft_kv_cache_manager(
        self, resource_manager: ResourceManager
    ) -> Optional[Union[KVCacheManager, KVCacheManagerV2]]:
        """
        Returns the draft KV cache manager only in one-model speculative decoding
        mode where the target model manages a separate draft KV cache.
        """
        return get_draft_kv_cache_manager(self.config.spec_config, resource_manager)
