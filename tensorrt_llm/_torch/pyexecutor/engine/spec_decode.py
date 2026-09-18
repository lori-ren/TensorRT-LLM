# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding metadata construction for the PyTorch model engine."""

from typing import Any

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.speculative import SpecMetadata, get_spec_metadata
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

from ..resource_manager import ResourceManager, ResourceManagerType
from ..scheduler import ScheduledRequests
from .metadata import update_spec_metadata


class SpecMetadataBuilder:
    """Builds and refreshes the per-iteration speculative-decoding metadata.

    Every family that speculates does the same three things: take the spec
    resource manager off the resource manager, build the metadata from it, then
    update it for the scheduled batch. The arguments are the same whichever
    family runs it, so the engine resolves them once here instead of each
    runner configuration carrying them.

    Latching those is safe: the engine assigns each exactly once, before it
    constructs the builder. What changes between forwards -- whether
    speculation is on for this pass, and the runtime draft length -- is what
    the methods below take per call instead.
    """

    def __init__(
        self,
        *,
        spec_config: DecodingBaseConfig | None,
        model_config: Any,
        is_draft_model: bool,
        original_max_draft_len: int,
        original_max_total_draft_tokens: int,
        spec_dec_max_total_draft_tokens: int,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        num_seq_slots: int | None,
        attn_backend: type[AttentionMetadata],
    ) -> None:
        self._spec_config = spec_config
        self._model_config = model_config
        self._is_draft_model = is_draft_model
        self._original_max_draft_len = original_max_draft_len
        self._original_max_total_draft_tokens = original_max_total_draft_tokens
        self._spec_dec_max_total_draft_tokens = spec_dec_max_total_draft_tokens
        self._max_batch_size = max_batch_size
        self._max_num_tokens = max_num_tokens
        self._max_seq_len = max_seq_len
        self._num_seq_slots = num_seq_slots
        self._attn_backend = attn_backend

    @property
    def enabled(self) -> bool:
        """Whether this engine was configured to speculate at all."""
        return self._spec_config is not None

    def runtime_tokens_per_gen_step(self, runtime_draft_len: int) -> int:
        if self._spec_config is None:
            return 1
        return self._spec_config.get_runtime_tokens_per_gen_step(runtime_draft_len)

    def resource_managers(self, resource_manager: ResourceManager) -> tuple[Any, Any]:
        """The spec resource manager and the tree manager it may carry."""
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        return spec_resource_manager, getattr(spec_resource_manager, "spec_tree_manager", None)

    def build(self, spec_resource_manager: Any, *, enabled: bool = True) -> SpecMetadata | None:
        """Metadata for a batch this pass will speculate on, or None."""
        return get_spec_metadata(
            self._spec_config if enabled else None,
            self._model_config,
            self._max_batch_size,
            max_num_tokens=self._max_num_tokens,
            spec_resource_manager=spec_resource_manager,
            is_draft_model=self._is_draft_model,
            max_seq_len=self._max_seq_len,
            num_seq_slots=self._num_seq_slots,
        )

    def update(
        self,
        spec_metadata: SpecMetadata,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        *,
        spec_tree_manager: Any,
        runtime_draft_len: int,
    ) -> None:
        update_spec_metadata(
            spec_metadata,
            scheduled_requests,
            attn_metadata,
            spec_tree_manager=spec_tree_manager,
            runtime_draft_len=runtime_draft_len,
            runtime_tokens_per_gen_step=self.runtime_tokens_per_gen_step(runtime_draft_len),
            is_draft_model=self._is_draft_model,
            attention_backend=self._attn_backend,
            original_max_draft_len=self._original_max_draft_len,
            original_max_total_draft_tokens=self._original_max_total_draft_tokens,
            spec_dec_max_total_draft_tokens=self._spec_dec_max_total_draft_tokens,
        )
