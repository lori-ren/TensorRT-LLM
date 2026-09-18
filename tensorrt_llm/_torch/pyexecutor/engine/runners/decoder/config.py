# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup settings the decoder family reads while serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..interface import RunnerConfig


@dataclass(frozen=True, kw_only=True)
class DecoderConfigMixin:
    """Decoder-specific fields shared by runner configuration types.

    The counterpart of ``EncoderConfigMixin``: whichever config a runner
    carries, this is the part its decoder half reads, so the half is handed
    the config under this type and never under a leaf's.

    The decoder half is whatever its runner's ``forward`` runs, so it reads the
    contract's ``max_batch_size`` and ``max_num_tokens`` directly.
    """

    dtype: torch.dtype
    spec_config: Any
    is_draft_model: bool
    original_max_draft_len: int
    kv_cache_manager_key: Any
    cuda_graph_specialize_lora: bool
    input_processor: Any
    lora_model_config: Any
    initial_runtime_draft_len: int
    max_total_draft_tokens: int
    max_draft_len: int
    max_draft_loop_tokens: int
    enable_attention_dp: bool
    disable_overlap_scheduler: bool
    enable_in_graph_sampling: bool
    sparse_attention_config: Any
    prefill_cuda_graph_backend: Any
    prefill_cuda_graph_num_tokens: Any
    decoder_cuda_graph_batch_sizes: Any
    dynamic_draft_len_mapping: Any
    steady_gen_positions_pinned: Any
    torch_compile_enabled: bool
    torch_compile_piecewise_cuda_graph: bool
    cache_indirection_attention: Optional[torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class DecoderRunnerConfig(DecoderConfigMixin, RunnerConfig):
    """Configuration for the decoder-only runner."""
