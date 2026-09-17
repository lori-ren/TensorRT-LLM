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
    """Startup settings the decoder family reads while serving.

    Named as ``EncoderDecoderRunnerConfig`` anticipated: encoder-decoder composes
    this beside ``EncoderConfigMixin`` because it runs the same decoder half.
    """

    batch_size: int
    decoder_max_num_tokens: int
    dtype: torch.dtype
    is_draft_model: bool
    is_spec_decode: bool
    spec_config: Any
    kv_cache_manager_key: Any
    llm_args: Any
    input_processor: Any
    lora_model_config: Any
    lora: Any
    moe_load_balancer: Any
    original_max_draft_len: int
    original_max_total_draft_tokens: int
    spec_dec_max_total_draft_tokens: int
    initial_runtime_draft_len: int
    max_total_draft_tokens: int
    max_draft_len: int
    max_draft_loop_tokens: int
    max_num_seq_slots: int
    enable_attention_dp: bool
    disable_overlap_scheduler: bool
    enable_in_graph_sampling: bool
    enable_disagg_adp_overlap_headroom: bool
    sparse_attention_config: Any
    prefill_cuda_graph_backend: Any
    prefill_cuda_graph_num_tokens: Any
    cuda_graph_batch_sizes: Any
    dynamic_draft_len_mapping: Any
    encoder_graph_shapes: Any
    steady_gen_positions_pinned: Any
    torch_compile_enabled: bool
    torch_compile_piecewise_cuda_graph: bool
    torch_compile_backend: Any
    backend_num_streams: Any
    cache_indirection_attention: Optional[torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class DecoderRunnerConfig(DecoderConfigMixin, RunnerConfig):
    """Configuration for the decoder-only runner."""
