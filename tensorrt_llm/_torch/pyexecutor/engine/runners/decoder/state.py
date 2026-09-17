# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder family's mutable state, declared rather than accumulated.

Every field here is written by the family while it runs. Nothing outside the
family reads one directly; the coordinator forwards the two the executor sets
through, which is what keeps the cross-module state set empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class DecoderState:
    """Per-batch and per-run state owned by one decoder family instance."""

    capture_sample_type: Any = None
    cross_attn_stable_cached_tokens: Optional[List[int]] = None
    cross_attn_stable_request_ids: Optional[List[int]] = None
    encoder_decoder_host_buffer_pool: List[Dict[str, Any]] = field(default_factory=list)
    encoder_decoder_input_fast_path_static_eligible: Any = None
    encoder_decoder_position_id_offset: Any = None
    encoder_decoder_staged_request_ids: Any = None
    force_lora_graph_for_capture: Any = None
    prepare_inputs_event: Optional[torch.cuda.Event] = None
    stage_in_graph_sampling: Optional[Callable] = None
    steady_gen_cache: Any = None
    trtllm_gen_jit_warmup: Any = False
    attn_metadata: Any = None
    breakable_cuda_graph_runner: Any = None
    cuda_graph_lora_manager: Any = None
    cuda_graph_runner: Any = None
    enable_spec_decode: Any = None
    forward_pass_callable: Optional[Callable] = None
    guided_decoder: Any = None
    iter_states: Dict[str, Any] = field(default_factory=dict)
    previous_request_ids: List[int] = field(default_factory=list)
    runtime_draft_len: Any = None
    sample_in_graph_callable: Optional[Callable] = None
    spec_metadata: Any = None
