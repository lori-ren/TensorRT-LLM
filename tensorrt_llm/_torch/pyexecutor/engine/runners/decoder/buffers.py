# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The device buffers the decoder family writes.

The engine allocates them -- sizing them is part of its capacity planning, which
T2.5 owns -- and hands them over. After that the family is their only reader and
writer, which is what invariant 3 asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DecoderBuffers:
    """Device buffers the decoder family writes.

    The engine allocates them — sizing them is part of ``__init__``'s capacity
    planning, which T2.5 owns — and hands them over. After that the runner is
    their only reader and writer, which is what invariant 3 asks for.
    """

    input_ids_cuda: torch.Tensor
    position_ids_cuda: torch.Tensor
    gather_ids_cuda: Optional[torch.Tensor] = None
    draft_tokens_cuda: Optional[torch.Tensor] = None
    mrope_position_ids_cuda: Optional[torch.Tensor] = None
    num_accepted_draft_tokens_cuda: Optional[torch.Tensor] = None
    previous_batch_indices_cuda: Optional[torch.Tensor] = None
    previous_kv_lens_offsets_cuda: Optional[torch.Tensor] = None
    previous_pos_id_offsets_cuda: Optional[torch.Tensor] = None
    previous_pos_indices_cuda: Optional[torch.Tensor] = None
    draft_ctx_seq_slots_cuda: Optional[torch.Tensor] = None
    draft_ctx_token_indices_cuda: Optional[torch.Tensor] = None
    draft_first_draft_indices_cuda: Optional[torch.Tensor] = None
    draft_first_draft_seq_slots_cuda: Optional[torch.Tensor] = None
    draft_request_indices_buffer_cuda: Optional[torch.Tensor] = None
    draft_seq_slots_buffer_cuda: Optional[torch.Tensor] = None
