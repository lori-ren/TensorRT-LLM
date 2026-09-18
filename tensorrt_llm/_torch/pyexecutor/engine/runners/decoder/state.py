# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The values in force for the duration of a call, not for the family's lifetime.

Everything else the decoder family holds now lives with whatever writes it. What
is left here is the set that is overridden and restored around a piece of work --
a capture pass, a warmup batch, one iteration's draft length. They are still
fields because the paths that read them have not been threaded yet; each one is a
parameter waiting to happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DecoderState:
    """Scoped overrides the family reads while a call is in flight."""

    # Set per iteration by the executor, through the coordinator.
    enable_spec_decode: bool = False
    runtime_draft_len: int = 0

    # Passed into forward, then read by the phases it drives.
    cuda_graph_lora_manager: Any = None

    # Set by a capture pass for its own duration.
    force_lora_graph_for_capture: Optional[bool] = None
    trtllm_gen_jit_warmup: bool = False
