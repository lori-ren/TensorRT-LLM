# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder family."""

from __future__ import annotations

from typing import Any, Callable

from ..interface import RunnerDeps
from .buffers import DecoderBuffers
from .config import DecoderConfigMixin, DecoderRunnerConfig
from .mixin import DecoderMixin

__all__ = [
    "DecoderBuffers",
    "DecoderConfigMixin",
    "DecoderMixin",
    "DecoderRunner",
    "DecoderRunnerConfig",
]


class DecoderRunner(DecoderMixin):
    """Run a decoder-only model whose batch arrives as scheduled requests."""

    def __init__(
        self,
        model: Any,
        deps: RunnerDeps,
        config: DecoderRunnerConfig,
        buffers: DecoderBuffers,
        *,
        warmup_flag: Callable[[], bool],
    ) -> None:
        self._initialize_decoder(model, deps, config, buffers, warmup_flag=warmup_flag)
