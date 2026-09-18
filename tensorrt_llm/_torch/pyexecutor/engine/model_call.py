# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calling the model the way a compiled backend expects, for any runner."""

import weakref
from typing import Any

import torch
from torch import nn

from tensorrt_llm._utils import is_trace_enabled, trace_func

from ..utils import get_model_extra_attrs


class ModelCaller:
    """Publishes the per-iteration extra attrs, then calls the model.

    Every family calls the model this way, so the engine resolves the pieces
    once here rather than each runner rebuilding the call. Nothing reaches back
    into the engine: the model, the compile backend and its streams are all
    final before the engine constructs a runner.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        torch_compile_backend: Any,
        backend_num_streams: Any,
    ) -> None:
        self._model = model
        self._torch_compile_backend = torch_compile_backend
        self._backend_num_streams = backend_num_streams

    def __call__(self, **kwargs):
        attrs = get_model_extra_attrs()
        assert attrs is not None, "Model extra attrs is not set"
        attrs["attention_metadata"] = weakref.ref(kwargs["attn_metadata"])
        attrs.update(self._model.model_config.extra_attrs)
        attrs["spec_metadata"] = kwargs.get("spec_metadata", None)

        if self._torch_compile_backend is not None:
            # Register aux streams and events to model extra attrs.
            # The streams and events are list which could be updated during compilation.
            attrs["aux_streams"] = weakref.ref(self._backend_num_streams)
            attrs["events"] = weakref.ref(self._torch_compile_backend.events)
            attrs["global_stream"] = torch.cuda.current_stream()

        if is_trace_enabled("TLLM_TRACE_MODEL_FORWARD"):
            return trace_func(self._model.forward)(**kwargs)
        return self._model.forward(**kwargs)
