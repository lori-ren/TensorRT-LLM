# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared decoder-family mechanics.

The counterpart of ``EncoderMixin``: it owns the decoder half's context, its
phases and its lifecycle, and knows nothing about any other half. A runner
composes it with whatever else its model needs, and substitutes a specialized
phase through the factories below.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, cast

from ..interface import RunnerConfig, RunnerDeps
from .buffers import DecoderBuffers
from .config import DecoderConfigMixin
from .context import DecoderContext
from .forward import ForwardExecutor
from .prepare import InputPreparer
from .state import DecoderState
from .warmup import WarmupDriver


class DecoderMixin:
    """The decoder half: assembly, dispatch and state."""

    def _initialize_decoder(
        self,
        model: Any,
        deps: RunnerDeps,
        config: DecoderConfigMixin,
        buffers: DecoderBuffers,
        *,
        warmup_flag: Callable[[], bool],
    ) -> None:
        self._ctx = DecoderContext(
            config=config,
            runner_config=cast(RunnerConfig, config),
            deps=deps,
            buffers=buffers,
            state=DecoderState(
                enable_spec_decode=config.spec_config is not None,
                runtime_draft_len=config.initial_runtime_draft_len,
            ),
            warmup_flag=warmup_flag,
        )
        self._preparer = self._make_preparer()
        self._executor = self._make_executor(self._preparer)
        self._warmup = self._make_warmup(self._preparer, self._executor)

    # ---- what the coordinator installs; each phase gets its own handle ----
    @property
    def cuda_graph_runner(self):
        return self._ctx.cuda_graph_runner

    @cuda_graph_runner.setter
    def cuda_graph_runner(self, value) -> None:
        self._ctx.cuda_graph_runner = value

    @property
    def breakable_cuda_graph_runner(self):
        return self._ctx.breakable_cuda_graph_runner

    @breakable_cuda_graph_runner.setter
    def breakable_cuda_graph_runner(self, value) -> None:
        self._ctx.breakable_cuda_graph_runner = value

    @property
    def guided_decoder(self):
        return self._ctx.guided_decoder

    @guided_decoder.setter
    def guided_decoder(self, value) -> None:
        self._ctx.guided_decoder = value

    @property
    def forward_pass_callable(self):
        return self._executor.forward_pass_callable

    @forward_pass_callable.setter
    def forward_pass_callable(self, value) -> None:
        self._executor.forward_pass_callable = value

    @property
    def sample_in_graph_callable(self):
        return self._executor.sample_in_graph_callable

    @sample_in_graph_callable.setter
    def sample_in_graph_callable(self, value) -> None:
        self._executor.sample_in_graph_callable = value

    @property
    def stage_in_graph_sampling(self):
        return self._executor.stage_in_graph_sampling

    @stage_in_graph_sampling.setter
    def stage_in_graph_sampling(self, value) -> None:
        self._executor.stage_in_graph_sampling = value

    @property
    def attn_metadata(self):
        return self._ctx.attn_metadata

    @attn_metadata.setter
    def attn_metadata(self, value) -> None:
        self._ctx.attn_metadata = value

    @property
    def spec_metadata(self):
        return self._ctx.spec_metadata

    @spec_metadata.setter
    def spec_metadata(self, value) -> None:
        self._ctx.spec_metadata = value

    @property
    def iter_states(self) -> dict:
        return self._ctx.iter_states

    def _make_preparer(self) -> InputPreparer:
        return InputPreparer(self._ctx)

    def _make_executor(self, preparer: InputPreparer) -> ForwardExecutor:
        return ForwardExecutor(self._ctx, preparer)

    def _make_warmup(self, preparer: InputPreparer, executor: ForwardExecutor) -> WarmupDriver:
        return WarmupDriver(self._ctx, preparer, executor)

    def forward(
        self,
        scheduled_requests,
        *,
        resource_manager,
        cuda_graph_lora_manager=None,
        runtime_draft_len=None,
        gather_context_logits: bool = False,
        new_tensors_device=None,
        cache_indirection_buffer=None,
        num_accepted_tokens_device=None,
        req_id_to_old_request=None,
    ):
        if cuda_graph_lora_manager is not None:
            self._ctx.state.cuda_graph_lora_manager = cuda_graph_lora_manager
        if runtime_draft_len is not None:
            self._ctx.state.runtime_draft_len = runtime_draft_len
        return self._executor._forward_scheduled(
            scheduled_requests,
            resource_manager,
            new_tensors_device=new_tensors_device,
            gather_context_logits=gather_context_logits,
            cache_indirection_buffer=cache_indirection_buffer,
            num_accepted_tokens_device=num_accepted_tokens_device,
            req_id_to_old_request=req_id_to_old_request,
        )

    def warmup(self, resource_manager) -> None:
        self._warmup.warmup(resource_manager)

    def capture_graphs(self, resource_manager) -> None:
        self._warmup.capture_graphs(resource_manager)

    def cleanup(self) -> None:
        self._ctx.attn_metadata = None
        self._ctx.spec_metadata = None
        self._preparer._steady_gen_cache = None
        pool = getattr(self._preparer, "_encoder_decoder_host_buffer_pool", None)
        if pool is not None:
            pool.clear()

    @property
    def moe_load_balancer_iter_info(self):
        moe_load_balancer = self._ctx.deps.moe_load_balancer
        if moe_load_balancer is not None:
            return moe_load_balancer.enable_statistic, moe_load_balancer.enable_update_weights
        return False, False

    @moe_load_balancer_iter_info.setter
    def moe_load_balancer_iter_info(self, value: Tuple[bool, bool]):
        moe_load_balancer = self._ctx.deps.moe_load_balancer
        if moe_load_balancer is not None:
            moe_load_balancer.set_iter_info(
                enable_statistic=value[0], enable_update_weights=value[1]
            )
