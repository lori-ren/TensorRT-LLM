# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decoder family's forward path."""

from __future__ import annotations

import functools
import inspect
import weakref
from typing import Any, Callable, Dict, Optional

import torch._dynamo.config

from tensorrt_llm._utils import is_trace_enabled, nvtx_range, trace_func
from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys

from .....attention.backends.trtllm import TrtllmAttentionMetadata
from .....memory_buffer_utils import with_shared_pool
from .....moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
from .....speculative import (
    get_spec_metadata,
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)
from .....utils import get_model_extra_attrs, get_per_request_prefill_cuda_graph_flag
from ....cuda_graph_runner import CUDAGraphRunner
from ....llm_request import LlmRequest, get_draft_token_length
from ....resource_manager import BaseResourceManager, ResourceManager, ResourceManagerType
from ....sampler import SampleStateTensors
from ....sampler.sampler_common import SampleType
from ....scheduler import ScheduledRequests
from ...metadata import update_spec_metadata
from ..common import get_top_level_model, prepare_multimodal_indices
from .context import DecoderContext
from .prepare import InputPreparer


def _make_single_token_context_graph_batch(
    scheduled_requests: ScheduledRequests,
    is_multimodal_decode_compatible: Optional[Callable[[LlmRequest], bool]] = None,
) -> tuple[ScheduledRequests, frozenset[int]]:
    """Build a decode-shaped graph candidate for final one-token contexts.

    Multimodal rows remain fail-closed unless the engine proves that their one
    remaining prompt token is representable by the existing decode provider.
    """
    if scheduled_requests.num_context_requests == 0:
        return scheduled_requests, frozenset()

    context_requests = scheduled_requests.context_requests_last_chunk
    if scheduled_requests.encoder_requests or scheduled_requests.context_requests_chunking:
        return scheduled_requests, frozenset()

    for request in context_requests:
        if (
            request.context_chunk_size != 1
            or request.context_remaining_length != 1
            or request.context_current_position + 1 != request.py_prompt_len
            or request.py_beam_width != 1
            or get_draft_token_length(request) > 0
            or request.py_is_first_draft
            or request.is_context_only_request
            or request.is_generation_only_request
            or request.py_disaggregated_params is not None
            or request.py_mm_encoder_event is not None
            or (
                request.py_multimodal_data is not None
                and (
                    is_multimodal_decode_compatible is None
                    or not is_multimodal_decode_compatible(request)
                )
            )
        ):
            return scheduled_requests, frozenset()

    for request in scheduled_requests.generation_requests:
        if (
            request.py_beam_width != 1
            or get_draft_token_length(request) > 0
            or request.py_is_first_draft
            or request.py_disaggregated_params is not None
        ):
            return scheduled_requests, frozenset()

    graph_batch = ScheduledRequests()
    graph_batch.generation_requests = list(context_requests) + list(
        scheduled_requests.generation_requests
    )
    graph_batch.paused_requests = list(scheduled_requests.paused_requests)
    promoted_context_request_ids = frozenset(request.py_request_id for request in context_requests)
    return graph_batch, promoted_context_request_ids


class ForwardExecutor:
    """Execute a prepared batch and post-process its outputs."""

    def __init__(self, ctx: DecoderContext, preparer: InputPreparer) -> None:
        self._ctx = ctx
        self._preparer = preparer

    """The decoder family's forward path."""

    def _forward_scheduled(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        new_tensors_device: Optional[SampleStateTensors],
        gather_context_logits: bool,
        cache_indirection_buffer: Optional[torch.Tensor],
        num_accepted_tokens_device: Optional[torch.Tensor],
        req_id_to_old_request: Optional[Dict[int, LlmRequest]],
    ):
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        assert kv_cache_manager is not None, "the legacy runner requires a KV cache manager"
        draft_kv_cache_manager = self._ctx.get_draft_kv_cache_manager(resource_manager)

        attn_metadata = self._ctx.set_up_attn_metadata(kv_cache_manager, draft_kv_cache_manager)
        if isinstance(attn_metadata, TrtllmAttentionMetadata):
            attn_metadata.trtllm_gen_jit_warmup = self._ctx.state.trtllm_gen_jit_warmup
        if self._ctx.state.enable_spec_decode:
            spec_resource_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER
            )
            spec_tree_manager = None
            if spec_resource_manager is not None and hasattr(
                spec_resource_manager, "spec_tree_manager"
            ):
                spec_tree_manager = spec_resource_manager.spec_tree_manager
            spec_metadata = self.set_up_spec_metadata(spec_resource_manager)
            assert spec_metadata is not None
            update_spec_metadata(
                spec_metadata,
                scheduled_requests,
                attn_metadata,
                spec_tree_manager=spec_tree_manager,
                runtime_draft_len=self._ctx.state.runtime_draft_len,
                runtime_tokens_per_gen_step=(
                    self._ctx.get_runtime_tokens_per_gen_step(self._ctx.state.runtime_draft_len)
                ),
                is_draft_model=self._ctx.config.is_draft_model,
                attention_backend=self._ctx.config.attention_backend,
                original_max_draft_len=self._ctx.config.original_max_draft_len,
                original_max_total_draft_tokens=(self._ctx.config.original_max_total_draft_tokens),
                spec_dec_max_total_draft_tokens=(self._ctx.config.spec_dec_max_total_draft_tokens),
            )
        else:
            spec_resource_manager = None
            spec_metadata = None

        moe_load_balancer = self._ctx.config.moe_load_balancer
        graph_requests = scheduled_requests
        promoted_context_request_ids: frozenset[int] = frozenset()
        # Non-linear tree input preparation expands runtime_draft_len to the
        # total tree width after graph selection. Only linear-tree zero-draft
        # iterations can therefore safely reuse a zero-draft graph.
        can_promote_spec_decode = not self._ctx.state.enable_spec_decode or (
            not self._ctx.config.is_draft_model
            and self._ctx.state.runtime_draft_len == 0
            and self._ctx.config.spec_config is not None
            and self._ctx.config.spec_config.is_linear_tree
        )
        # TODO: Generalize these conservative gates as actual-draft, beam, and
        # context-parallel providers for decoder-only LLMs gain support for
        # promoted final-context rows. Each relaxation must preserve whole-batch
        # fallback on graph miss and prove parity with the provider's native
        # q_len=1 path. Encoder-decoder and non-LLM engines remain out of scope.
        if (
            scheduled_requests.num_context_requests > 0
            and self._ctx.state.cuda_graph_runner.enabled
            and can_promote_spec_decode
            and not self._ctx.use_beam_search
            and not self._ctx.is_encoder_decoder
            # PLE owns recurrent n-gram and convolution state. Promoting a
            # fresh final-context row would skip its cache-slot reset.
            and not self.model_uses_ple_recurrent_state
            and self._ctx.deps.mapping.cp_size == 1
        ):
            graph_requests, promoted_context_request_ids = _make_single_token_context_graph_batch(
                scheduled_requests, self.is_final_multimodal_context_decode_compatible
            )

        with self._ctx.state.cuda_graph_runner.pad_batch(
            graph_requests, resource_manager, self._ctx.state.runtime_draft_len
        ) as padded_graph_requests:
            # Callee already no-ops when use_mrope=False, but the Python call /
            # frame setup itself is non-trivial under high concurrency. Gating
            # at the caller avoids that overhead for non-mrope models.
            if self._ctx.use_mrope:
                self._preparer._pad_batch_seed_mrope_delta_cache(padded_graph_requests)

            # Refresh is_all_greedy_sample for the *current* batch BEFORE the
            # CUDA graph key is built below. The key includes this flag to pick
            # the argmax vs advanced-sampling graph variant; populate (inside
            # _prepare_inputs) runs later and fills the matching GPU buffers.
            # Without this pre-scan the key would use the previous iteration's
            # stale value and could replay the advanced graph against
            # unpopulated (greedy) buffers, hanging the run (e.g. MTP nextn>=2).
            if spec_metadata is not None:
                spec_metadata.update_is_all_greedy_sample(padded_graph_requests.all_requests())
                self._sync_group_all_greedy_sample(spec_metadata)

            peft_cache_data_type = None
            if getattr(self, "cuda_graph_lora_manager", None) is not None:
                peft_cache_manager = resource_manager.get_resource_manager(
                    ResourceManagerType.PEFT_CACHE_MANAGER
                )
                peft_cache_data_type = peft_cache_manager.data_type

            use_lora_graph = self.use_lora_cuda_graph(padded_graph_requests)
            maybe_attn_metadata, maybe_spec_metadata, key = (
                self._ctx.state.cuda_graph_runner.maybe_get_cuda_graph(
                    padded_graph_requests,
                    enable_spec_decode=self._ctx.state.enable_spec_decode,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    draft_tokens_cuda=self._ctx.buffers.draft_tokens_cuda
                    if self._ctx.config.is_spec_decode
                    else None,
                    new_tensors_device=new_tensors_device,
                    spec_resource_manager=spec_resource_manager,
                    promoted_context_request_ids=promoted_context_request_ids,
                    peft_cache_data_type=peft_cache_data_type,
                    use_lora_graph=use_lora_graph,
                )
            )

            can_run_graph = key is not None
            if can_run_graph:
                attn_metadata = maybe_attn_metadata
                spec_metadata = maybe_spec_metadata
                execution_requests = padded_graph_requests
                execution_promoted_context_ids = promoted_context_request_ids
            else:
                attn_metadata = self._ctx.state.attn_metadata
                if self._ctx.state.enable_spec_decode:
                    spec_metadata = self._ctx.state.spec_metadata
                else:
                    spec_metadata = None
                execution_requests = scheduled_requests
                execution_promoted_context_ids = frozenset()

            # Stage in-graph sampling now that the batch is settled: the staged
            # scatter width has to match the batch the forward actually runs on,
            # which differs between the graph (padded) and eager (unpadded)
            # branches above. Falling back to eager also means no graph replays,
            # so the tier is dropped and the sampler runs after the forward.
            #
            # Promoted one-token contexts join the graph batch's generation
            # requests, but sample_async is handed the original scheduled batch,
            # which excludes them. Staging a tier here would sample rows that
            # are then discarded and advance those requests' Philox streams an
            # extra time, so keep those steps on the eager path.
            if self._ctx.state.stage_in_graph_sampling is not None:
                staged_sample_type = (
                    key.sample_type
                    if can_run_graph and not execution_promoted_context_ids
                    else SampleType.FULL
                )
                self._ctx.state.stage_in_graph_sampling(execution_requests, staged_sample_type)

            # Fill slot-ID buffer for scatter inside draft loop
            if (
                self._ctx.state.enable_spec_decode
                and spec_tree_manager is not None
                and spec_tree_manager.use_dynamic_tree
                and not self._ctx.config.is_draft_model
            ):
                spec_tree_manager.slot_storage.fill_all_slot_ids(
                    execution_requests.context_requests,
                    execution_requests.generation_requests,
                )

            inputs, gather_ids = self._preparer._prepare_inputs(
                execution_requests,
                kv_cache_manager,
                attn_metadata,
                spec_metadata,
                new_tensors_device,
                cache_indirection_buffer,
                num_accepted_tokens_device,
                req_id_to_old_request,
                resource_manager,
                can_run_graph,
                execution_promoted_context_ids,
                use_lora_graph=use_lora_graph,
            )
            if execution_promoted_context_ids:
                self._ctx.state.iter_states["num_ctx_requests"] = (
                    scheduled_requests.num_context_requests
                )
                self._ctx.state.iter_states["num_ctx_tokens"] = sum(
                    request.context_chunk_size for request in scheduled_requests.context_requests
                )
                self._ctx.state.iter_states["num_generation_tokens"] = (
                    scheduled_requests.num_generation_requests
                )
            self._ctx.state.prepare_inputs_event = torch.cuda.Event()
            self._ctx.state.prepare_inputs_event.record()

            breakable_runner = self._ctx.state.breakable_cuda_graph_runner

            with with_shared_pool(self._ctx.state.cuda_graph_runner.get_graph_pool()):

                def forward_step():
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        return self._forward_step(
                            inputs,
                            gather_ids=gather_ids,
                            gather_context_logits=gather_context_logits,
                        )

                if not can_run_graph:
                    if breakable_runner is not None and breakable_runner.is_capturing:
                        return breakable_runner.capture_model_body(forward_step)

                    num_tokens = inputs["input_ids"].shape[0]
                    can_run_breakable_graph = (
                        breakable_runner is not None
                        and get_per_request_prefill_cuda_graph_flag()
                        and not gather_context_logits
                        and breakable_runner.has_graph(num_tokens)
                    )
                    if can_run_breakable_graph and not breakable_runner.is_warming_up:
                        outputs = breakable_runner.execute(num_tokens, forward_step)
                    else:
                        # real eager or BCG warmup or PCG
                        outputs = forward_step()
                else:
                    needs_capture = self._ctx.state.cuda_graph_runner.needs_capture(key)
                    if needs_capture:

                        def capture_forward_fn(inputs: Dict[str, Any]):
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                return self._forward_step(
                                    inputs,
                                    gather_ids=gather_ids,
                                    gather_context_logits=gather_context_logits,
                                )

                        def capture_postprocess_fn(inputs: Dict[str, Any]):
                            self._preparer._postprocess_inputs(inputs)

                        capture_outputs = self._ctx.state.cuda_graph_runner.capture(
                            key,
                            capture_forward_fn,
                            inputs,
                            enable_spec_decode=self._ctx.state.enable_spec_decode,
                            postprocess_fn=capture_postprocess_fn,
                        )

                    if self._ctx.state.cuda_graph_runner.is_warmup_only:
                        outputs = capture_outputs
                    elif needs_capture:
                        # Refresh attention metadata for the current batch's
                        # draft cache before replaying the captured graph.
                        saved_draft = prepare_attn_metadata_for_draft_replay(
                            attn_metadata, draft_kv_cache_manager
                        )
                        try:
                            outputs = self._ctx.state.cuda_graph_runner.replay(key, inputs)
                        finally:
                            restore_attn_metadata_after_draft_replay(attn_metadata, saved_draft)
                    else:
                        saved_draft = prepare_attn_metadata_for_draft_replay(
                            attn_metadata, draft_kv_cache_manager
                        )
                        try:
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                outputs = self._ctx.state.cuda_graph_runner.replay(key, inputs)
                        finally:
                            restore_attn_metadata_after_draft_replay(attn_metadata, saved_draft)

            if self._ctx.state.forward_pass_callable is not None:
                self._ctx.state.forward_pass_callable()

            self._execute_logit_post_processors(scheduled_requests, outputs)

            return outputs

    @nvtx_range("_forward_step")
    def _forward_step(
        self,
        inputs: Dict[str, Any],
        *,
        gather_ids: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
    ) -> Dict[str, Any]:
        inputs = self._preparer._preprocess_inputs(inputs)
        if inputs.get("spec_metadata", None):
            gather_ids = inputs["spec_metadata"].gather_ids

        # For simplicity, just return all the the logits if we have special gather_ids
        # from speculative decoding.
        outputs = self.model_forward(
            **inputs,
            return_context_logits=gather_ids is not None or gather_context_logits,
        )

        if self._ctx.config.without_logits:
            return outputs

        if isinstance(outputs, dict):
            # If the model returns a dict, get the logits from it. All other keys are kept.
            logits = outputs.get("logits", None)
            # If the logits are not found, no further processing is needed.
            if logits is None:
                return outputs
        else:
            # If the model returns a single tensor, assume it is the logits and wrap it in a dict.
            logits = outputs
            outputs = {"logits": logits}

        # If we have special gather_ids, gather the logits
        if gather_ids is not None:
            outputs["logits"] = logits[gather_ids]

        # Sample at the tail of the forward pass, so that under CUDA graph
        # capture the sampling kernels are recorded as part of this graph. The
        # hook is a no-op unless the sampler staged a graph-capturable tier for
        # this batch.
        #
        # Only the last PP rank runs the LM head. The others get a placeholder
        # that is the right shape but never filled, so a shape check waves it
        # through and they would sample garbage every step -- discarded, but
        # not free. _execute_logit_post_processors skips them for this reason.
        if (
            self._ctx.state.sample_in_graph_callable is not None
            and self._ctx.deps.mapping.is_last_pp_rank()
        ):
            self._ctx.state.sample_in_graph_callable(outputs)

        return outputs

    def model_forward(self, **kwargs):
        attrs = get_model_extra_attrs()
        assert attrs is not None, "Model extra attrs is not set"
        attrs["attention_metadata"] = weakref.ref(kwargs["attn_metadata"])
        attrs.update(self._ctx.deps.model.model_config.extra_attrs)
        attrs["spec_metadata"] = kwargs.get("spec_metadata", None)

        if self._ctx.config.torch_compile_backend is not None:
            # Register aux streams and events to model extra attrs.
            # The streams and events are list which could be updated during compilation.
            attrs["aux_streams"] = weakref.ref(self._ctx.config.backend_num_streams)
            attrs["events"] = weakref.ref(self._ctx.config.torch_compile_backend.events)
            attrs["global_stream"] = torch.cuda.current_stream()

        if is_trace_enabled("TLLM_TRACE_MODEL_FORWARD"):
            return trace_func(self._ctx.deps.model.forward)(**kwargs)
        else:
            return self._ctx.deps.model.forward(**kwargs)

    def _execute_logit_post_processors(self, scheduled_requests: ScheduledRequests, outputs: dict):
        """Apply logit post processors (in-place modify outputs Tensors) if any."""

        if not (self._ctx.deps.mapping.is_last_pp_rank()):
            return

        if not isinstance(outputs, dict) or "logits" not in outputs:
            # TODO: support models that don't return outputs as dict
            return

        logits_tensor = outputs["logits"]

        logits_row_offset = 0
        request_groups = (
            (scheduled_requests.context_requests, True),
            (scheduled_requests.generation_requests, False),
        )

        for requests, is_context_request in request_groups:
            for request in requests:
                if is_context_request:
                    beam_width = 1
                    row_stride = 1
                else:
                    # Generation rows are laid out at the static admission
                    # width, so that is the stride between requests, while
                    # only the leading beam_width rows hold live beams under
                    # a variable beam width array. Advancing the offset by the
                    # narrower width would make every request after the first
                    # rewrite another request's logits rows in place.
                    beam_width = request.get_beam_width_by_iter(for_next_iteration=False)
                    row_stride = request.py_beam_width

                logits_processors = getattr(request, "py_logits_post_processors", None)
                if logits_processors:
                    token_ids = (
                        [request.get_tokens(0)]
                        if is_context_request
                        else [request.get_tokens(beam_idx) for beam_idx in range(beam_width)]
                    )
                    if is_context_request and request.py_orig_prompt_len < len(token_ids[0]):
                        # Skip as we only need to apply logit processor on the last context request
                        logits_row_offset += row_stride
                        continue

                    self._apply_logits_processors(
                        request,
                        logits_processors,
                        logits_tensor,
                        beam_width,
                        token_ids,
                        logits_row_offset,
                    )
                logits_row_offset += row_stride

    @staticmethod
    def _apply_logits_processors(
        request, logits_processors, logits_tensor, beam_width, token_ids, logits_row_offset
    ):
        logits_rows = logits_tensor[logits_row_offset : logits_row_offset + beam_width]
        # Reshape to align w/ the shape used in the TRT backend,
        # so the same logit processors can be used across both backends.
        logits_rows = logits_rows.view(beam_width, 1, -1)
        for lp in logits_processors:
            lp_params = inspect.signature(lp).parameters

            assert 4 <= len(lp_params) <= 5, (
                "Logit post processor signature must match the `LogitsProcessor` interface "
                "defined in `tensorrtllm.sampling_params`."
            )
            lp(request.py_request_id, logits_rows, token_ids, None, None)

    def _sync_group_all_greedy_sample(self, spec_metadata) -> None:
        """All-gather the per-rank greedy flags and store the group AND.

        Why the sampling-path choice must be group-uniform under
        ADP + LM-head TP is documented on the anchor,
        ``SpecMetadata.group_all_greedy_sample``. Local contract: called once
        per iteration, right after ``update_is_all_greedy_sample`` and BEFORE
        the CUDA graph key is built. The gate is pure config (identical on
        every rank), so ranks also agree on whether the exchange happens; the
        gather spans the whole TP group, a superset of any LM-head-TP
        subgroup. A dedicated host all-gather rather than a piggyback on the
        ``all_rank_num_tokens`` exchange, which runs in ``_prepare_inputs`` --
        after the graph key, too late for the key to see the synced value.
        """
        # enable_lm_head_tp_in_adp implies enable_attention_dp (asserted in
        # Mapping.__init__), so ADP needs no separate check here.
        if not (
            self._ctx.deps.mapping.enable_lm_head_tp_in_adp and spec_metadata.use_rejection_sampling
        ):
            return
        local_flag = bool(spec_metadata.is_all_greedy_sample)
        all_flags = self._ctx.deps.dist.tp_allgather_int64([local_flag])[:, 0]
        spec_metadata.group_all_greedy_sample = bool(all_flags.all())
        # Also overwrite the live flag directly: this iteration's scan already
        # ran (update_is_all_greedy_sample just returned) and the CUDA graph
        # key reads the flag next -- the stored override only takes effect on
        # the NEXT rescan (populate), which is after key selection.
        spec_metadata.is_all_greedy_sample = spec_metadata.group_all_greedy_sample

    def _run_batch(self, batch, resource_manager):
        """The family's own forward entry, used by warmup and capture."""
        return self._forward_scheduled(
            batch,
            resource_manager,
            new_tensors_device=None,
            gather_context_logits=False,
            cache_indirection_buffer=None,
            num_accepted_tokens_device=None,
            req_id_to_old_request=None,
        )

    def set_up_spec_metadata(self, spec_resource_manager: Optional[BaseResourceManager]):
        spec_config = self._ctx.config.spec_config if self._ctx.state.enable_spec_decode else None
        # The disaggregated attention-DP overlap path opts into larger metadata
        # buffers. Passing None preserves the established max_num_requests
        # fallback for other configurations, including PP.
        num_seq_slots = (
            self._ctx.config.max_num_seq_slots
            if self._ctx.config.enable_disagg_adp_overlap_headroom
            else None
        )
        if self._ctx.state.spec_metadata is not None:
            return self._ctx.state.spec_metadata
        self._ctx.state.spec_metadata = get_spec_metadata(
            spec_config,
            self._ctx.deps.model.config,
            self._ctx.config.batch_size,
            max_num_tokens=self._ctx.config.decoder_max_num_tokens,
            spec_resource_manager=spec_resource_manager,
            is_draft_model=self._ctx.config.is_draft_model,
            max_seq_len=self._ctx.config.max_seq_len,
            num_seq_slots=num_seq_slots,
        )
        return self._ctx.state.spec_metadata

    def is_final_multimodal_context_decode_compatible(self, request: LlmRequest) -> bool:
        """Return whether the final prompt token uses the decode input path.

        KV reuse has already materialized every preceding prompt token. A
        multimodal final-context row therefore needs its prepared embedding
        only when the one remaining token is itself an MM placeholder. Text
        tokens can use the existing decode provider; MRoPE deltas are seeded
        into the per-sequence cache before graph lookup. An MRoPE request with
        real MM payload remains eager until its delta is available.
        """
        final_prompt_token = request.get_tokens(0)[request.context_current_position]
        _, mm_token_indices = prepare_multimodal_indices(
            [final_prompt_token], model=self._ctx.deps.model
        )
        if mm_token_indices.numel() != 0:
            return False

        multimodal_data = request.py_multimodal_data
        if not self._ctx.use_mrope or not _has_mm_payload_keys(multimodal_data):
            return True
        return CUDAGraphRunner._get_mrope_position_delta(request) is not None

    def use_lora_cuda_graph(self, scheduled_requests: ScheduledRequests) -> bool:
        """
        Determines whether a non-LoRA or LoRA CUDA graph should be used, if
        both are available (cuda_graph_specialize_lora==True).
        """
        if self._ctx.state.cuda_graph_lora_manager is None:
            return False
        # Needed during graph capture to enforce a given mode
        if self._ctx.state.force_lora_graph_for_capture is not None:
            return self._ctx.state.force_lora_graph_for_capture
        if not self._ctx.config.llm_args.lora_config.cuda_graph_specialize_lora:
            return True
        return any(
            request.lora_task_id is not None for request in scheduled_requests.generation_requests
        )

    @functools.cached_property
    def model_uses_ple_recurrent_state(self) -> bool:
        """Detect PLE on text-only and multimodal model wrappers.

        The answer is fixed once the model is loaded, and the CUDA-graph gate
        below consults it on every forward that has context requests.
        """
        top_level_model = get_top_level_model(self._ctx.deps.model)
        if getattr(top_level_model, "has_ple", False):
            return True
        llm = getattr(top_level_model, "llm", None)
        text_model = getattr(llm, "model", llm)
        return bool(getattr(text_model, "has_ple", False))
