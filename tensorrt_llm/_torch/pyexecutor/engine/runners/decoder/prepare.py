# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input preparation for the decoder family."""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch._dynamo.config

from tensorrt_llm._utils import maybe_pin_memory, nvtx_range, prefer_pinned
from tensorrt_llm.inputs.multimodal import (
    MultimodalParams,
    MultimodalRuntimeData,
    check_mm_embed_cumsum_if_needed,
    strip_mm_data_for_generation,
)
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType

from .....attention.backends.interface import AttentionMetadata
from .....attention.backends.trtllm import TrtllmAttentionMetadata
from .....metadata import KVCacheParams
from .....models.modeling_multimodal_mixin import _build_request_multimodal_input
from .....speculative import SpecMetadata, get_num_extra_kv_tokens
from .....speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from .....speculative.spec_sampler_base import SampleStateTensorsSpec
from .....utils import set_per_request_prefill_cuda_graph_flag
from ....kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from ....llm_request import LlmRequest, get_draft_token_length
from ....resource_manager import KVCacheManager, ResourceManager, ResourceManagerType
from ....sampler import SampleStateTensors
from ....scheduler import ScheduledRequests
from ...multimodal import is_multimodal, mm_encoder_cache_enabled
from ..common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_padding_params,
    get_position_id_offset,
    prepare_multimodal_indices,
    set_spec_metadata_all_rank_num_tokens,
    ship_multimodal_indices,
)
from .context import DecoderContext


def _get_context_prompt_lookahead_token(request: LlmRequest, chunk_end: int) -> int:
    """Prompt token immediately following a context chunk. Uses the live C++
    ``mPromptLen``; ``py_prompt_len`` goes stale after a non-recompute preemption.
    """
    if request.is_last_context_chunk:
        return INVALID_PROMPT_LOOKAHEAD_TOKEN
    return request.get_token(0, chunk_end)


class InputPreparer:
    """Turn a scheduled batch into the model's inputs."""

    def __init__(self, ctx: DecoderContext) -> None:
        self._ctx = ctx
        self._steady_gen_cache = None
        self._previous_request_ids: list[int] = []
        self._encoder_decoder_staged_request_ids = None

    @nvtx_range("_prepare_inputs")
    def _prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None,
        req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None,
        resource_manager: Optional[ResourceManager] = None,
        maybe_graph: bool = False,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        use_lora_graph: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        set_per_request_prefill_cuda_graph_flag(False)
        if self._ctx.deps.mapping is not None and "cp_type" in self._ctx.deps.mapping.cp_config:
            cp_type = self._ctx.deps.mapping.cp_config["cp_type"]
            if cp_type in (CpType.HELIX, CpType.ULYSSES):
                # Take the usual route of _prepare_tp_inputs.
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported cp_type {getattr(cp_type, 'name', cp_type)}."
                )

        # Initialize SA state for new requests (MTP+SA, EAGLE3+SA, PARD+SA, etc.)
        has_sa_enhancer = (
            self._ctx.config.spec_config is not None
            and getattr(self._ctx.config.spec_config, "sa_config", None) is not None
        )
        if (
            has_sa_enhancer
            and resource_manager is not None
            and self._ctx.deps.mapping.is_last_pp_rank()
        ):
            from tensorrt_llm._torch.speculative.suffix_automaton import SuffixAutomatonManager

            spec_rm = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER
            )
            sa_manager = None
            if spec_rm is not None:
                if isinstance(spec_rm, SuffixAutomatonManager):
                    sa_manager = spec_rm
                else:
                    sa_manager = getattr(spec_rm, "sa_manager", None)
            if sa_manager is not None:
                for request in scheduled_requests.all_requests():
                    if request.py_request_id not in sa_manager._initialized_requests:
                        sa_manager.add_request(request.py_request_id, request.get_tokens(0))
                        sa_manager._initialized_requests.add(request.py_request_id)

        return self._prepare_tp_inputs(
            scheduled_requests,
            kv_cache_manager,
            attn_metadata,
            spec_metadata,
            new_tensors_device,
            cache_indirection_buffer,
            num_accepted_tokens_device,
            req_id_to_old_request,
            resource_manager,
            maybe_graph,
            promoted_context_request_ids,
            use_lora_graph=use_lora_graph,
        )

    def _prepare_tp_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None,
        req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None,
        resource_manager: Optional[ResourceManager] = None,
        maybe_graph: bool = False,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        use_lora_graph: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        """
        Prepare inputs for Pytorch Model.
        """

        new_tokens_device, new_tokens_lens_device, next_draft_tokens_device = None, None, None
        if new_tensors_device is not None:
            # speculative decoding cases: [batch, 1 + draft_len], others: [batch]
            new_tokens_device = new_tensors_device.new_tokens
            # When using overlap scheduler with speculative decoding, the target model's inputs would be
            #   SampleStateTensorsSpec.
            if isinstance(new_tensors_device, SampleStateTensorsSpec):
                assert self._ctx.state.enable_spec_decode and not self._ctx.config.is_draft_model
                new_tokens_lens_device = new_tensors_device.new_tokens_lens  # [batch]
                next_draft_tokens_device = (
                    new_tensors_device.next_draft_tokens
                )  # [batch, draft_len]

        # Must be before the update of py_batch_idx
        if self._ctx.guided_decoder is not None:
            self._ctx.guided_decoder.add_batch(
                scheduled_requests,
                new_tokens=new_tokens_device,
                runtime_draft_len=self._ctx.state.runtime_draft_len,
            )

        if (
            not promoted_context_request_ids
            and type(attn_metadata) is TrtllmAttentionMetadata
            and self._can_use_input_fast_path(
                scheduled_requests, new_tokens_device, next_draft_tokens_device
            )
        ):
            return self._prepare_inputs_fast(
                scheduled_requests,
                kv_cache_manager,
                attn_metadata,
                new_tokens_device,
                resource_manager,
            )

        self._encoder_decoder_staged_request_ids = None
        if not promoted_context_request_ids and self._can_use_steady_gen_fast_prepare(
            scheduled_requests, new_tokens_device, next_draft_tokens_device, spec_metadata
        ):
            return self._apply_steady_gen_fast_prepare(
                kv_cache_manager, attn_metadata, new_tensors_device, resource_manager
            )
        # Any full pass invalidates the steady-state cache; it is re-recorded
        # at the end of this pass when the batch qualifies.
        self._steady_gen_cache = None

        # Hoist use_mrope to a function-scope local so the per-request /
        # per-context-request mrope branches use LOAD_FAST instead of LOAD_ATTR.
        _use_mrope = self._ctx.use_mrope

        # if new_tensors_device exist, input_ids will only contain new context tokens
        input_ids = []  # per sequence
        sequence_lengths = []  # per sequence
        prompt_lengths = []  # per sequence
        request_ids = []  # per request
        gather_ids = []
        position_ids = []  # per sequence
        num_cached_tokens_per_seq = []  # per sequence
        draft_tokens = []
        draft_lens = []
        gen_request_seq_slots = []  # per generation request
        # One-model rejection: slots of gen requests that produced 0 real draft
        # tokens this step (marked in _handle_dynamic_draft_len); their stale
        # draft_probs rows are one-hot'd after spec_metadata.prepare().
        padding_gen_slots = []
        multimodal_params_list = []
        mrope_position_ids = []  # (start_idx, end_idx, (3,1,L) mrope_pos_ids) per multimodal request
        mrope_delta_write_seq_slots = []
        mrope_delta_read_seq_slots = []
        # Whether any generation request in this batch carries real MRoPE
        # metadata; see the post-loop cleanup below.
        has_gen_mrope_delta = False
        # Extra model-side cache slot reserved for CUDA graph / warmup dummy
        # requests, whose outputs are discarded, and for generation requests
        # that carry no MRoPE metadata at all. The cache is zero-initialized and
        # the write path only ever targets real ``py_seq_slot``s, so this slot
        # permanently reads back a zero delta.
        mrope_dummy_seq_slot = (
            self._ctx.runner_config.max_num_tokens * self._ctx.deps.mapping.pp_size
        )
        num_accepted_draft_tokens = []  # per request
        is_enc_dec = self._ctx.is_encoder_decoder
        cross_encoder_hidden_states: List[torch.Tensor] = []
        cross_encoder_seq_lens: List[int] = []  # new encoder K/V tokens per decoder sequence
        cross_encoder_cached_tokens_per_seq: List[int] = []
        # Variables for updating the inputs of draft model
        # Base values for gather_ids computation
        first_draft_base_gather_ids = []
        # seq_slots to index into num_accepted_tokens_device
        first_draft_seq_slots = []
        # Indices in the num_accepted_draft_tokens list
        first_draft_request_indices = []

        # (start_idx, end_idx, seq_slot) for context requests
        context_input_ids_positions = []
        # (start_idx, end_idx, seq_slot) for first_draft requests
        first_draft_input_ids_positions = []

        context_prompt_lookahead = None
        if spec_metadata is not None and spec_metadata.context_prompt_lookahead_tokens is not None:
            context_prompt_lookahead = []

        def append_cross_attention_state(
            request: LlmRequest, project_encoder_output: bool, repeat: int = 1
        ) -> None:
            if not is_enc_dec:
                return

            encoder_output_len = int(request.encoder_output_len)
            if project_encoder_output:
                encoder_output = getattr(request, "py_encoder_output", None)
                if encoder_output is None:
                    raise RuntimeError(
                        "Decoder context request "
                        f"{request.py_request_id} has no encoder output. "
                        "The encoder iteration must populate "
                        "req.py_encoder_output before the first decoder "
                        "context step."
                    )
                if encoder_output.shape[0] != encoder_output_len:
                    raise RuntimeError(
                        "Decoder context request "
                        f"{request.py_request_id} encoder output length "
                        f"({encoder_output.shape[0]}) does not match "
                        f"encoder_output_len ({encoder_output_len})."
                    )
                cross_encoder_hidden_states.append(encoder_output)
                cross_encoder_seq_lens.append(encoder_output_len)
                cross_encoder_cached_tokens_per_seq.append(0)
                return

            for _ in range(repeat):
                cross_encoder_seq_lens.append(0)
                cross_encoder_cached_tokens_per_seq.append(encoder_output_len)

        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            draft_lens.append(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            if context_prompt_lookahead is not None:
                context_prompt_lookahead.append(
                    _get_context_prompt_lookahead_token(request, end_compute)
                )
            # Fetch only the current chunk. get_tokens(0) marshals the whole
            # O(seq_len) VecTokens into a Python list of boxed ints; chunked
            # prefill re-enters this loop for every chunk of the same prompt, so
            # that is O(L) per chunk = O(L^2/chunk) over the prefill.
            # get_tokens_range copies only [begin, end) -> O(chunk).
            prompt_tokens = request.get_tokens_range(0, begin_compute, end_compute)
            position_ids.extend(range(begin_compute, begin_compute + len(prompt_tokens)))

            # Start offset of this request's (current-chunk) tokens within the
            # flattened input_ids. Recorded on multimodal_params below so models
            # that rewrite token IDs in place write into the request's own span
            # rather than assuming a contiguous multimodal prefix.
            context_start_idx = len(input_ids)
            # Track position for updating the inputs of draft model
            if self._ctx.config.is_draft_model and num_accepted_tokens_device is not None:
                input_ids.extend(prompt_tokens)
                end_idx = len(input_ids)
                slot_idx = req_id_to_old_request[request.py_request_id].py_seq_slot
                context_input_ids_positions.append(
                    (context_start_idx, end_idx - 1, slot_idx)
                )  # end_idx-1 is the last token position
            else:
                input_ids.extend(prompt_tokens)

            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(prompt_tokens))
            num_accepted_draft_tokens.append(len(prompt_tokens) - 1)
            prompt_lengths.append(len(prompt_tokens))
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num - request.py_num_compressed_tokens)
            request.cached_tokens = past_seen_token_num
            append_cross_attention_state(
                request,
                project_encoder_output=not request.py_skip_cross_kv_projection
                and (
                    not getattr(request, "is_dummy", False)
                    or getattr(request, "py_encoder_output", None) is not None
                ),
            )

            # Embed mask is required only for partial iterations (chunked
            # prefill or KV-cache reuse); full-prefill degrades gracefully.
            check_mm_embed_cumsum_if_needed(
                request.py_multimodal_data,
                begin_compute=past_seen_token_num,
                end_compute=end_compute,
                prompt_len=request.get_num_tokens(0),
            )
            mm_data = request.py_multimodal_data or {}
            cumsum = mm_data.get("multimodal_embed_mask_cumsum")
            py_multimodal_runtime = None
            if cumsum is not None:
                py_multimodal_runtime = MultimodalRuntimeData(
                    embed_mask_cumsum=cumsum,
                    past_seen_token_num=past_seen_token_num,
                    chunk_end_pos=end_compute,
                )

            multimodal_params = MultimodalParams(
                multimodal_input=_build_request_multimodal_input(
                    request, self.mm_encoder_cache_enabled
                ),
                multimodal_data=request.py_multimodal_data,
                multimodal_runtime=py_multimodal_runtime,
                mm_item_order=getattr(request, "py_mm_item_order", None),
                input_ids_start_offset=context_start_idx,
            )
            # Transfer any cross-iter MM encoder prefetch event stamped on the request onto the
            # freshly-built MultimodalParams. The downstream consume site reads it from the wrapper,
            # not from the request.
            # NOTE: the prefetch producer always writes the cached embedding into
            # `py_multimodal_data` before stamping the event, so whenever the event is present,
            # `has_content()` below is `True` and the wrapper reaches the consume site that waits on
            # it.
            mm_encoder_event = request.py_mm_encoder_event
            if mm_encoder_event is not None:
                multimodal_params.encoder_event = mm_encoder_event
                request.py_mm_encoder_event = None
            if multimodal_params.has_content():
                # TODO(TRTLLM-14726): Check the persistent MM encoder cache before H2D and avoid
                # transferring raw encoder inputs for full hits in both regular and
                # side-stream-prefetched paths.
                multimodal_params.to_device(
                    "multimodal_data",
                    "cuda",
                    pin_memory=prefer_pinned(),
                    target_keywords=getattr(
                        self._ctx.deps.model, "multimodal_data_device_paths", None
                    ),
                )
                if _use_mrope:
                    # A request may carry multimodal content but no MRoPE
                    # metadata (a text-only prompt whose input processor skips
                    # ``mrope_config``, or a model that does not consume it).
                    # Its per-axis positions are just the scalar positions,
                    # which the (3,1,N) seeding further below already
                    # broadcasts, so leave that span alone.
                    mrope_config = multimodal_params.multimodal_data.get("mrope_config") or {}
                    mrope_pos_ids = mrope_config.get("mrope_position_ids")
                    if mrope_pos_ids is not None:
                        ctx_mrope_position_ids = mrope_pos_ids[
                            :, :, begin_compute : begin_compute + len(prompt_tokens)
                        ]
                        # Record as (start_idx, end_idx, (3,1,L) mrope_pos_ids)
                        mrope_position_ids.append(
                            (
                                len(position_ids) - len(prompt_tokens),
                                len(position_ids),
                                ctx_mrope_position_ids,
                            )
                        )
                    mrope_position_delta = mrope_config.get("mrope_position_deltas")
                    if mrope_position_delta is not None:
                        request.py_mrope_position_delta = mrope_position_delta
                    if mrope_position_delta is not None and request.py_seq_slot is not None:
                        mrope_delta_write_seq_slots.append(request.py_seq_slot)
                        request.py_mrope_delta_cache_slot = request.py_seq_slot

                # re-assign the multimodal_data to the request after to_device for generation requests
                request.py_multimodal_data = multimodal_params.multimodal_data
                multimodal_params_list.append(multimodal_params)

                # Re-register mrope tensors for context-only requests (EPD disaggregated serving).
                # This creates new IPC handles owned by the prefill worker, so the decode worker
                # can access them even after the encode worker's GC deallocates the original memory.
                # Without this, the decode worker would receive handles pointing to freed memory.
                if (
                    request.is_context_only_request
                    and _use_mrope
                    and "mrope_config" in multimodal_params.multimodal_data
                ):
                    mrope_config = multimodal_params.multimodal_data["mrope_config"]
                    _mrope_position_ids = mrope_config.get("mrope_position_ids")
                    _mrope_position_deltas = mrope_config.get("mrope_position_deltas")
                    if _mrope_position_ids is not None and _mrope_position_deltas is not None:
                        # Clone to allocate new memory owned by this (prefill) worker.
                        request.py_result.set_mrope_position(
                            _mrope_position_ids.clone(), _mrope_position_deltas.clone()
                        )

            request.py_batch_idx = request.py_seq_slot

        num_ctx_requests = scheduled_requests.num_context_requests
        num_ctx_tokens = len(input_ids)
        if len(multimodal_params_list) > 0:
            # input_ids holds only context tokens here; extend/draft tokens are
            # appended below and are by construction text, so we reuse the
            # CPU-side text_token_indices and just extend it with the
            # post-context arange instead of recomputing via a bool mask +
            # torch.where over the full range.
            text_token_indices_ctx, mm_token_indices = prepare_multimodal_indices(
                input_ids, model=self._ctx.deps.model
            )
        else:
            text_token_indices_ctx = None
            mm_token_indices = None

        # Requests with draft tokens are treated like extend requests. Dummy extend requests should be
        # at the end of extend_requests.
        extend_requests = []
        extend_dummy_requests = []
        generation_requests = []
        first_draft_requests = []
        # Collect generation request IDs during categorization to avoid
        # a separate iteration over scheduled_requests.generation_requests later.
        all_gen_request_ids = []
        for request in scheduled_requests.generation_requests:
            is_promoted_context = request.py_request_id in promoted_context_request_ids
            if not is_promoted_context:
                all_gen_request_ids.append(request.py_request_id)
            # In speculative iterations, keep promoted rows ahead of existing
            # generation rows in the extend-request packing order. Although
            # their q_len is one, this category provides the "no previous
            # speculative tensor" branch needed to source their prompt token
            # without disturbing the overlap offsets of ordinary generation
            # siblings. Non-speculative promoted rows retain the established
            # ordinary generation path below.
            if is_promoted_context and self._ctx.state.enable_spec_decode:
                extend_requests.append(request)
            elif is_promoted_context:
                generation_requests.append(request)
            elif get_draft_token_length(request) > 0 or next_draft_tokens_device is not None:
                if request.is_dummy:
                    extend_dummy_requests.append(request)
                else:
                    extend_requests.append(request)
            elif request.py_is_first_draft:
                first_draft_requests.append(request)
            else:
                generation_requests.append(request)
        extend_requests += extend_dummy_requests

        spec_config = self._ctx.config.spec_config if self._ctx.state.enable_spec_decode else None
        if not self._ctx.config.disable_overlap_scheduler and spec_config is not None:
            assert spec_config.spec_dec_mode.support_overlap_scheduler(), (
                f"{spec_config.decoding_type} does not support overlap scheduler"
            )

        # For tree decoding, runtime_draft_len should match total tree
        # tokens (not tree depth).  py_executor resets it every iteration.
        if spec_config is not None and not spec_config.is_linear_tree:
            self._ctx.state.runtime_draft_len = self._ctx.config.max_total_draft_tokens

        # will contain previous batch indices of generation requests
        previous_batch_indices = []
        previous_pos_indices = []
        runtime_tokens_per_gen_step = self._ctx.deps.spec.runtime_tokens_per_gen_step(
            self._ctx.state.runtime_draft_len
        )
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        for request in extend_requests:
            is_promoted_context = request.py_request_id in promoted_context_request_ids
            if getattr(request, "py_needs_onehot_draft_probs", False):
                if request.py_seq_slot is not None:
                    padding_gen_slots.append(request.py_seq_slot)
                request.py_needs_onehot_draft_probs = False  # consume once
            request_ids.append(request.py_request_id)
            # the request has no previous tensor:
            # (1) next_draft_tokens_device is None, which means overlap scheduler is disabled; or
            # (2) a dummy request; or
            # (3) the first step in the generation server of disaggregated serving
            if (
                is_promoted_context
                or next_draft_tokens_device is None
                or request.is_dummy
                or request.py_batch_idx is None
            ):
                # get token ids, including input token ids and draft token ids. For these dummy requests,
                # no need to copy the token ids.
                if not (request.is_attention_dp_dummy or request.is_cuda_graph_dummy):
                    if is_promoted_context:
                        input_ids.append(request.get_tokens(0)[request.context_current_position])
                    else:
                        input_ids.append(request.get_last_tokens(0))
                    input_ids.extend(request.py_draft_tokens)
                    draft_tokens.extend(request.py_draft_tokens)
                # get other ids and lengths
                num_draft_tokens = get_draft_token_length(request)
                past_seen_token_num = (
                    request.context_current_position
                    if is_promoted_context
                    else request.max_beam_num_tokens - 1
                )
                draft_lens.append(num_draft_tokens)
                if (
                    self._ctx.state.enable_spec_decode
                    and spec_config.spec_dec_mode.extend_ctx(
                        self._ctx.runner_config.attention_backend
                    )
                    and spec_config.is_linear_tree
                ):
                    # We're treating the prompt lengths as context requests here, so
                    # the the prompt lens should not include the cached tokens.
                    prompt_lengths.append(1 + num_draft_tokens)
                else:
                    prompt_lengths.append(request.py_prompt_len)

                sequence_lengths.append(1 + num_draft_tokens)
                num_accepted_draft_tokens.append(num_draft_tokens)
                gather_ids.extend(
                    list(range(len(position_ids), len(position_ids) + 1 + num_draft_tokens))
                )
                position_ids.extend(
                    list(range(past_seen_token_num, past_seen_token_num + 1 + num_draft_tokens))
                )
                num_cached_tokens_per_seq.append(
                    past_seen_token_num - request.py_num_compressed_tokens
                )
                request.cached_tokens = past_seen_token_num
                # update batch index
                request.py_batch_idx = request.py_seq_slot
            else:
                # update batch index
                previous_batch_idx = request.py_batch_idx
                request.py_batch_idx = request.py_seq_slot

                sequence_lengths.append(runtime_tokens_per_gen_step)
                num_accepted_draft_tokens.append(request.py_num_accepted_draft_tokens)
                past_seen_token_num = request.max_beam_num_tokens - 1

                draft_lens.append(runtime_draft_token_buffer_width)
                gather_ids.extend(
                    list(range(len(position_ids), len(position_ids) + runtime_tokens_per_gen_step))
                )
                position_ids.extend(
                    list(
                        range(
                            past_seen_token_num, past_seen_token_num + runtime_tokens_per_gen_step
                        )
                    )
                )
                # previous tensor
                previous_batch_indices.append(previous_batch_idx)
                previous_pos_indices.extend([previous_batch_idx] * runtime_tokens_per_gen_step)

                num_cached_tokens_per_seq.append(
                    past_seen_token_num
                    + runtime_tokens_per_gen_step
                    - request.py_num_compressed_tokens
                )
                request.cached_tokens = past_seen_token_num + runtime_tokens_per_gen_step
                if (
                    self._ctx.state.enable_spec_decode
                    and spec_config.spec_dec_mode.extend_ctx(
                        self._ctx.runner_config.attention_backend
                    )
                    and spec_config.is_linear_tree
                ):
                    prompt_lengths.append(runtime_tokens_per_gen_step)
                else:
                    prompt_lengths.append(request.py_prompt_len)

            append_cross_attention_state(request, project_encoder_output=False)

        for request in first_draft_requests:
            request_ids.append(request.py_request_id)
            draft_lens.append(0)
            # Only the length and the last (original_max_draft_len+1) tokens are
            # needed here; get_num_tokens is O(1) and get_tokens_range copies only
            # the requested window, whereas get_tokens(0) marshals the whole
            # O(seq_len) VecTokens into a Python list.
            _num_tokens = request.get_num_tokens(0)
            begin_compute = _num_tokens - self._ctx.config.original_max_draft_len - 1
            end_compute = begin_compute + self._ctx.config.original_max_draft_len + 1
            prompt_tokens = request.get_tokens_range(0, begin_compute, end_compute)
            position_ids.extend(range(begin_compute, begin_compute + len(prompt_tokens)))

            # Track position for updating the inputs of draft model
            if self._ctx.config.is_draft_model and num_accepted_tokens_device is not None:
                start_idx = len(input_ids)
                input_ids.extend(prompt_tokens)
                end_idx = len(input_ids)
                # For first_draft, we need to replace the last original_max_draft_len+1 tokens
                slot_idx = req_id_to_old_request[request.py_request_id].py_seq_slot
                first_draft_input_ids_positions.append((start_idx, end_idx, slot_idx))

                # Store info for GPU computation of gather_ids and num_accepted_draft_tokens
                base_gather_id = len(input_ids) - 1 - self._ctx.config.original_max_draft_len
                # Placeholder, will be corrected on GPU
                gather_ids.append(base_gather_id)
                first_draft_base_gather_ids.append(base_gather_id)
                first_draft_seq_slots.append(slot_idx)
                first_draft_request_indices.append(len(num_accepted_draft_tokens))

                # Placeholder, will be corrected on GPU
                num_accepted_draft_tokens.append(0)
            else:
                input_ids.extend(prompt_tokens)
                gather_ids.append(
                    len(input_ids)
                    - 1
                    - (
                        self._ctx.config.original_max_draft_len
                        - request.py_num_accepted_draft_tokens
                    )
                )
                num_accepted_draft_tokens.append(request.py_num_accepted_draft_tokens)

            sequence_lengths.append(1 + self._ctx.config.original_max_draft_len)
            prompt_lengths.append(request.py_prompt_len)
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num - request.py_num_compressed_tokens)
            append_cross_attention_state(request, project_encoder_output=False)

            # update batch index
            request.py_batch_idx = request.py_seq_slot

        helix_is_inactive_rank, helix_position_offsets = [], []
        # Cache invariant method result to avoid repeated calls per-request
        _has_cp_helix = self._ctx.deps.mapping.has_cp_helix()
        _n_gen = len(generation_requests)
        # One-shot batch-level flag — True iff any generation request actually
        # carries multimodal payload. Lets the strip_mm_data branch below
        # short-circuit on a LOAD_FAST rather than a per-request LOAD_ATTR
        # of py_multimodal_data for non-multimodal models (the gpt-oss-120b
        # GEN case).
        _has_any_multimodal_request = any(
            r.py_multimodal_data is not None for r in generation_requests
        )
        if _n_gen > 0:
            # The whole batch is laid out with request 0's beam width: every
            # generation request contributes exactly this many rows to
            # input_ids / position_ids / sequence_lengths and to the logits the
            # model returns. The sampler, in turn, locates a request's logits by
            # accumulating the *per-request* beam widths
            # (TorchSampler._select_generated_logits ->
            # calculate_request_offsets). Both agree only while every request in
            # the batch has the same beam width.
            #
            # Mixing widths would desynchronize the two: the sampler would read
            # a request's rows at the wrong offset, and `logits.view(batch,
            # beam_width_in, vocab)` succeeds for any shape whose element count
            # divides, so the result is silently wrong rather than an error.
            # Supporting mixed widths needs the forward path to emit a fixed
            # max_beam_width stride and the sampler offsets to match; until
            # then, fail loudly.
            beam_width = generation_requests[0].py_beam_width
            # Admission pins every request to max_beam_width, but a
            # variable-beam-width request narrows or widens per iteration, so
            # the widths can still diverge mid-batch. Compare the
            # *per-iteration* width: py_beam_width is fixed at admission and
            # would be identical across those requests. Dummy requests are
            # excluded -- they carry no user request and are built at their own
            # width (CUDA-graph padding at the engine width, attention-DP and
            # warmup dummies at width one), so they would otherwise trip this
            # on an ordinary padded batch.
            real_requests = [req for req in generation_requests if not req.is_dummy]
            iter_widths = {req.get_beam_width_by_iter() for req in real_requests}
            if len(iter_widths) > 1:
                # NB: this aborts the whole batch, not just the offending
                # requests -- ModelEngine has no per-request failure channel,
                # and by this point the batch is already scheduled. Scoping the
                # failure needs the scheduler to group by beam width in the
                # first place, so that no such batch is formed; TRTLLM-14792.
                raise ValueError(
                    "Generation requests in one batch must all have the same "
                    f"beam width; got {sorted(iter_widths)}. Mixed beam widths "
                    "within a batch are not supported yet (TRTLLM-14792)."
                )

            # Pre-extend constant-value lists to avoid per-request append
            # overhead (saves ~3 append calls per request).
            draft_lens.extend([0] * (_n_gen * beam_width))
            sequence_lengths.extend([1] * (_n_gen * beam_width))
            num_accepted_draft_tokens.extend([0] * (_n_gen * beam_width))

            for request in generation_requests:
                request_ids.append(request.py_request_id)
                is_promoted_context = request.py_request_id in promoted_context_request_ids
                if is_promoted_context:
                    input_ids.append(request.get_tokens(0)[request.context_current_position])
                    past_seen_token_num = request.context_current_position
                    request_has_previous_tensor = False
                # The request has no previous tensor:
                # (1) new_tokens_device is None, which means overlap scheduler is disabled; or
                # (2) a dummy request; or
                # (3) the first step in the generation server of disaggregated serving.
                elif new_tokens_device is None or request.is_dummy or request.py_batch_idx is None:
                    # skip adding input_ids of CUDA graph dummy requests so that new_tokens_device
                    # can be aligned to the correct positions.
                    if not request.is_cuda_graph_dummy:
                        for beam in range(beam_width):
                            # Track position for GPU update (draft model only)
                            if (
                                self._ctx.config.is_draft_model
                                and num_accepted_tokens_device is not None
                            ):
                                start_idx = len(input_ids)
                                input_ids.append(request.get_last_tokens(beam))
                                end_idx = len(input_ids)
                                slot_idx = req_id_to_old_request[request.py_request_id].py_seq_slot
                                first_draft_input_ids_positions.append(
                                    (start_idx, end_idx, slot_idx)
                                )
                            else:
                                input_ids.append(request.get_last_tokens(beam))
                    past_seen_token_num = request.max_beam_num_tokens - 1
                    request_has_previous_tensor = False
                else:
                    # the request has previous tensor
                    # previous_batch_indices is per-request, not per-beam
                    previous_batch_indices.append(request.py_batch_idx)
                    past_seen_token_num = request.max_beam_num_tokens
                    request_has_previous_tensor = True

                position_id = past_seen_token_num
                if _has_cp_helix:
                    # We compute a global position_id because each helix rank has only a subset of
                    # tokens for a sequence.
                    position_id = request.total_input_len_cp + request.py_decoding_iter - 1
                    if request_has_previous_tensor:
                        # With the overlap scheduler this batch is prepared
                        # before the previous iteration's _update_requests has
                        # advanced py_decoding_iter, so the counter is one
                        # behind. Compensate exactly like the non-helix path
                        # above, which uses max_beam_num_tokens *without* the
                        # -1 in this case. Without this, the position repeats
                        # once (L, L, L+1, ...) and the new token's K is roped
                        # at the wrong position before being written to the KV
                        # cache, corrupting every later step.
                        # TODO: revisit for helix x speculative decoding -
                        # the base formula and this +1 both assume exactly
                        # one new token per step (draft-token modes are
                        # currently rejected under helix).
                        position_id += 1
                    if request.py_helix_is_inactive_rank:
                        past_seen_token_num = request.seqlen_this_rank_cp
                    else:
                        # Discount the token added to active rank in resource manager as it hasn't
                        # been previously seen.
                        past_seen_token_num = request.seqlen_this_rank_cp - 1

                    for beam in range(beam_width):
                        # Update helix-specific parameters.
                        helix_is_inactive_rank.append(request.py_helix_is_inactive_rank)
                        helix_position_offsets.append(position_id)

                request.cached_tokens = past_seen_token_num
                for beam in range(beam_width):
                    position_ids.append(position_id)
                    num_cached_tokens_per_seq.append(
                        past_seen_token_num - request.py_num_compressed_tokens
                    )
                    prompt_lengths.append(request.py_prompt_len)
                    gather_ids.append(len(position_ids) - 1)

                if _use_mrope:
                    mrope_position_delta = getattr(request, "py_mrope_position_delta", None)
                    if mrope_position_delta is None and request.py_multimodal_data:
                        mrope_config = request.py_multimodal_data.get("mrope_config") or {}
                        mrope_position_delta = mrope_config.get("mrope_position_deltas")
                        if mrope_position_delta is not None:
                            if mrope_position_delta.device.type == "cpu":
                                mrope_position_delta = maybe_pin_memory(mrope_position_delta).to(
                                    device="cuda", dtype=torch.int32, non_blocking=True
                                )
                                mrope_config["mrope_position_deltas"] = mrope_position_delta
                            request.py_mrope_position_delta = mrope_position_delta
                    if mrope_position_delta is not None:
                        has_gen_mrope_delta = True
                        # NOTE: Expanding position_ids to 3D tensor who is using mrope
                        gen_mrope_position_ids = (
                            past_seen_token_num + mrope_position_delta
                        ).expand(3, 1, 1)
                        update_mrope_delta = (
                            request.py_seq_slot is not None
                            and not request.is_dummy
                            and getattr(request, "py_mrope_delta_cache_slot", None)
                            != request.py_seq_slot
                        )
                        delta_read_seq_slot = (
                            mrope_dummy_seq_slot
                            if request.is_dummy or request.py_seq_slot is None
                            else request.py_seq_slot
                        )
                        if update_mrope_delta:
                            multimodal_params = MultimodalParams(
                                multimodal_data={
                                    "mrope_config": {"mrope_position_deltas": mrope_position_delta}
                                }
                            )
                            mrope_delta_write_seq_slots.append(request.py_seq_slot)
                            multimodal_params_list.append(multimodal_params)
                            request.py_mrope_delta_cache_slot = request.py_seq_slot
                        for beam in range(beam_width):
                            # Locate this beam's single token in the flat array.
                            token_start = len(position_ids) - beam_width + beam
                            mrope_position_ids.append(
                                (token_start, token_start + 1, gen_mrope_position_ids)
                            )
                            mrope_delta_read_seq_slots.append(delta_read_seq_slot)
                    else:
                        # No MRoPE metadata for this request (text-only prompt
                        # on an MRoPE model): its delta is zero by construction,
                        # so read the reserved zero slot instead of skipping the
                        # append. The kernel indexes ``mrope_position_deltas``
                        # by *generation batch index*
                        # (decoderMaskedMultiheadAttentionTemplate.h), so a list
                        # that is sparse w.r.t. the generation batch would
                        # silently shift every later request onto another
                        # request's delta. No ``mrope_position_ids`` span is
                        # recorded: the broadcast scalar position is already
                        # this request's answer on all three axes.
                        for _ in range(beam_width):
                            mrope_delta_read_seq_slots.append(mrope_dummy_seq_slot)
                # Equivalent to the original `is_generation_admission and
                # request.py_multimodal_data`. The batch-level flag is checked
                # first so non-multimodal models pay one LOAD_FAST per request
                # instead of LOAD_ATTR(py_multimodal_data) + LOAD_ATTR(py_batch_idx).
                if (
                    _has_any_multimodal_request
                    and request.py_multimodal_data
                    and request.py_batch_idx is None
                ):
                    strip_mm_data_for_generation(request.py_multimodal_data)

                request.py_batch_idx = request.py_seq_slot
                append_cross_attention_state(
                    request, project_encoder_output=False, repeat=beam_width
                )
                # Do not add a gen_request_seq_slot for CUDA graph dummy requests
                # to prevent access errors due to None values
                if not request.is_cuda_graph_dummy:
                    gen_request_seq_slots.append(request.py_seq_slot)

        if _use_mrope and not has_gen_mrope_delta:
            # Every generation request in this batch resolved to the zero slot,
            # so the gathered deltas would be an all-zero vector -- identical to
            # passing no deltas at all. Dropping the list keeps the steady-state
            # generation fast path (which requires the mrope lists to be empty)
            # reachable for text-only batches on MRoPE models.
            mrope_delta_read_seq_slots.clear()

        previous_batch_len = len(previous_batch_indices)

        def previous_seq_slots_device():
            previous_batch_indices_host = torch.tensor(
                previous_batch_indices, dtype=torch.int, pin_memory=prefer_pinned()
            )
            previous_slots = self._ctx.buffers.previous_batch_indices_cuda[:previous_batch_len]
            previous_slots.copy_(previous_batch_indices_host, non_blocking=True)
            return previous_slots

        num_tokens = len(input_ids)
        num_draft_tokens = len(draft_tokens)
        total_num_tokens = len(position_ids)
        max_num_tokens = self._ctx.runner_config.max_num_tokens
        assert total_num_tokens <= max_num_tokens, (
            f"total_num_tokens ({total_num_tokens}) should be less than or "
            f"equal to max_num_tokens ({max_num_tokens})"
        )
        # if exist requests that do not have previous batch, copy input_ids and draft_tokens
        if num_tokens > 0:
            input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=prefer_pinned())
            self._ctx.buffers.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

            # Update input_ids_cuda with new tokens from new_tensors_device (draft model only)
            if self._ctx.config.is_draft_model and num_accepted_tokens_device is not None:
                # For context requests: replace the last token with new_tensors_device[0, seq_slot, 0]
                if len(context_input_ids_positions) > 0:
                    # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                    num_ctx_positions = len(context_input_ids_positions)
                    ctx_token_indices_cpu = torch.tensor(
                        [last_token_idx for _, last_token_idx, _ in context_input_ids_positions],
                        dtype=torch.long,
                        pin_memory=prefer_pinned(),
                    )
                    ctx_seq_slots_cpu = torch.tensor(
                        [seq_slot for _, _, seq_slot in context_input_ids_positions],
                        dtype=torch.long,
                        pin_memory=prefer_pinned(),
                    )
                    # Copy to pre-allocated GPU buffers
                    self._ctx.buffers.draft_ctx_token_indices_cuda[:num_ctx_positions].copy_(
                        ctx_token_indices_cpu, non_blocking=True
                    )
                    self._ctx.buffers.draft_ctx_seq_slots_cuda[:num_ctx_positions].copy_(
                        ctx_seq_slots_cpu, non_blocking=True
                    )
                    self._ctx.buffers.input_ids_cuda[
                        self._ctx.buffers.draft_ctx_token_indices_cuda[:num_ctx_positions]
                    ] = new_tensors_device.new_tokens[
                        0, self._ctx.buffers.draft_ctx_seq_slots_cuda[:num_ctx_positions], 0
                    ]

                # For first_draft requests: replace the last (original_max_draft_len+1) tokens
                # with new_tensors_device[:, seq_slot, 0]
                if len(first_draft_input_ids_positions) > 0:
                    # All first_draft requests have same token length (original_max_draft_len + 1)
                    # Build index tensors on CPU first, then copy to GPU to avoid implicit sync
                    num_requests = len(first_draft_input_ids_positions)
                    tokens_per_request = (
                        first_draft_input_ids_positions[0][1]
                        - first_draft_input_ids_positions[0][0]
                    )

                    # Create flat index array for all tokens to update on CPU
                    all_indices = []
                    all_seq_slots = []
                    for start_idx, end_idx, seq_slot in first_draft_input_ids_positions:
                        all_indices.extend(range(start_idx, end_idx))
                        all_seq_slots.extend([seq_slot] * (end_idx - start_idx))

                    # Create CPU tensors with pinned memory
                    total_tokens = len(all_indices)
                    idx_tensor_cpu = torch.tensor(
                        all_indices, dtype=torch.long, pin_memory=prefer_pinned()
                    )
                    seq_slots_tensor_cpu = torch.tensor(
                        all_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
                    )

                    # Copy to pre-allocated GPU buffers
                    self._ctx.buffers.draft_first_draft_indices_cuda[:total_tokens].copy_(
                        idx_tensor_cpu, non_blocking=True
                    )
                    self._ctx.buffers.draft_first_draft_seq_slots_cuda[:total_tokens].copy_(
                        seq_slots_tensor_cpu, non_blocking=True
                    )

                    # Create token position indices (repeating 0..tokens_per_request for each request)
                    token_positions = torch.arange(
                        tokens_per_request, dtype=torch.long, device="cuda"
                    ).repeat(num_requests)

                    self._ctx.buffers.input_ids_cuda[
                        self._ctx.buffers.draft_first_draft_indices_cuda[:total_tokens]
                    ] = new_tensors_device.new_tokens[
                        token_positions,
                        self._ctx.buffers.draft_first_draft_seq_slots_cuda[:total_tokens],
                        0,
                    ]

        if num_draft_tokens > 0:
            draft_tokens = torch.tensor(draft_tokens, dtype=torch.int, pin_memory=prefer_pinned())
            self._ctx.buffers.draft_tokens_cuda[: len(draft_tokens)].copy_(
                draft_tokens, non_blocking=True
            )
        if self._ctx.config.spec_config is not None and len(num_accepted_draft_tokens) > 0:
            num_accepted_draft_tokens = torch.tensor(
                num_accepted_draft_tokens, dtype=torch.int, pin_memory=prefer_pinned()
            )
            self._ctx.buffers.num_accepted_draft_tokens_cuda[
                : len(num_accepted_draft_tokens)
            ].copy_(num_accepted_draft_tokens, non_blocking=True)

            # Update num_accepted_draft_tokens_cuda for first_draft_requests directly from num_accepted_tokens_device
            #   (draft model only)
            if self._ctx.config.is_draft_model and len(first_draft_seq_slots) > 0:
                # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                num_first_draft = len(first_draft_seq_slots)
                first_draft_seq_slots_cpu = torch.tensor(
                    first_draft_seq_slots, dtype=torch.int, pin_memory=prefer_pinned()
                )
                first_draft_indices_cpu = torch.tensor(
                    first_draft_request_indices, dtype=torch.int, pin_memory=prefer_pinned()
                )

                # Copy to pre-allocated GPU buffers
                self._ctx.buffers.draft_seq_slots_buffer_cuda[:num_first_draft].copy_(
                    first_draft_seq_slots_cpu, non_blocking=True
                )
                self._ctx.buffers.draft_request_indices_buffer_cuda[:num_first_draft].copy_(
                    first_draft_indices_cpu, non_blocking=True
                )

                # Extract accepted tokens for first_draft requests from device tensor
                accepted_tokens = num_accepted_tokens_device[
                    self._ctx.buffers.draft_seq_slots_buffer_cuda[:num_first_draft]
                ]
                # Update the correct positions in num_accepted_draft_tokens_cuda
                self._ctx.buffers.num_accepted_draft_tokens_cuda[
                    self._ctx.buffers.draft_request_indices_buffer_cuda[:num_first_draft]
                ] = accepted_tokens
        if next_draft_tokens_device is not None:
            # Initialize these two values to zeros
            self._ctx.buffers.previous_pos_id_offsets_cuda *= 0
            self._ctx.buffers.previous_kv_lens_offsets_cuda *= 0
            runtime_tokens_per_gen_step = self._ctx.deps.spec.runtime_tokens_per_gen_step(
                self._ctx.state.runtime_draft_len
            )
            runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1

            if previous_batch_len > 0:
                previous_slots = previous_seq_slots_device()
                # previous input ids
                previous_batch_tokens = previous_batch_len * runtime_tokens_per_gen_step
                new_tokens = new_tokens_device.transpose(0, 1)[
                    previous_slots, :runtime_tokens_per_gen_step
                ].flatten()
                self._ctx.buffers.input_ids_cuda[
                    num_tokens : num_tokens + previous_batch_tokens
                ].copy_(new_tokens, non_blocking=True)

                # previous draft tokens
                previous_batch_draft_tokens = previous_batch_len * runtime_draft_token_buffer_width
                if runtime_draft_token_buffer_width > 0:
                    self._ctx.buffers.draft_tokens_cuda[
                        num_draft_tokens : num_draft_tokens + previous_batch_draft_tokens
                    ].copy_(
                        next_draft_tokens_device[
                            previous_slots, :runtime_draft_token_buffer_width
                        ].flatten(),
                        non_blocking=True,
                    )
                # prepare data for the preprocess inputs
                kv_len_offsets_device = new_tokens_lens_device - runtime_tokens_per_gen_step
                previous_pos_indices_host = torch.tensor(
                    previous_pos_indices, dtype=torch.int, pin_memory=prefer_pinned()
                )
                self._ctx.buffers.previous_pos_indices_cuda[0:previous_batch_tokens].copy_(
                    previous_pos_indices_host, non_blocking=True
                )

                # The order of requests in a batch: [context requests, generation requests]
                # generation requests: ['requests that do not have previous batch', 'requests that already have previous
                #   batch', 'dummy requests']
                # 1) 'requests that do not have previous batch': disable overlap scheduler or the first step in the
                #   generation server of disaggregated serving.
                #   2) 'requests that already have previous batch': previous iteration's requests.
                #   3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                # Therefore, both of previous_pos_id_offsets_cuda and previous_kv_lens_offsets_cuda are also 3 segments.
                # For 1) 'requests that do not have previous batch': disable overlap scheduler or the first step in the
                #   generation server of disaggregated serving.
                # Set these requests' previous_pos_id_offsets and previous_kv_lens_offsets to '0' to skip the value
                #   changes in _preprocess_inputs.
                #       Already set to '0' during initialization.
                #   For 2) 'requests that already have previous batch': enable overlap scheduler.
                # Set their previous_pos_id_offsets and previous_kv_lens_offsets according to new_tokens_lens_device and
                #   kv_len_offsets_device.
                #   For 3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                #       Already set to '0' during initialization.

                num_extend_reqeust_wo_dummy = len(extend_requests) - len(extend_dummy_requests)
                self._ctx.buffers.previous_pos_id_offsets_cuda[
                    (num_extend_reqeust_wo_dummy - previous_batch_len)
                    * runtime_tokens_per_gen_step : num_extend_reqeust_wo_dummy
                    * runtime_tokens_per_gen_step
                ].copy_(
                    new_tokens_lens_device[
                        self._ctx.buffers.previous_pos_indices_cuda[0:previous_batch_tokens]
                    ],
                    non_blocking=True,
                )

                self._ctx.buffers.previous_kv_lens_offsets_cuda[
                    num_extend_reqeust_wo_dummy - previous_batch_len : num_extend_reqeust_wo_dummy
                ].copy_(kv_len_offsets_device[previous_slots], non_blocking=True)

        elif new_tokens_device is not None:
            seq_slots_device = previous_seq_slots_device()
            max_draft_len = max(draft_lens)
            new_tokens = new_tokens_device[
                : max_draft_len + 1, seq_slots_device, : self._ctx.runner_config.max_beam_width
            ]
            self._ctx.buffers.input_ids_cuda[
                num_tokens : num_tokens
                + previous_batch_len * self._ctx.runner_config.max_beam_width
            ].copy_(new_tokens.flatten(), non_blocking=True)

        if (
            not self._ctx.config.disable_overlap_scheduler
            and next_draft_tokens_device is None
            and len(extend_requests) > 0
        ):
            # During warmup, for those generation requests, we don't have previous tensors,
            # so we need to set the previous_pos_id_offsets and previous_kv_lens_offsets to zeros
            # to skip the value changes in _preprocess_inputs. Otherwise, there will be illegal memory access
            # when writing key/values to the KV cache.
            self._ctx.buffers.previous_pos_id_offsets_cuda *= 0
            self._ctx.buffers.previous_kv_lens_offsets_cuda *= 0

        position_ids = apply_position_id_offset(position_ids, model=self._ctx.deps.model)
        host_position_ids = torch.tensor(position_ids, dtype=torch.int, pin_memory=prefer_pinned())
        # Use the (3,1,N) MRoPE layout whenever the model declares MRoPE, even
        # for text-only batches: keeping position_ids rank-consistent between
        # warmup and serving keeps torch.compile guards stable, so piecewise
        # CUDA graphs captured at warmup remain usable at runtime.
        if self._ctx.use_mrope:
            # Mixed batches may have only some requests with multimodal MRoPE
            # data. Seed the full (3,1,N) buffer from scalar position_ids
            # (text-only tokens get the same value on all 3 axes), then
            # overwrite only the multimodal spans with their real MRoPE coords.
            self._ctx.buffers.position_ids_cuda[:total_num_tokens].copy_(
                host_position_ids, non_blocking=True
            )
            # Broadcast [N] to [3,1,N]: default for text-only tokens.
            self._ctx.buffers.mrope_position_ids_cuda[:, :, :total_num_tokens].copy_(
                self._ctx.buffers.position_ids_cuda[:total_num_tokens]
                .view(1, 1, -1)
                .expand(3, 1, -1),
                non_blocking=True,
            )
            # Overwrite multimodal spans with per-axis MRoPE positions.
            for start_idx, end_idx, segment in mrope_position_ids:
                if segment.ndim != 3:
                    raise RuntimeError(
                        f"Expected 3D mrope_position_ids, got shape {tuple(segment.shape)}"
                    )
                if segment.shape[0] != 3 and segment.shape[-1] == 3:
                    logger.warning(
                        "Transposing unexpected mrope_position_ids shape from "
                        f"{tuple(segment.shape)}"
                    )
                    segment = segment.transpose(0, 2).contiguous()
                if segment.shape[:2] != (3, 1):
                    raise RuntimeError(
                        f"Unexpected mrope_position_ids shape {tuple(segment.shape)} for span {start_idx}:{end_idx}"
                    )
                segment = segment.contiguous()
                if segment.device.type == "cpu":
                    segment = maybe_pin_memory(segment)
                self._ctx.buffers.mrope_position_ids_cuda[:, :, start_idx:end_idx].copy_(
                    segment[:, :, : end_idx - start_idx], non_blocking=True
                )
            final_position_ids = self._ctx.buffers.mrope_position_ids_cuda[:, :, :total_num_tokens]
        else:
            self._ctx.buffers.position_ids_cuda[:total_num_tokens].copy_(
                host_position_ids, non_blocking=True
            )
            final_position_ids = self._ctx.buffers.position_ids_cuda[:total_num_tokens].unsqueeze(0)

        if self._ctx.state.enable_spec_decode:
            self._ctx.buffers.gather_ids_cuda[: len(gather_ids)].copy_(
                torch.tensor(gather_ids, dtype=torch.int, pin_memory=prefer_pinned()),
                non_blocking=True,
            )

            # Update gather_ids for first_draft_requests on GPU (draft model only)
            if self._ctx.config.is_draft_model and len(first_draft_seq_slots) > 0:
                # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                num_first_draft = len(first_draft_seq_slots)
                first_draft_seq_slots_cpu = torch.tensor(
                    first_draft_seq_slots, dtype=torch.int, pin_memory=prefer_pinned()
                )
                first_draft_indices_cpu = torch.tensor(
                    first_draft_request_indices, dtype=torch.int, pin_memory=prefer_pinned()
                )

                # Copy to pre-allocated GPU buffers
                self._ctx.buffers.draft_seq_slots_buffer_cuda[:num_first_draft].copy_(
                    first_draft_seq_slots_cpu, non_blocking=True
                )
                self._ctx.buffers.draft_request_indices_buffer_cuda[:num_first_draft].copy_(
                    first_draft_indices_cpu, non_blocking=True
                )

                # Extract accepted tokens for first_draft requests from device tensor
                accepted_tokens = num_accepted_tokens_device[
                    self._ctx.buffers.draft_seq_slots_buffer_cuda[:num_first_draft]
                ]
                # Update gather_ids: gather_id = base_gather_id + num_accepted_tokens
                # (since gather_id = len(input_ids) - 1 - (max_draft_len - num_accepted))
                self._ctx.buffers.gather_ids_cuda[
                    self._ctx.buffers.draft_request_indices_buffer_cuda[:num_first_draft]
                ] += accepted_tokens

        if self._ctx.deps.mapping.has_cp_helix():
            attn_metadata.update_helix_param(
                helix_position_offsets=helix_position_offsets,
                helix_is_inactive_rank=helix_is_inactive_rank,
            )

        if not attn_metadata.is_cuda_graph:
            # Assumes seq lens do not change between CUDA graph invocations. This applies
            # to draft sequences too. This means that all draft sequences must be padded.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=prefer_pinned(),
            )

        num_generation_requests = len(gen_request_seq_slots)
        # Cache indirection is only used for beam search on generation requests
        if self._ctx.use_beam_search and num_generation_requests > 0:
            if cache_indirection_buffer is not None:
                # Copy cache indirection to local buffer with offsets changing:  seq_slots[i] -> i
                # Convert to GPU tensor to avoid implicit sync
                gen_request_seq_slots_tensor = torch.tensor(
                    gen_request_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
                ).to(device="cuda", non_blocking=True)
                self._ctx.config.cache_indirection_attention[:num_generation_requests].copy_(
                    cache_indirection_buffer[gen_request_seq_slots_tensor]
                )
            if cache_indirection_buffer is not None or self._ctx.is_warmup:
                attn_metadata.beam_width = self._ctx.runner_config.max_beam_width
        else:
            attn_metadata.beam_width = 1

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = scheduled_requests.num_context_requests
        # Use num_chunked_ctx_requests to record the number of extend context requests,
        # so that we can update the kv_lens_cuda correctly in _preprocess_inputs.
        attn_metadata.num_chunked_ctx_requests = 0
        if (
            self._ctx.state.enable_spec_decode
            and spec_config.spec_dec_mode.extend_ctx(self._ctx.runner_config.attention_backend)
            and spec_config.is_linear_tree
        ):
            # For the tree decoding, we want to use XQA to process the draft tokens for the target model.
            # Therefore, we do not treat them as the chunked context requests.
            attn_metadata.num_contexts += len(extend_requests)
            attn_metadata.num_chunked_ctx_requests = len(extend_requests)

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec_config),
            use_full_generation_page_table=(
                self.should_use_full_generation_page_table(spec_config, attn_metadata)
            ),
        )
        attn_metadata.kv_cache_manager = kv_cache_manager

        if hasattr(self._ctx.deps.model.model_config.pretrained_config, "chunk_size"):
            attn_metadata.mamba_chunk_size = (
                self._ctx.deps.model.model_config.pretrained_config.chunk_size
            )
        # Some sparse backends (RocketKV) clamp
        # kv_cache_params.num_cached_tokens_per_seq in place during prepare(),
        # and KVCacheParams holds the list by reference. Snapshot the true
        # pre-prepare counts so the steady-gen recording below stores values
        # that the per-step prepare() can re-clamp from scratch.
        num_cached_tokens_snapshot = list(num_cached_tokens_per_seq)
        attn_metadata.prepare()
        cross_attention_inputs = (
            self._prepare_cross_attn_inputs(
                cross_encoder_hidden_states,
                cross_encoder_seq_lens,
                cross_encoder_cached_tokens_per_seq,
                attn_metadata,
                resource_manager,
            )
            if is_enc_dec
            else {}
        )

        peft_cache_manager = resource_manager and resource_manager.get_resource_manager(
            ResourceManagerType.PEFT_CACHE_MANAGER
        )
        lora_params = self._ctx.deps.lora.build(
            scheduled_requests,
            attn_metadata,
            cuda_graph_lora_manager=self._ctx.state.cuda_graph_lora_manager,
            enable_spec_decode=self._ctx.state.enable_spec_decode,
            runtime_draft_len=self._ctx.state.runtime_draft_len,
            peft_cache_manager=peft_cache_manager,
            maybe_graph=maybe_graph,
            use_lora_graph=use_lora_graph,
        )

        if spec_metadata is not None:
            # Set the per-batch counts here, before the attention-DP allgather
            # below: the allgather and prepare() must derive the DP token count
            # from the same fields (see SpecMetadata.dp_num_tokens). Use
            # scheduled_requests.num_generation_requests -- the same-named
            # local above excludes CUDA-graph dummies and would not match the
            # count prepare() uses.
            spec_metadata.num_tokens = total_num_tokens
            spec_metadata.num_generations = scheduled_requests.num_generation_requests
            spec_metadata.seq_lens = sequence_lengths

        spec_all_rank_counts = None
        if spec_metadata is not None and self._ctx.config.enable_attention_dp:
            (attn_all_rank_num_tokens, spec_all_rank_counts) = (
                self._get_all_rank_num_tokens_and_spec_counts(attn_metadata, spec_metadata)
            )
        else:
            attn_all_rank_num_tokens = get_all_rank_num_tokens(
                attn_metadata,
                enable_attention_dp=self._ctx.config.enable_attention_dp,
                mapping=self._ctx.deps.mapping,
                dist=self._ctx.deps.dist,
            )
        (padded_num_tokens, can_run_prefill_cuda_graph, attn_all_rank_num_tokens) = (
            get_padding_params(
                total_num_tokens,
                num_ctx_requests,
                attn_all_rank_num_tokens,
                dist=self._ctx.deps.dist,
                enable_attention_dp=self._ctx.config.enable_attention_dp,
                prefill_cuda_graph_backend=self._ctx.config.prefill_cuda_graph_backend,
                prefill_cuda_graph_num_tokens=self._ctx.config.prefill_cuda_graph_num_tokens,
            )
        )
        set_per_request_prefill_cuda_graph_flag(can_run_prefill_cuda_graph)
        attn_metadata.padded_num_tokens = (
            padded_num_tokens if padded_num_tokens != total_num_tokens else None
        )

        virtual_num_tokens = total_num_tokens
        if attn_metadata.padded_num_tokens is not None:
            self._ctx.buffers.input_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens
            # Match the rank of the unpadded branch: MRoPE models always use
            # the (3,1,N) layout (see the seeding block above), so the padded
            # view must stay 3D as well to keep torch.compile guards stable.
            if self._ctx.use_mrope:
                # Zero-fill padding on dim 2 (token dim) of (3,1,N) buffer.
                self._ctx.buffers.mrope_position_ids_cuda[
                    :, :, total_num_tokens:padded_num_tokens
                ].fill_(0)
                final_position_ids = self._ctx.buffers.mrope_position_ids_cuda[
                    :, :, :virtual_num_tokens
                ]
            else:
                self._ctx.buffers.position_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
                final_position_ids = self._ctx.buffers.position_ids_cuda[
                    :virtual_num_tokens
                ].unsqueeze(0)

        if self._ctx.config.enable_attention_dp:
            attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens

        # Prepare inputs
        inputs = {
            "attn_metadata": attn_metadata,
            "input_ids": self._ctx.buffers.input_ids_cuda[:virtual_num_tokens],
            "position_ids": final_position_ids,
            "inputs_embeds": None,
            "multimodal_params": multimodal_params_list,
            "resource_manager": resource_manager,
        }
        inputs.update(cross_attention_inputs)

        if self._ctx.use_mrope:
            if mrope_delta_write_seq_slots:
                delta_write_seq_slots = torch.tensor(
                    mrope_delta_write_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
                )
                inputs["mrope_delta_write_seq_slots"] = delta_write_seq_slots.to(
                    device="cuda", non_blocking=True
                )

            if mrope_delta_read_seq_slots:
                delta_read_seq_slots = torch.tensor(
                    mrope_delta_read_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
                )
                inputs["mrope_delta_read_seq_slots"] = delta_read_seq_slots.to(
                    device="cuda", non_blocking=True
                )

        if bool(lora_params):
            inputs["lora_params"] = lora_params

        if spec_metadata is not None:
            total_draft_lens = sum(draft_lens)
            spec_metadata.draft_tokens = self._ctx.buffers.draft_tokens_cuda[:total_draft_lens]
            spec_metadata.request_ids = request_ids
            spec_metadata.gather_ids = self._ctx.buffers.gather_ids_cuda[: len(gather_ids)]
            # num_generations / num_tokens / seq_lens are set above, before the
            # attention-DP allgather that must agree with prepare().
            spec_metadata.host_position_ids = host_position_ids
            spec_metadata.num_accepted_draft_tokens = (
                self._ctx.buffers.num_accepted_draft_tokens_cuda[: len(num_accepted_draft_tokens)]
            )
            if context_prompt_lookahead is not None:
                spec_metadata.populate_context_prompt_lookahead(context_prompt_lookahead)
            # No-op for non 1-model
            spec_metadata.populate_sampling_params_for_one_model(scheduled_requests.all_requests())
            spec_metadata.prepare()
            # One-model rejection: one-hot the stale draft_probs rows of gen
            # requests that produced no draft tokens this step, so the (possibly
            # captured) rejection kernel reads a legal placeholder distribution.
            spec_metadata.write_padding_onehot_draft_probs(
                padding_gen_slots, self._ctx.state.runtime_draft_len
            )
            inputs["spec_metadata"] = spec_metadata

            if self._ctx.config.enable_attention_dp:
                set_spec_metadata_all_rank_num_tokens(spec_metadata, *spec_all_rank_counts)

        if mm_token_indices is not None:
            ship_multimodal_indices(
                inputs,
                mm_token_indices_cpu=mm_token_indices,
                text_token_indices_cpu=text_token_indices_ctx,
                num_ctx_tokens=num_ctx_tokens,
                total_num_tokens=total_num_tokens,
            )

        num_generation_tokens = (
            len(generation_requests)
            + len(extend_requests)
            + sum(draft_lens)
            + len(first_draft_requests)
        )
        self._ctx.iter_states["num_ctx_requests"] = num_ctx_requests
        self._ctx.iter_states["num_ctx_tokens"] = num_ctx_tokens
        self._ctx.iter_states["num_generation_tokens"] = num_generation_tokens
        # Count the already-cached prefix for the sequences scheduled this iteration.
        self._ctx.iter_states["cached_kv_tokens"] = sum(num_cached_tokens_per_seq)

        if not self._ctx.is_warmup:
            self._previous_request_ids = all_gen_request_ids

            # Record the steady-state generation cache when this pass handled
            # purely non-dummy generation requests that all carried a previous
            # overlap-scheduler tensor (previous_batch_len == _n_gen implies
            # every request took that branch and none appended input_ids).
            # While the batch composition holds, the next passes only need to
            # advance positions by one and refresh per-step metadata.
            # MRoPE models are supported only for batches with no actual mrope
            # work (text-only requests, empty mrope lists below): the full
            # pass routes use_mrope models through the (3,1,N)
            # mrope_position_ids_cuda layout even then (to keep torch.compile
            # guards stable), with all three axes equal to the scalar
            # positions, so the fast path advances that buffer in place and
            # returns the same layout (see _apply_steady_gen_fast_prepare).
            if (
                self._ctx.config.spec_config is None
                and not self._ctx.config.is_draft_model
                and spec_metadata is None
                and new_tokens_device is not None
                and self._ctx.guided_decoder is None
                and not self._ctx.config.enable_attention_dp
                and not mrope_position_ids
                and not mrope_delta_write_seq_slots
                and not mrope_delta_read_seq_slots
                and not self._ctx.use_beam_search
                and self._ctx.runner_config.max_beam_width == 1
                and not is_enc_dec
                and not _has_cp_helix
                and num_ctx_requests == 0
                and not extend_requests
                and not first_draft_requests
                and _n_gen > 0
                and previous_batch_len == _n_gen
                and num_tokens == 0
                and not _has_any_multimodal_request
                and not multimodal_params_list
                and not lora_params
                and attn_metadata.padded_num_tokens is None
                and get_position_id_offset(self._ctx.deps.model) == 0
                and not getattr(kv_cache_manager, "kv_compression_manages_history", False)
            ):
                self._ctx.config.steady_gen_positions_pinned[:_n_gen].copy_(
                    torch.as_tensor(num_cached_tokens_snapshot, dtype=torch.int)
                )
                self._steady_gen_cache = {
                    "num_requests": _n_gen,
                    "request_ids": all_gen_request_ids,
                    "prompt_lens": prompt_lengths,
                    "seq_lens_ones": maybe_pin_memory(torch.ones(_n_gen, dtype=torch.int)),
                    "use_mrope": _use_mrope,
                }

        return inputs, self._ctx.buffers.gather_ids_cuda[
            : len(gather_ids)
        ] if self._ctx.state.enable_spec_decode else None

    def _preprocess_inputs(self, inputs: Dict[str, Any]):
        """
        Make some changes to the device inputs and avoid blocking the async data transfer
        """
        attn_meta = inputs.get("attn_metadata")
        # Invalidate per-forward-pass caches so they are recomputed (and captured) on every _forward_step.
        if attn_meta is not None:
            attn_meta.on_update_kv_lens()

        if self._ctx.state.enable_spec_decode and not self._ctx.config.disable_overlap_scheduler:
            # When enabling overlap scheduler, the kv cache for draft tokens will
            # be prepared in advance by using the max_total_draft_tokens. But we need to use
            # new_tokens_lens_device to get the real past kv lengths and the
            # correct position ids. And to avoid blocking the async data transfer,
            # we need to preprocess the inputs in forward to update the position_ids and
            # kv cache length.
            if inputs["attn_metadata"].kv_cache_manager is not None:
                num_seqs = inputs["attn_metadata"].num_seqs
                num_ctx_requests = inputs["attn_metadata"].num_contexts
                num_gen_requests = inputs["attn_metadata"].num_generations
                num_ctx_tokens = inputs["attn_metadata"].num_ctx_tokens
                num_chunked_ctx_requests = inputs["attn_metadata"].num_chunked_ctx_requests
                previous_batch_tokens = inputs["input_ids"].shape[0] - num_ctx_tokens
                if inputs["position_ids"].ndim == 3:  # mrope: [3, 1, N]
                    inputs["position_ids"][:, :, num_ctx_tokens:] += (
                        self._ctx.buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )
                else:
                    inputs["position_ids"][0, num_ctx_tokens:] += (
                        self._ctx.buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )

                if hasattr(inputs["attn_metadata"], "kv_lens_cuda"):
                    if (
                        num_ctx_requests >= num_chunked_ctx_requests
                        and num_chunked_ctx_requests > 0
                    ):
                        # The generation requests with draft_tokens are treated as chunked context requests when
                        #   extend_ctx returns True.
                        inputs["attn_metadata"].kv_lens_cuda[
                            num_ctx_requests - num_chunked_ctx_requests : num_ctx_requests
                        ] += self._ctx.buffers.previous_kv_lens_offsets_cuda[
                            :num_chunked_ctx_requests
                        ]
                    else:
                        inputs["attn_metadata"].kv_lens_cuda[num_ctx_requests:num_seqs] += (
                            self._ctx.buffers.previous_kv_lens_offsets_cuda[:num_gen_requests]
                        )
                    inputs["attn_metadata"].on_update_kv_lens()
                # TRTLLM uses `kv_lens_cuda` above; FlashInfer exposes this backend-specific
                # correction without coupling the engine to its metadata type.
                elif hasattr(inputs["attn_metadata"], "apply_spec_decode_kv_lens_offsets"):
                    inputs["attn_metadata"].apply_spec_decode_kv_lens_offsets(
                        self._ctx.buffers.previous_kv_lens_offsets_cuda,
                        num_gen_requests,
                        self._ctx.deps.spec.runtime_tokens_per_gen_step(
                            self._ctx.state.runtime_draft_len
                        ),
                        num_chunked_contexts=num_chunked_ctx_requests,
                    )

        if self._ctx.guided_decoder is not None:
            self._ctx.guided_decoder.token_event.record()

        return inputs

    def _postprocess_inputs(self, inputs: Dict[str, Any]):
        """
        Postprocess to make sure model forward doesn't change the inputs.
        It is only used in cuda graph capture, because other cases will prepare
        new inputs before the model forward.
        """
        if self._ctx.state.enable_spec_decode and not self._ctx.config.disable_overlap_scheduler:
            if inputs["attn_metadata"].kv_cache_manager is not None:
                num_seqs = inputs["attn_metadata"].num_seqs
                num_ctx_requests = inputs["attn_metadata"].num_contexts
                num_gen_requests = inputs["attn_metadata"].num_generations
                num_ctx_tokens = inputs["attn_metadata"].num_ctx_tokens
                num_chunked_ctx_requests = inputs["attn_metadata"].num_chunked_ctx_requests
                previous_batch_tokens = inputs["input_ids"].shape[0] - num_ctx_tokens
                if inputs["position_ids"].ndim == 3:  # mrope: [3, 1, N]
                    inputs["position_ids"][:, :, num_ctx_tokens:] -= (
                        self._ctx.buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )
                else:
                    inputs["position_ids"][0, num_ctx_tokens:] -= (
                        self._ctx.buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )

                # Only TrtllmAttentionMetadata has kv_lens_cuda.
                if isinstance(inputs["attn_metadata"], TrtllmAttentionMetadata):
                    if (
                        num_ctx_requests >= num_chunked_ctx_requests
                        and num_chunked_ctx_requests > 0
                    ):
                        inputs["attn_metadata"].kv_lens_cuda[
                            num_ctx_requests - num_chunked_ctx_requests : num_ctx_requests
                        ] -= self._ctx.buffers.previous_kv_lens_offsets_cuda[
                            :num_chunked_ctx_requests
                        ]
                    else:
                        inputs["attn_metadata"].kv_lens_cuda[num_ctx_requests:num_seqs] -= (
                            self._ctx.buffers.previous_kv_lens_offsets_cuda[:num_gen_requests]
                        )
                # Restore the FlashInfer-specific logical KV lengths through the same optional hook
                # used by `_preprocess_inputs`.
                elif hasattr(inputs["attn_metadata"], "apply_spec_decode_kv_lens_offsets"):
                    inputs["attn_metadata"].apply_spec_decode_kv_lens_offsets(
                        self._ctx.buffers.previous_kv_lens_offsets_cuda,
                        num_gen_requests,
                        self._ctx.deps.spec.runtime_tokens_per_gen_step(
                            self._ctx.state.runtime_draft_len
                        ),
                        num_chunked_contexts=num_chunked_ctx_requests,
                        restore=True,
                    )

    @nvtx_range("_apply_steady_gen_fast_prepare")
    def _apply_steady_gen_fast_prepare(
        self,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        attn_metadata: AttentionMetadata,
        new_tensors_device: SampleStateTensors,
        resource_manager: Optional[ResourceManager],
    ):
        """Prepare inputs for an unchanged generation-only batch.

        Every request advanced by exactly one committed token since the last
        prepare, so instead of re-walking the batch in Python this advances
        the cached positions in place (device position buffer plus a pinned
        host counter), reuses the seq-slot buffer already on device, and
        refreshes only the per-step metadata. For mrope models (recorded only
        for batches with no actual mrope work) the (3,1,N) broadcast buffer
        the model reads is the one advanced.
        """
        cache = self._steady_gen_cache
        num_requests = cache["num_requests"]

        # Positions and cached-token counts are the same values in this
        # regime; advance both by one. The device-side position buffer is
        # advanced in place: it still holds the previous step's positions
        # because only _prepare_tp_inputs writes it and the cache validity
        # invariant guarantees the previous pass wrote these same rows. This
        # avoids reusing a mutated pinned buffer as the source of an async
        # H2D whose previous-step copy may still be pending under the overlap
        # scheduler (the nvbug 6293536 hazard class; see
        # KVCacheManager._stage_block_offsets_for_copy). The pinned buffer is
        # host-side bookkeeping only.
        use_mrope = cache["use_mrope"]
        positions = self._ctx.config.steady_gen_positions_pinned[:num_requests]
        positions.add_(1)
        if use_mrope:
            # Text-only batch on an mrope model: the recording pass broadcast
            # the scalar positions onto all three axes of the (3,1,N) buffer,
            # which is what the model (and any captured CUDA graph) reads, so
            # advance it in place. position_ids_cuda is reseeded by the next
            # full pass.
            self._ctx.buffers.mrope_position_ids_cuda[:, :, :num_requests].add_(1)
        else:
            self._ctx.buffers.position_ids_cuda[:num_requests].add_(1)
        num_cached_tokens_per_seq = positions.tolist()

        # Gather this step's input tokens from the previous iteration's device
        # sample buffer; the seq-slot indices in previous_batch_indices_cuda
        # are unchanged since the last full pass.
        previous_slots = self._ctx.buffers.previous_batch_indices_cuda[:num_requests]
        torch.index_select(
            new_tensors_device.new_tokens[0, :, : self._ctx.runner_config.max_beam_width],
            0,
            previous_slots,
            out=self._ctx.buffers.input_ids_cuda[
                : num_requests * self._ctx.runner_config.max_beam_width
            ].view(num_requests, self._ctx.runner_config.max_beam_width),
        )

        if not attn_metadata.is_cuda_graph:
            attn_metadata.seq_lens = cache["seq_lens_ones"]
        attn_metadata.beam_width = 1
        attn_metadata.request_ids = cache["request_ids"]
        attn_metadata.prompt_lens = cache["prompt_lens"]
        attn_metadata.num_contexts = 0
        attn_metadata.num_chunked_ctx_requests = 0
        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(None),
        )
        attn_metadata.kv_cache_manager = kv_cache_manager
        if hasattr(self._ctx.deps.model.model_config.pretrained_config, "chunk_size"):
            attn_metadata.mamba_chunk_size = (
                self._ctx.deps.model.model_config.pretrained_config.chunk_size
            )
        with nvtx_range("steady_gen_metadata_prepare"):
            attn_metadata.prepare()

        attn_all_rank_num_tokens = get_all_rank_num_tokens(
            attn_metadata,
            enable_attention_dp=self._ctx.config.enable_attention_dp,
            mapping=self._ctx.deps.mapping,
            dist=self._ctx.deps.dist,
        )
        padded_num_tokens, can_run_piecewise_cuda_graph, attn_all_rank_num_tokens = (
            get_padding_params(
                num_requests,
                0,
                attn_all_rank_num_tokens,
                dist=self._ctx.deps.dist,
                enable_attention_dp=self._ctx.config.enable_attention_dp,
                prefill_cuda_graph_backend=self._ctx.config.prefill_cuda_graph_backend,
                prefill_cuda_graph_num_tokens=self._ctx.config.prefill_cuda_graph_num_tokens,
            )
        )
        set_per_request_prefill_cuda_graph_flag(can_run_piecewise_cuda_graph)
        attn_metadata.padded_num_tokens = (
            padded_num_tokens if padded_num_tokens != num_requests else None
        )
        virtual_num_tokens = num_requests
        if attn_metadata.padded_num_tokens is not None:
            self._ctx.buffers.input_ids_cuda[num_requests:padded_num_tokens].fill_(0)
            # Zero-fill the padding tail of whichever position layout the
            # model consumes, matching the full pass.
            if use_mrope:
                self._ctx.buffers.mrope_position_ids_cuda[
                    :, :, num_requests:padded_num_tokens
                ].fill_(0)
            else:
                self._ctx.buffers.position_ids_cuda[num_requests:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens

        self._ctx.iter_states["num_ctx_requests"] = 0
        self._ctx.iter_states["num_ctx_tokens"] = 0
        self._ctx.iter_states["num_generation_tokens"] = num_requests
        self._ctx.iter_states["cached_kv_tokens"] = sum(num_cached_tokens_per_seq)

        if use_mrope:
            final_position_ids = self._ctx.buffers.mrope_position_ids_cuda[
                :, :, :virtual_num_tokens
            ]
        else:
            final_position_ids = self._ctx.buffers.position_ids_cuda[:virtual_num_tokens].unsqueeze(
                0
            )
        inputs = {
            "attn_metadata": attn_metadata,
            "input_ids": self._ctx.buffers.input_ids_cuda[:virtual_num_tokens],
            "position_ids": final_position_ids,
            "inputs_embeds": None,
            "multimodal_params": [],
            "resource_manager": resource_manager,
        }
        return inputs, None

    def _can_use_steady_gen_fast_prepare(
        self,
        scheduled_requests: ScheduledRequests,
        new_tokens_device: Optional[torch.Tensor],
        next_draft_tokens_device: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata],
    ) -> bool:
        """Check whether the cached steady-state generation prepare applies.

        The cache is only recorded by a full _prepare_tp_inputs pass whose
        batch consisted purely of non-dummy generation requests that all had
        a previous overlap-scheduler tensor (see the recording site), so the
        per-step check only needs to confirm the dynamic conditions: still a
        generation-only batch with the exact same requests in the same order.
        """
        cache = self._steady_gen_cache
        if cache is None or self._ctx.is_warmup:
            return False
        if (
            new_tokens_device is None
            or next_draft_tokens_device is not None
            or spec_metadata is not None
        ):
            return False
        if scheduled_requests.num_context_requests > 0:
            return False
        generation_requests = scheduled_requests.generation_requests
        if len(generation_requests) != cache["num_requests"]:
            return False
        return cache["request_ids"] == [request.py_request_id for request in generation_requests]

    def _get_all_rank_num_tokens_and_spec_counts(
        self, attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata
    ) -> Tuple[Optional[List[int]], Optional[List[List[int]]]]:
        """Exchange the attention and speculative per-rank counts in a single
        collective instead of one collective each.

        The spec token count is derived via ``SpecMetadata.dp_num_tokens``
        because this collective runs *before* ``spec_metadata.prepare()``
        rewrites ``num_tokens`` into the same shape (the attention count it
        shares the collective with is consumed earlier, by padding selection).
        Both read the same fields, so callers must have set
        ``spec_metadata.num_tokens``, ``num_generations`` and ``seq_lens`` for
        this batch before calling this. ``num_generations`` counts every
        generation request, CUDA-graph dummies included.
        """
        if not self._ctx.config.enable_attention_dp:
            return None, None
        spec_counts = (
            spec_metadata.dp_num_tokens(),
            len(spec_metadata.seq_lens),
            spec_metadata.num_generations,
        )
        if self._ctx.deps.mapping.cp_size > 1 and not self._ctx.deps.mapping.has_cp_helix():
            # attn counts span TP only while spec counts span TP*CP; keep the
            # two exchanges separate.
            gathered = self._ctx.deps.dist.tp_cp_allgather_int64(list(spec_counts))
            return (
                get_all_rank_num_tokens(
                    attn_metadata,
                    enable_attention_dp=self._ctx.config.enable_attention_dp,
                    mapping=self._ctx.deps.mapping,
                    dist=self._ctx.deps.dist,
                ),
                gathered.T.tolist(),
            )
        num_tokens = attn_metadata.num_tokens
        if self._ctx.deps.mapping.has_cp_helix():
            num_tokens = math.ceil(num_tokens / self._ctx.deps.mapping.cp_size)
        gathered = self._ctx.deps.dist.tp_cp_allgather_int64([num_tokens, *spec_counts])
        cols = gathered.T.tolist()
        return cols[0], cols[1:]

    def _pad_batch_seed_mrope_delta_cache(self, padded_requests: ScheduledRequests) -> None:
        if not self._ctx.use_mrope or padded_requests.num_generation_requests == 0:
            return

        mrope_position_deltas_cache = getattr(
            self._ctx.deps.model, "mrope_position_deltas_cache", None
        )
        if mrope_position_deltas_cache is None:
            mrope_position_deltas_cache = getattr(
                getattr(self._ctx.deps.model, "draft_model", None),
                "mrope_position_deltas_cache",
                None,
            )
        if mrope_position_deltas_cache is None:
            return

        mrope_seed_seq_slots = []
        mrope_seed_deltas = []
        mrope_seed_requests = []
        for request in padded_requests.generation_requests:
            if (
                request.py_seq_slot is None
                or request.is_dummy
                or getattr(request, "py_mrope_delta_cache_slot", None) == request.py_seq_slot
            ):
                continue
            mrope_position_delta = getattr(request, "py_mrope_position_delta", None)
            if mrope_position_delta is None and request.py_multimodal_data:
                mrope_config = request.py_multimodal_data.get("mrope_config")
                if mrope_config is not None:
                    mrope_position_delta = mrope_config.get("mrope_position_deltas")
            if mrope_position_delta is None:
                continue
            if mrope_position_delta.device.type == "cpu":
                mrope_position_delta = maybe_pin_memory(mrope_position_delta).to(
                    device="cuda", dtype=torch.int32, non_blocking=True
                )
            elif mrope_position_delta.dtype != torch.int32:
                mrope_position_delta = mrope_position_delta.to(dtype=torch.int32)
            request.py_mrope_position_delta = mrope_position_delta
            mrope_seed_seq_slots.append(request.py_seq_slot)
            mrope_seed_deltas.append(mrope_position_delta.reshape(1))
            mrope_seed_requests.append(request)

        if not mrope_seed_seq_slots:
            return

        mrope_seed_seq_slots_tensor = torch.tensor(
            mrope_seed_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
        ).to(device="cuda", non_blocking=True)
        mrope_seed_deltas_tensor = torch.cat(mrope_seed_deltas, dim=0)
        mrope_position_deltas_cache.index_copy_(
            0,
            mrope_seed_seq_slots_tensor,
            mrope_seed_deltas_tensor.to(dtype=mrope_position_deltas_cache.dtype),
        )
        for request in mrope_seed_requests:
            request.py_mrope_delta_cache_slot = request.py_seq_slot

    # ---- where encoder-decoder specializes input preparation ----
    def _can_use_input_fast_path(
        self, scheduled_requests, new_tokens_device, next_draft_tokens_device
    ) -> bool:
        """Decoder-only models have no encoder-decoder fast path."""
        return False

    def _prepare_inputs_fast(self, *args, **kwargs):
        raise NotImplementedError("reached only when _can_use_input_fast_path is true")

    def _prepare_cross_attn_inputs(self, *args, **kwargs):
        """Decoder-only models have no cross attention."""
        return None

    @functools.cached_property
    def mm_encoder_cache_enabled(self) -> bool:
        return mm_encoder_cache_enabled(self._ctx.deps.model)

    def should_use_full_generation_page_table(
        self, spec_config: Optional[DecodingBaseConfig], attn_metadata: AttentionMetadata
    ) -> bool:
        """Return whether overlap decode needs every reserved generation page."""
        # FlashInfer metadata owns the optional device-side KV-length correction used with this
        # wider page table.
        return (
            self._ctx.state.enable_spec_decode
            and not self._ctx.config.disable_overlap_scheduler
            and getattr(spec_config, "_use_shared_kv_cache", False)
            and hasattr(attn_metadata, "apply_spec_decode_kv_lens_offsets")
        )

    @functools.cached_property
    def is_multimodal(self) -> bool:
        return is_multimodal(self._ctx.deps.model, self._ctx.config.input_processor)
