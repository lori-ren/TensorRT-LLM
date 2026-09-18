# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warmup and CUDA graph capture for the decoder family."""

from __future__ import annotations

import contextlib
import gc
import math
import os
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch._dynamo.config

from tensorrt_llm._utils import global_mpi_rank
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend, SeqLenAwareSparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType
from tensorrt_llm.sampling_params import SamplingParams

from .....attention.backends.trtllm import TrtllmAttentionMetadata
from .....autotuner import AutoTuner, autotune
from .....compilation.utils import capture_piecewise_cuda_graph
from .....memory_buffer_utils import clear_memory_buffers
from .....modules.mamba.mamba2_metadata import Mamba2Metadata
from .....speculative import get_num_extra_kv_tokens
from .....speculative.eagle3 import Eagle3ResourceManager
from ....cuda_graph_runner import ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM
from ....kv_cache.mamba_cache_manager import MambaHybridCacheManager
from ....llm_request import LlmRequestState
from ....resource_manager import KVCacheManager, ResourceManager, ResourceManagerType
from ....sampler.ops.flashinfer import warmup_sample_from_logits_op, warmup_sampling_module
from ....sampler.sampler_common import SampleType
from ....scheduler import ScheduledRequests
from ....trace_log_utils import log_mem_snapshot
from ..common import _set_moe_a2a_warmup
from .context import DecoderContext
from .forward import ForwardExecutor
from .prepare import InputPreparer

NON_GREEDY_CAPTURE_SAMPLING_PARAMS = SamplingParams(temperature=0.7, top_k=50, top_p=0.9)


@contextlib.contextmanager
def _moe_a2a_steady_state_budget_for_capture():
    """Force the steady-state MoE all-to-all budget across CUDA-graph capture.

    The budget is a kernel launch argument, so it is frozen into each captured
    graph. Capture happens inside the warmup window, so without this a replay
    would keep warmup's relaxed deadline for the life of the process.
    """
    _set_moe_a2a_warmup(False)
    try:
        yield
    finally:
        _set_moe_a2a_warmup(True)


class WarmupDriver:
    """Warm up the family and capture its CUDA graphs."""

    def __init__(
        self, ctx: DecoderContext, preparer: InputPreparer, executor: ForwardExecutor
    ) -> None:
        self._ctx = ctx
        self._preparer = preparer
        self._executor = executor
        self._capture_sample_type = None

    def warmup(self, resource_manager: ResourceManager) -> None:
        """Warm up and capture this family's graphs.

        Capture happens inside warmup, which is how the engine sequenced
        it; ``capture_graphs`` is therefore a no-op for this family.
        """
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        # Ahead of the legacy early returns below: only the advanced-sampling
        # CUDA graph capture pass exercises the non-greedy sampler, so with
        # cuda_graph_config=None flashinfer's sampling kernels would be
        # JIT-built mid-serving.
        warmup_sampling_module()
        if self._ctx.config.enable_in_graph_sampling:
            # The fast tier samples inside the captured graph via a
            # torch.compile'd op; compile it now so capture does not.
            warmup_sample_from_logits_op(
                self._ctx.deps.model.config.vocab_size,
                torch.device("cuda"),
                self._ctx.config.dtype,
                self._ctx.config.decoder_cuda_graph_batch_sizes or [],
            )

        if kv_cache_manager is None:
            logger.info("Skipping warm up as no KV Cache manager allocated.")
            return

        # The lifetime of model engine and kv cache manager can be different.
        # Reset the global cuda graph dummy requests in warmup.
        self._ctx.cuda_graph_runner.padding_dummy_requests = {}

        is_enc_dec = self._ctx.is_encoder_decoder
        if self._ctx.deps.mapping.cp_size > 1:
            cp_type = self._ctx.deps.mapping.cp_config.get("cp_type", None)
            if cp_type != CpType.HELIX:
                logger.info(
                    f"[ModelEngine::warmup] Skipping warmup for cp_type: {None if cp_type is None else cp_type.name}."
                )
                return

        # Create AutoTuner singleton in eager context before any compiled forward.
        # Otherwise the first get() can happen inside torch.compile tracing and
        # trigger non-traceable code (time.time(), torch.cuda.*) in the cache.
        AutoTuner.get()

        # ``guided_decoder`` is installed only on the last pipeline rank, so
        # this predicate is not rank-uniform on its own. Agree it before it
        # gates either the attention or the general phase.
        can_run_general_warmup = self._agree_warmup_flag(
            not is_enc_dec
            and not self._ctx.config.is_draft_model
            and not self._ctx.deps.mapping.has_cp_helix()
            and self._ctx.guided_decoder is None
            and not isinstance(kv_cache_manager, MambaHybridCacheManager)
        )

        log_mem_snapshot("warmup/before_warmup")
        # Compile the DSv4 indexer-Q CuTe DSL kernels before the first
        # collective-bearing forward, so their JIT cost is not charged against the
        # MoE all-to-all completion-flag deadline.
        self._prewarm_cute_dsl_indexer_q()
        log_mem_snapshot("warmup/after_cute_dsl_indexer_q")
        if not is_enc_dec:
            self._run_attention_warmup(resource_manager, can_run_general_warmup)

        if can_run_general_warmup:
            # Specialize torch.compile graphs across the key input shapes before CUDA graph capture.
            warmup_requests_configs = self._agree_warmup_shapes(
                self._get_full_general_warmup_requests(resource_manager)
            )
            # Currently graph has not been captured, disable cuda graph for this warmup.
            with self.no_cuda_graph():
                self._general_warmup(resource_manager, warmup_requests_configs)
                # Release C++ MoE workspace buffers so the autotuner can
                # reclaim the memory.  They will be re-allocated on next use.
                from .....custom_ops.torch_custom_ops import MoERunner

                MoERunner.clear_all_workspaces()
                # Clear Cache now as autotuner may use additional memory.
                # Memory pool will be warmed up later.
                gc.collect()
                torch.cuda.empty_cache()

        # Helix CP is decode-only and runs into issues with the
        # autotuner warmup's context requests.
        if not is_enc_dec and not self._ctx.deps.mapping.has_cp_helix():
            self._run_autotuner_warmup(resource_manager)
            log_mem_snapshot("warmup/after_autotuner")
            # Pre-JIT Mamba SSD multi-seq + HAS_INITSTATES=True Triton kernels
            # for Mamba hybrid models. Runs regardless of enable_autotuner,
            # since MambaHybridCacheManager skips _general_warmup and the
            # default autotuner shape is single-seq / no-initstates. Safe
            # no-op for non-Mamba models.
            self._run_mamba_hybrid_warmup(resource_manager)
            log_mem_snapshot("warmup/after_mamba_hybrid")
            # Release the autotuner's exploration-mode intermediates. The
            # exploration leftovers are pure waste that hide tens of GiB from
            # non-torch allocators (cuBLAS handle workspace, UCX/NIXL,
            # NVSHMEM).
            gc.collect()
            torch.cuda.empty_cache()
        # Warm up every graph shape before capturing any graph. Attention
        # kernels can switch implementations at smaller batch sizes and require
        # a larger workspace, so the first pass grows the workspace to its
        # maximum size. The second pass runs the final per-shape warmup and
        # captures without resizing the workspace.
        # Capture with the steady-state MoE all-to-all budget: the timeout is a
        # launch argument and is baked into every later replay.
        with _moe_a2a_steady_state_budget_for_capture():
            with self._ctx.cuda_graph_runner.allow_capture():
                self._ctx.cuda_graph_runner.is_warmup_only = True
                try:
                    with self.maybe_autotune_lora():
                        self._run_cuda_graph_warmup(resource_manager)
                finally:
                    self._ctx.cuda_graph_runner.is_warmup_only = False
                self._ctx.cuda_graph_runner.padding_dummy_requests = {}
                self._run_cuda_graph_warmup(resource_manager)
        log_mem_snapshot("warmup/after_cuda_graph_capture")
        # Pre-compile DeepGEMM paged_mqa_logits_metadata for every 32-aligned
        # batch bucket the runtime can produce (max_batch_size scaled by the
        # MTP / DSL expansion factor when applicable). CUDA-graph warmup only
        # exercises the batch sizes in cuda_graph_batch_sizes, which round
        # up to a subset of buckets; any inference iter whose
        # context_lens.size(0) lands on an uncovered bucket triggers an
        # nvcc-driven JIT compile (~3s stall inside _prepare_inputs) on
        # first touch. Pre-touching every bucket funnels that cost into
        # warmup. No-op on non-DSA models.
        # Both DSA hooks read attn_metadata, which only a warmup forward
        # creates; build it when every forward above was skipped.
        self._ensure_dsa_attn_metadata_for_warmup(resource_manager)
        self._warmup_dg_paged_mqa_logits_metadata()
        log_mem_snapshot("warmup/after_dg_paged_mqa_logits_metadata")
        self._warmup_cute_dsl_radix_topk()
        log_mem_snapshot("warmup/after_cute_dsl_radix_topk")
        if can_run_general_warmup:
            # Pre-populate the memory pool with max-shape allocations to reduce
            # fragmentation at runtime.
            warmup_requests_configs = self._get_max_shape_warmup_requests(resource_manager)
            self._general_warmup(resource_manager, warmup_requests_configs)
            log_mem_snapshot("warmup/after_memory_pool_prepop")

        # Allocate the CUDA graph padding dummies now, while the KV cache is
        # empty. Waiting for the first padded step can race KV saturation:
        # once the cache is full, the lazy allocation in _get_padded_batch
        # fails every step and padded batches silently run eager.
        self._ctx.cuda_graph_runner.preallocate_padding_dummies(resource_manager)
        log_mem_snapshot("warmup/after_preallocate_padding_dummies")

        # If this is a BOLT-instrumented build (the profile-gen job sets
        # TLLM_BOLT_CLEAR_COUNTERS=1), reset the instrumentation counters now
        # that all startup JIT/autotune/graph-capture is done, so the emitted
        # .fdata reflects steady-state serving only. No-op on normal builds.
        from .....bolt_profiling import maybe_bolt_clear_counters

        maybe_bolt_clear_counters()

    def capture_graphs(self, resource_manager: ResourceManager) -> None:
        """No-op: this family captures inside :meth:`warmup`."""

    def _capture_generation_cuda_graphs(self, resource_manager: ResourceManager):
        """Warm up or capture pure-generation CUDA graph shapes."""
        if not self._ctx.cuda_graph_runner.enabled:
            return

        operation = "warmup" if self._ctx.cuda_graph_runner.is_warmup_only else "capture"
        logger.info(
            f"Running CUDA graph {operation} for "
            f"{len(self._ctx.config.decoder_cuda_graph_batch_sizes)} batch sizes."
        )

        # Reverse order so smaller graphs can reuse memory from larger ones
        cuda_graph_batch_sizes = sorted(
            self._ctx.config.decoder_cuda_graph_batch_sizes, reverse=True
        )

        # Determine which graph shapes to process.
        graphs_to_capture = self._get_graphs_to_capture(cuda_graph_batch_sizes)
        graphs_to_capture = sorted(graphs_to_capture, reverse=True)
        # Create CUDA graphs for short and long sequences separately for sparse attention.
        # max_seq_len is the global max sequence length. For Helix CP each
        # rank only holds max_seq_len / cp_size tokens, so scale accordingly to
        # avoid creating warmup requests whose position_ids exceed the RoPE
        # table (max_position_embeddings).
        effective_max_seq_len = self._ctx.runner_config.max_seq_len
        if self._ctx.deps.mapping is not None and self._ctx.deps.mapping.has_cp_helix():
            effective_max_seq_len = (
                self._ctx.runner_config.max_seq_len // self._ctx.deps.mapping.cp_size
            )

        sparse_config = self._ctx.config.sparse_attention_config
        if (
            isinstance(sparse_config, SeqLenAwareSparseAttentionConfig)
            and sparse_config.needs_separate_short_long_cuda_graphs()
        ):
            # For short sequences, subtract the maximum runtime tokens consumed
            # by a generation step so all current-step tokens stay within the
            # sequence length threshold. PARD uses 2K tokens here, not K+1.
            max_runtime_tokens_per_gen_step = self._ctx.deps.spec.runtime_tokens_per_gen_step(
                self._ctx.config.max_draft_len
            )
            # For long sequences, use the default maximum sequence length.
            max_seq_len = sparse_config.seq_len_threshold - max_runtime_tokens_per_gen_step
            if max_seq_len < effective_max_seq_len:
                max_seq_len_list = [effective_max_seq_len, max_seq_len]
            else:
                max_seq_len_list = [effective_max_seq_len]
        else:
            max_seq_len_list = [effective_max_seq_len]

        def prepare_cross_batch(
            batch: ScheduledRequests, resource_manager: ResourceManager
        ) -> None:
            """Populate dummy gen requests' cross-KV cache before capture.

            Dummy generation requests used for graph capture never ran a
            context step, so their cross-KV cache blocks are uninitialized
            and captured kernels would read garbage. Temporarily switch each
            request to a one-token context chunk with a fake encoder output
            to run just the cross-KV projection (via _populate_cross_kv_cache),
            then restore generation state for the actual capture.
            """
            if not batch.generation_requests:
                return

            max_encoder_output_len = self._max_encoder_output_len(resource_manager)
            hidden_size = self._enc_dec_hidden_size()
            saved_request_state = []
            for request in batch.generation_requests:
                saved_request_state.append(
                    (
                        request,
                        request.py_encoder_output,
                        request.py_skip_cross_kv_projection,
                        request.state,
                        request.py_batch_idx,
                        request._cached_tokens,
                        request._cached_tokens_set,
                    )
                )
                request.py_encoder_output = torch.ones(
                    (max_encoder_output_len, hidden_size),
                    device="cuda",
                    dtype=self._ctx.config.dtype,
                )
                request.py_skip_cross_kv_projection = False
                request.state = LlmRequestState.CONTEXT_INIT
                request.context_current_position = 0
                request.context_chunk_size = 1

            projection_batch = ScheduledRequests()
            projection_batch.reset_context_requests(batch.generation_requests)
            kv_cache_manager = resource_manager.get_resource_manager(
                self._ctx.config.kv_cache_manager_key
            )
            draft_kv_cache_manager = self._ctx.get_draft_kv_cache_manager(resource_manager)
            attn_metadata = self._ctx.set_up_attn_metadata(kv_cache_manager, draft_kv_cache_manager)
            with self.no_cuda_graph():
                projection_inputs, _ = self._preparer._prepare_inputs(
                    projection_batch,
                    kv_cache_manager,
                    attn_metadata,
                    spec_metadata=None,
                    new_tensors_device=None,
                    resource_manager=resource_manager,
                    maybe_graph=False,
                )
                self._populate_cross_kv_cache(projection_inputs)
            torch.cuda.synchronize()

            for (
                request,
                encoder_output,
                skip_cross_kv_projection,
                state,
                batch_idx,
                cached_tokens,
                cached_tokens_set,
            ) in saved_request_state:
                request.py_encoder_output = encoder_output
                request.py_skip_cross_kv_projection = skip_cross_kv_projection
                request.state = state
                if state == LlmRequestState.GENERATION_IN_PROGRESS:
                    request.context_current_position = request.prompt_len
                request.py_batch_idx = batch_idx
                request._cached_tokens = cached_tokens
                request._cached_tokens_set = cached_tokens_set

        def _run_capture_pass(
            force_non_greedy: bool,
            label: str,
            force_lora_graph: bool,
            sample_type: Optional[SampleType] = None,
        ) -> None:
            assert self._ctx.state.force_lora_graph_for_capture is None
            self._ctx.state.force_lora_graph_for_capture = force_lora_graph
            # Pin the sampling tier for this pass. maybe_get_cuda_graph reads
            # it to build the graph key, so every graph captured below records
            # the kernels of this tier and nothing else. Passes that do not name
            # a tier capture FULL graphs -- ones carrying no sampling at all --
            # which is what a batch resolving to FULL replays.
            pinned_tier = sample_type or SampleType.FULL
            self._capture_sample_type = pinned_tier
            self._ctx.cuda_graph_runner.set_capture_sample_type(pinned_tier)
            try:
                for bs, draft_len in graphs_to_capture:
                    if bs > self._ctx.runner_config.max_batch_size:
                        continue

                    for max_seq_len in max_seq_len_list:
                        warmup_request = self._create_cuda_graph_warmup_request(
                            resource_manager,
                            bs,
                            draft_len,
                            max_seq_len,
                            force_non_greedy=force_non_greedy,
                        )
                        with self._release_batch_context(warmup_request, resource_manager) as batch:
                            if batch is None:
                                # No KV cache space for this batch size. During KV
                                # cache estimation this makes the profiling peak
                                # unrepresentative (the final executor still
                                # captures this graph), so don't skip silently.
                                logger.warning(
                                    f"Skipping CUDA graph warmup ({label}) for "
                                    f"batch size={bs}, draft_len={draft_len}: "
                                    f"not enough KV cache space."
                                )
                                continue
                            logger.info(
                                f"Run generation-only CUDA graph {operation} ({label}) "
                                f"for batch size={bs}, draft_len={draft_len}, "
                                f"max_seq_len={max_seq_len}"
                            )
                            enable_spec_decode = (
                                draft_len > 0
                                or self._ctx.config.is_draft_model
                                or (
                                    self._ctx.config.spec_config is not None
                                    and self._ctx.config.spec_config.spec_dec_mode.use_one_engine()
                                )
                            )
                            with self._spec_decode_override(
                                enable=enable_spec_decode, draft_len=draft_len
                            ):
                                self._update_draft_inference_state_for_warmup(
                                    batch, draft_len > 0, resource_manager
                                )
                                if self._ctx.is_encoder_decoder:
                                    prepare_cross_batch(batch, resource_manager)
                                self._executor._run_batch(batch, resource_manager)
                                torch.cuda.synchronize()
            finally:
                self._ctx.state.force_lora_graph_for_capture = None
                self._capture_sample_type = None
                self._ctx.cuda_graph_runner.set_capture_sample_type(None)

        if self._ctx.state.cuda_graph_lora_manager is None:
            lora_graph_cases = [False]
        elif self._ctx.config.cuda_graph_specialize_lora:
            # Capture the larger LoRA graph first so the base-only graph can
            # reuse its CUDA graph memory-pool allocations.
            lora_graph_cases = [True, False]
        else:
            lora_graph_cases = [True]

        # Which variants to capture depends on which sampler this engine will
        # actually run, and the two are mutually exclusive: a spec-decoding mode
        # in a one-engine mode samples inside its worker and gets a dedicated
        # sampler from get_spec_decoder, so TorchSampler -- the only sampler
        # implementing in-graph sampling -- never runs. Capturing the other
        # branch's variants would spend warmup on graphs whose keys no batch can
        # ever produce.
        #
        # The predicate is use_one_engine() rather than has_spec_decoder(): the
        # two-model eagle3 / mtp_eagle modes also "have a spec decoder", but
        # get_spec_decoder hands them a plain TorchSampler, so they belong on
        # the TorchSampler branch and do need the FAST graphs.
        uses_own_spec_decoder = (
            self._ctx.config.spec_config is not None
            and self._ctx.config.spec_config.spec_dec_mode.use_one_engine()
        )

        def _capture_variant(
            label: str, force_non_greedy: bool = False, sample_type: Optional[SampleType] = None
        ) -> None:
            for use_lora_graph in lora_graph_cases:
                variant_label = label
                if self._ctx.state.cuda_graph_lora_manager is not None:
                    variant_label += ", LoRA" if use_lora_graph else ", base-only"
                _run_capture_pass(
                    force_non_greedy=force_non_greedy,
                    label=variant_label,
                    force_lora_graph=use_lora_graph,
                    sample_type=sample_type,
                )

        if uses_own_spec_decoder:
            # One-engine spec branch: the greedy argmax fast path plus the
            # advanced-sampling variant. The latter is needed because on-the-fly
            # capture is disabled outside warmup, so a batch containing a
            # non-greedy request would otherwise fall back to eager.
            #
            # Dummy warmup requests carry no sampling params, so the greedy pass
            # needs no override while the advanced one has to force the flag.
            _capture_variant("greedy")
            _capture_variant("advanced sampling", force_non_greedy=True)
        else:
            # TorchSampler branch: FULL carries no sampling in the graph and is
            # what every batch replays unless in-graph sampling is enabled, so it
            # is always captured. FAST adds the in-graph sampling kernels and is
            # captured only when opted in, since it costs extra warmup time and
            # memory.
            _capture_variant("full", sample_type=SampleType.FULL)
            if self._ctx.config.enable_in_graph_sampling:
                # Give the warmup requests real non-greedy sampling params:
                # dummies otherwise carry none, resolve to greedy, and the
                # capture would record argmax rather than the fast tier's
                # top-k/top-p kernels. Same substitution the advanced-sampling
                # pass relies on.
                _capture_variant("fast", force_non_greedy=True, sample_type=SampleType.FAST)

        # update_is_all_greedy_sample inside each forward call during the
        # non-greedy capture pass leaves is_all_greedy_sample=False on
        # spec_metadata. Reset it so the first real iteration starts clean;
        # update_is_all_greedy_sample will refresh it on every iteration anyway.
        # This is a defensive guard.
        if self._ctx.spec_metadata is not None:
            self._ctx.spec_metadata.is_all_greedy_sample = True

    def _create_cuda_graph_warmup_request(
        self,
        resource_manager: ResourceManager,
        batch_size: int,
        draft_len: int,
        max_seq_len: int = None,
        mixed_context_encoder_output_lens: Optional[Sequence[int]] = None,
        mixed_context_query_len: int = ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM,
        force_non_greedy: bool = False,
    ) -> Optional[ScheduledRequests]:
        """Creates a dummy ScheduledRequests tailored for CUDA graph capture."""
        capture_sampling_params = NON_GREEDY_CAPTURE_SAMPLING_PARAMS if force_non_greedy else None
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        draft_kv_cache_manager = self._ctx.get_draft_kv_cache_manager(resource_manager)

        available_blocks = (
            kv_cache_manager.get_num_free_blocks() // self._ctx.runner_config.max_beam_width
        )
        if available_blocks < batch_size:
            return None

        result = ScheduledRequests()
        runtime_tokens_per_gen_step = self._ctx.deps.spec.runtime_tokens_per_gen_step(draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        is_enc_dec = self._ctx.is_encoder_decoder
        max_encoder_output_len = (
            self._max_encoder_output_len(resource_manager) if is_enc_dec else None
        )
        num_mixed_contexts = len(mixed_context_encoder_output_lens or ()) if is_enc_dec else 0
        if num_mixed_contexts >= batch_size:
            return None

        # Add (batch_size - 1) dummy requests with the minimal sequence
        # length. Mixed capture must create its context rows as real context
        # requests; converting generation dummies afterward leaves their
        # native prompt/context bookkeeping at one token.
        if mixed_context_encoder_output_lens:
            context_request_ids = list(range(num_mixed_contexts))
            context_requests = kv_cache_manager.add_dummy_requests(
                context_request_ids,
                token_nums=[mixed_context_query_len] * num_mixed_contexts,
                is_gen=False,
                max_num_draft_tokens=runtime_draft_token_buffer_width,
                kv_reserve_draft_tokens=self._ctx.config.max_draft_loop_tokens,
                use_mrope=self._ctx.use_mrope,
                max_beam_width=self._ctx.runner_config.max_beam_width,
                encoder_output_lens=list(mixed_context_encoder_output_lens),
                draft_kv_cache_manager=draft_kv_cache_manager,
                capture_sampling_params=capture_sampling_params,
            )
            if context_requests is None:
                return None

            generation_request_ids = list(range(num_mixed_contexts, batch_size - 1))
            generation_requests = []
            if generation_request_ids:
                generation_requests = kv_cache_manager.add_dummy_requests(
                    generation_request_ids,
                    token_nums=[ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM] * len(generation_request_ids),
                    is_gen=True,
                    max_num_draft_tokens=runtime_draft_token_buffer_width,
                    kv_reserve_draft_tokens=self._ctx.config.max_draft_loop_tokens,
                    use_mrope=self._ctx.use_mrope,
                    max_beam_width=self._ctx.runner_config.max_beam_width,
                    encoder_output_lens=[max_encoder_output_len] * len(generation_request_ids),
                    draft_kv_cache_manager=draft_kv_cache_manager,
                    capture_sampling_params=capture_sampling_params,
                )
                if generation_requests is None:
                    for request in context_requests:
                        kv_cache_manager.free_resources(request)
                        if draft_kv_cache_manager is not None:
                            draft_kv_cache_manager.free_resources(request)
                    return None
            requests = context_requests + generation_requests
        else:
            token_nums = (
                ([ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM] * (batch_size - 1)) if is_enc_dec else None
            )
            encoder_output_lens = (
                ([max_encoder_output_len] * (batch_size - 1)) if is_enc_dec else None
            )
            requests = kv_cache_manager.add_dummy_requests(
                list(range(batch_size - 1)),
                token_nums=token_nums,
                is_gen=True,
                max_num_draft_tokens=runtime_draft_token_buffer_width,
                kv_reserve_draft_tokens=self._ctx.config.max_draft_loop_tokens,
                use_mrope=self._ctx.use_mrope,
                max_beam_width=self._ctx.runner_config.max_beam_width,
                encoder_output_lens=encoder_output_lens,
                draft_kv_cache_manager=draft_kv_cache_manager,
                capture_sampling_params=capture_sampling_params,
            )
            if requests is None:
                return None

        def free_warmup_requests() -> None:
            for r in requests:
                kv_cache_manager.free_resources(r)
                if draft_kv_cache_manager is not None:
                    draft_kv_cache_manager.free_resources(r)

        # Add one dummy request with the maximum possible sequence length.
        max_seq_len = min(
            self._ctx.runner_config.max_seq_len if max_seq_len is None else max_seq_len,
            kv_cache_manager.max_seq_len,
        )

        # Use max_draft_loop_tokens for capacity estimation to account
        # for the actual KV reservation per request.
        _kv_draft = self._ctx.config.max_draft_loop_tokens
        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=max_seq_len, batch_size=batch_size, max_num_draft_tokens=_kv_draft
        )

        # Also consider draft KV cache capacity when it exists
        if draft_kv_cache_manager is not None:
            draft_available_tokens = draft_kv_cache_manager.get_num_available_tokens(
                batch_size=batch_size,
                token_num_upper_bound=max_seq_len,
                max_num_draft_tokens=_kv_draft,
            )
            available_tokens = min(available_tokens, draft_available_tokens)

        token_num = max(
            ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM if is_enc_dec else 1,
            min(
                available_tokens,
                max_seq_len - 1 - get_num_extra_kv_tokens(self._ctx.config.spec_config) - _kv_draft,
            ),
        )
        model_config = self._ctx.deps.model.model_config.pretrained_config
        max_position_embeddings = getattr(model_config, "max_position_embeddings", None)
        if is_enc_dec:
            # For enc-dec models the engine max_seq_len covers the encoder
            # sequence, which may exceed the decoder's position table (e.g.
            # Whisper: 1500 encoder positions vs max_target_positions=448).
            decoder_position_limit = getattr(model_config, "max_target_positions", None)
            if decoder_position_limit is not None:
                max_position_embeddings = (
                    decoder_position_limit
                    if max_position_embeddings is None
                    else min(max_position_embeddings, decoder_position_limit)
                )
        if max_position_embeddings is not None:
            token_num = min(token_num, max_position_embeddings - _kv_draft)

        token_num = int(token_num)  # Ensure int for range() in add_dummy_requests

        max_seq_len_request = kv_cache_manager.add_dummy_requests(
            request_ids=[batch_size - 1],
            token_nums=[token_num],
            is_gen=True,
            max_num_draft_tokens=runtime_draft_token_buffer_width,
            kv_reserve_draft_tokens=self._ctx.config.max_draft_loop_tokens,
            use_mrope=self._ctx.use_mrope,
            max_beam_width=self._ctx.runner_config.max_beam_width,
            encoder_output_lens=[max_encoder_output_len] if is_enc_dec else None,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
        )

        if max_seq_len_request is None:
            free_warmup_requests()
            return None
        else:
            max_seq_len_request = max_seq_len_request[0]

        if mixed_context_encoder_output_lens:
            requests.append(max_seq_len_request)
            for request in requests[:num_mixed_contexts]:
                request.state = LlmRequestState.CONTEXT_INIT
                request.context_current_position = 0
                request.context_chunk_size = mixed_context_query_len
                request.cached_tokens = 0
                request.py_batch_idx = None
            result.context_requests_last_chunk = requests[:num_mixed_contexts]
            result.generation_requests = requests[num_mixed_contexts:]
        else:
            # Insert the longest request first to simulate padding for the CUDA
            # graph.
            requests.insert(0, max_seq_len_request)
            result.generation_requests = requests
        if spec_resource_manager is not None:
            spec_resource_manager.add_dummy_requests(request_ids=list(range(batch_size)))
        if self._ctx.is_encoder_decoder:
            if not self._add_cross_dummy_requests(result.all_requests(), resource_manager):
                return None
        return result

    def _run_autotuner_warmup(self, resource_manager: ResourceManager) -> None:
        """Runs forward passes to populate the autotuner cache."""
        from .....custom_ops.torch_custom_ops import (
            IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE,
            MXFP8GemmRunner,
        )
        from .....modules.linear import MXFP8LinearMethod, flashinfer_mxfp8_autotune

        enable_trtllm_autotuner = self._ctx.runner_config.enable_autotuner
        if not enable_trtllm_autotuner:
            return

        mxfp8_methods = []
        for module in self._ctx.deps.model.modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, MXFP8LinearMethod):
                mxfp8_methods.append(quant_method)

        # This engine owns startup warmup, so it explicitly opts its MXFP8
        # methods into native tuning. Standalone modules and engine paths that
        # skip this warmup remain on the direct native op.
        for method in mxfp8_methods:
            method.enable_native_autotune()

        # Native and FlashInfer tuning are independent. Capture native
        # eligibility before enabling graph-only FlashInfer dispatch.
        native_mxfp8_methods = [method for method in mxfp8_methods if method.needs_native_autotune]
        use_mxfp8_flashinfer_graph_default = (
            self._ctx.cuda_graph_runner.enabled
            and "TRTLLM_MXFP8_GEMM_BACKEND" not in os.environ
            and any(
                getattr(module, "_use_flashinfer_mxfp8_decode_graph_default", False)
                for module in self._ctx.deps.model.modules()
            )
        )
        if use_mxfp8_flashinfer_graph_default:
            # CuTeDSL is the alternative backend; PP has no graph-pass handoff.
            tune_with_cute_dsl = (
                IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE and not self._ctx.deps.mapping.has_pp()
            )
            for quant_method in mxfp8_methods:
                quant_method.enable_flashinfer_auto()
                quant_method.tune_decode_graph_backends = (
                    tune_with_cute_dsl and quant_method.uses_flashinfer
                )
        flashinfer_mxfp8_methods = [
            method for method in mxfp8_methods if method.needs_flashinfer_autotune
        ]

        # Every TP and PP rank must make the same backend decision before any
        # rank returns or enters a tuning forward with model collectives.
        if self._ctx.deps.mapping.tp_size > 1 or self._ctx.deps.mapping.has_pp():
            local_flashinfer_enabled = int(bool(flashinfer_mxfp8_methods))
            all_flashinfer_enabled = [local_flashinfer_enabled]
            if self._ctx.deps.mapping.tp_size > 1:
                all_flashinfer_enabled = list(
                    self._ctx.deps.dist.tp_allgather(local_flashinfer_enabled)
                )
            if self._ctx.deps.mapping.has_pp():
                all_flashinfer_enabled = [
                    enabled
                    for stage_flags in self._ctx.deps.dist.pp_allgather(all_flashinfer_enabled)
                    for enabled in stage_flags
                ]
            if any(all_flashinfer_enabled) and not all(all_flashinfer_enabled):
                forced_flashinfer = any(method.backend == "flashinfer" for method in mxfp8_methods)
                for method in mxfp8_methods:
                    method.disable_flashinfer_auto()
                flashinfer_mxfp8_methods = []
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but is not "
                        "available on every TP/PP rank"
                    )
                logger.warning(
                    "FlashInfer MXFP8 availability differs across TP/PP ranks; "
                    "using the native TensorRT-LLM GEMM backend on every rank."
                )

        enable_flashinfer_mxfp8_autotuner = bool(flashinfer_mxfp8_methods)
        enable_native_mxfp8_autotuner = bool(native_mxfp8_methods)

        AutoTuner.get().setup_distributed_state(self._ctx.deps.mapping, self._ctx.deps.dist)
        logger.info(
            f"Running autotuner warmup (TRT-LLM={enable_trtllm_autotuner}, "
            f"native MXFP8={enable_native_mxfp8_autotuner}, "
            f"FlashInfer MXFP8={enable_flashinfer_mxfp8_autotuner})..."
        )
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        token_num_upper_bound = min(
            self._ctx.runner_config.max_num_tokens,
            self._ctx.runner_config.max_batch_size * (self._ctx.runner_config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._ctx.config.original_max_draft_len,
        )

        warmup_configs = [(curr_max_num_tokens, 0)]
        if (
            not self._ctx.config.is_draft_model
            and self._ctx.guided_decoder is None
            and not self._ctx.deps.mapping.has_pp()
        ):
            # Add generation request to warmup the autotuner cache.
            warmup_configs.append((1 + self._ctx.config.max_total_draft_tokens, 1))

        def run_autotuner_pass(autotune_context: Any, synchronize_trtllm_cache: bool) -> bool:
            """Run one isolated tuning pass with fresh synthetic batches."""
            ran_forward = False
            with self.no_cuda_graph(), autotune_context:
                for num_tokens, num_gen_requests in warmup_configs:
                    warmup_request = self._create_warmup_request(
                        resource_manager, num_tokens, num_gen_requests
                    )
                    with self._release_batch_context(warmup_request, resource_manager) as batch:
                        if not self._should_run_warmup_batch(
                            batch,
                            num_tokens,
                            f"autotuner, num_tokens={num_tokens}, "
                            f"num_gen_requests={num_gen_requests}",
                        ):
                            continue
                        # Reset the flag is_first_draft for the draft model.
                        # This is necessary for overlap scheduler.
                        spec_resource_manager = resource_manager.get_resource_manager(
                            ResourceManagerType.SPEC_RESOURCE_MANAGER
                        )
                        if self._ctx.config.is_draft_model and isinstance(
                            spec_resource_manager, Eagle3ResourceManager
                        ):
                            spec_resource_manager.is_first_draft = True

                        self._executor._run_batch(batch, resource_manager)
                        ran_forward = True
                        torch.cuda.synchronize()

                if ran_forward and synchronize_trtllm_cache:
                    # pp_recv in AutoTuner choose_one will never be called if there is no tuning op during the forward
                    #   pass.
                    # So we need to make an extra call to consume the previous rank's pp_send to guarantee that the
                    #   previous rank's pp_send is released.
                    AutoTuner.get().cache_pp_recv()
                    # Send the cache after the tuning process to the next PP rank
                    AutoTuner.get().cache_pp_send()
                    # Clean the pp flag to avoid deadlock with synchronous send/recv
                    AutoTuner.get().clean_pp_flag()
            return ran_forward

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        ran_native_forward = run_autotuner_pass(
            autotune(cache_path=cache_path), synchronize_trtllm_cache=True
        )
        ran_flashinfer_forward = False
        if enable_flashinfer_mxfp8_autotuner:
            ran_flashinfer_forward = run_autotuner_pass(
                flashinfer_mxfp8_autotune(), synchronize_trtllm_cache=False
            )

        if enable_flashinfer_mxfp8_autotuner:
            if ran_flashinfer_forward:
                for method in flashinfer_mxfp8_methods:
                    method.mark_flashinfer_autotuned()
            else:
                forced_flashinfer = any(
                    method.backend == "flashinfer" for method in flashinfer_mxfp8_methods
                )
                for method in flashinfer_mxfp8_methods:
                    method.disable_flashinfer_auto()
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but its autotuner "
                        "warmup forward could not run"
                    )
                logger.warning(
                    "FlashInfer MXFP8 autotuning could not run; using the native "
                    "TensorRT-LLM GEMM backend."
                )

        if enable_native_mxfp8_autotuner:
            if ran_native_forward:
                MXFP8GemmRunner.sync_all_tactic_caches(AutoTuner.get())
                for method in native_mxfp8_methods:
                    method.mark_native_autotuned()
            else:
                for method in native_mxfp8_methods:
                    method.disable_native_autotune()
                logger.warning(
                    "Native MXFP8 autotuning had no runnable warmup batch; "
                    "using the default native GEMM tactic."
                )

        logger.info(
            f"[Autotuner] Cache size after warmup is {len(AutoTuner.get().profiling_cache)}"
        )
        AutoTuner.get().print_profiling_cache()

        self._release_megamoe_profiling_scratch()

        # Clear workspace buffers allocated during the autotuner forward pass.
        # The autotuner runs a context-only forward with max_num_tokens, which
        # causes the global Buffers pool to cache large MoE/GEMM workspaces.
        # If not cleared, these inflate the memory baseline seen by the KV cache
        # profiler, reducing memory available for activations during inference.
        clear_memory_buffers()
        torch.cuda.empty_cache()

    def _run_mamba_hybrid_warmup(self, resource_manager: ResourceManager) -> None:
        """Pre-JIT the Mamba SSD multi-seq + HAS_INITSTATES=True Triton kernels.

        Mamba hybrid models (e.g. Nemotron 3 Super 120B, Nemotron-Nano-12B-v2)
        skip ``_general_warmup`` because ``can_run_general_warmup`` is False
        when the KV cache manager is a ``MambaHybridCacheManager``. The default
        ``_run_autotuner_warmup`` then issues a single ``least_requests=True``
        prefill = 1 sequence with ``num_cached_tokens_per_seq = 0``, which only
        compiles the ``num_seqs == 1`` / ``HAS_INITSTATES=False`` variants of
        the SSD kernels. The first real serve iteration with chunked prefill
        and multiple context requests then triggers autotune of the missing
        variants mid-inference, producing a ~30 s stall / large P99 spike.

        This method runs two extra forward passes to compile those variants
        during warmup:

        1. ``least_requests=False`` — splits ``curr_max_num_tokens`` into many
           short sequences, forcing the multi-seq path of
           ``cu_seqlens_to_chunk_indices_offsets_triton`` and its
           ``_cu_seqlens_triton_kernel``.
        2. ``least_requests=False`` inside
           ``Mamba2Metadata.force_initial_states_for_warmup()`` — same as (1)
           plus the ``HAS_INITSTATES=True`` variants of
           ``_state_passing_fwd_kernel``, ``_chunk_scan_fwd_kernel``, and
           ``_chunk_state_varlen_kernel``.

        Runs regardless of ``enable_autotuner``. Wraps in ``autotune()`` when
        the autotuner is enabled so op-level (M,N,K) caches also get primed
        for these shapes. Set ``TLLM_MAMBA_MULTISEQ_WARMUP=0`` to disable.
        """
        if os.environ.get("TLLM_MAMBA_MULTISEQ_WARMUP", "1") != "1":
            return
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        if kv_cache_manager is None or not isinstance(kv_cache_manager, MambaHybridCacheManager):
            return

        token_num_upper_bound = min(
            self._ctx.runner_config.max_num_tokens,
            self._ctx.runner_config.max_batch_size * (self._ctx.runner_config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._ctx.config.original_max_draft_len,
        )
        # Rank-local capacity, so peers can disagree. Leaving the phase alone
        # would unbalance the per-shape agreement inside it.
        if not self._agree_warmup_flag(curr_max_num_tokens >= 4):
            return

        # Cap the multi-seq warmup token count so we don't fill the KV cache
        # to the brim. The autotuner warmup that ran just before this uses
        # ``least_requests=True`` (few long sequences) which fits comfortably
        # even when ``curr_max_num_tokens`` is close to the block ceiling.
        # ``least_requests=False`` instead spreads the token budget across
        # ``batch_size`` short sequences; when each sequence's length lands
        # exactly on a block boundary AND the KV cache has
        # ``num_extra_kv_tokens`` > 0 (e.g. spec decoding cases),
        # ``add_token`` needs to allocate one extra block per sequence, which
        # ``_create_warmup_request``'s ``blocks_to_use`` estimate doesn't
        # account for. On a small KV pool (e.g. Qwen3.5 hybrid with DFlash spec
        # decoding on a single H100: 259 blocks total, ``max_num_tokens=8192``
        # nearly saturates it), that extra per-sequence block overflows the
        # pool and crashes with "Can't allocate new blocks for window size N".
        # The point of this warmup is only to trigger ``num_seqs > 1`` +
        # ``HAS_INITSTATES=True`` kernel variants — a modest token budget
        # achieves that with plenty of block headroom.
        WARMUP_TOKEN_CAP = 4096
        capped_num_tokens = min(curr_max_num_tokens, WARMUP_TOKEN_CAP)

        logger.info("Running Mamba hybrid warmup (multi-seq + HAS_INITSTATES=True)...")

        # (num_tokens, num_gen_requests, least_requests, force_initstates)
        mamba_warmup_shapes = [
            (capped_num_tokens, 0, False, False),
            (capped_num_tokens, 0, False, True),
        ]

        autotuner_enabled = self._ctx.runner_config.enable_autotuner
        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        autotune_ctx = (
            autotune(cache_path=cache_path) if autotuner_enabled else contextlib.nullcontext()
        )

        with self.no_cuda_graph(), autotune_ctx:
            for num_tokens_i, num_gen_requests_i, least_req_i, force_init_i in mamba_warmup_shapes:
                init_ctx = (
                    Mamba2Metadata.force_initial_states_for_warmup()
                    if force_init_i
                    else contextlib.nullcontext()
                )
                shape = (
                    f"Mamba hybrid, num_tokens={num_tokens_i}, "
                    f"num_gen_requests={num_gen_requests_i}, "
                    f"force_initstates={force_init_i}"
                )
                with init_ctx:
                    try:
                        warmup_request = self._create_warmup_request(
                            resource_manager,
                            num_tokens_i,
                            num_gen_requests_i,
                            least_requests=least_req_i,
                        )
                    except torch.OutOfMemoryError as e:
                        if self._is_distributed_forward():
                            raise
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    except RuntimeError as e:
                        # The known KV allocation failure happens before
                        # forward and is recoverable only when no peer worker
                        # can advance independently. Any other RuntimeError is
                        # a defect, not a capacity limit, and is fatal.
                        if self._is_distributed_forward():
                            raise
                        if "Can't allocate new blocks for window size" not in str(e):
                            raise
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        torch.cuda.empty_cache()
                        continue

                    try:
                        with self._release_batch_context(warmup_request, resource_manager) as batch:
                            if not self._should_run_warmup_batch(batch, num_tokens_i, shape):
                                continue
                            spec_resource_manager = resource_manager.get_resource_manager(
                                ResourceManagerType.SPEC_RESOURCE_MANAGER
                            )
                            if self._ctx.config.is_draft_model and isinstance(
                                spec_resource_manager, Eagle3ResourceManager
                            ):
                                spec_resource_manager.is_first_draft = True

                            self._executor._run_batch(batch, resource_manager)

                            if autotuner_enabled:
                                AutoTuner.get().cache_pp_recv()
                                AutoTuner.get().cache_pp_send()
                                AutoTuner.get().clean_pp_flag()

                            torch.cuda.synchronize()
                    # Once peers can enter warmup synchronization or forward,
                    # any rank-local exception can strand them in a collective.
                    except Exception as e:  # noqa: BLE001
                        if self._is_distributed_forward():
                            raise
                        # ``torch.OutOfMemoryError`` is a ``RuntimeError``
                        # subclass; anything outside that hierarchy is a defect
                        # rather than a capacity limit.
                        if not isinstance(e, RuntimeError):
                            raise
                        # A single-rank warmup is a pure perf optimization. If
                        # a forward shape does not fit, it can be compiled
                        # lazily on the first real request.
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        # An OOM between dispatch() and combine() leaves the
                        # local MoE A2A state in ``dispatched``.
                        self._reset_moe_alltoall_state()
                        torch.cuda.empty_cache()

        clear_memory_buffers()
        torch.cuda.empty_cache()

    def _create_warmup_request(
        self,
        resource_manager: ResourceManager,
        num_tokens: int,
        num_gen_requests: int,
        least_requests: bool = True,
    ) -> Optional[ScheduledRequests]:
        """Creates a generic dummy ScheduledRequests object for warmup."""
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        draft_kv_cache_manager = self._ctx.get_draft_kv_cache_manager(resource_manager)

        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )

        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=num_tokens,
            max_num_draft_tokens=self._ctx.config.max_total_draft_tokens,
        )
        available_blocks = kv_cache_manager.get_num_free_blocks()
        if num_tokens > self._ctx.runner_config.max_num_tokens or num_tokens > available_tokens:
            return None

        if num_gen_requests > self._ctx.runner_config.max_batch_size:
            return None
        num_gen_tokens = num_gen_requests * (1 + self._ctx.config.max_total_draft_tokens)
        if num_gen_tokens > self._ctx.runner_config.max_num_tokens:
            return None

        num_ctx_tokens = num_tokens - num_gen_tokens
        num_ctx_requests = 0
        ctx_requests = []
        gen_requests = []

        # Leave room for at least one decode token per request.
        max_seq_len = self._ctx.runner_config.max_seq_len - 1
        if max_seq_len < 1:
            return None
        num_full_seqs = 0
        num_left_over_tokens = 0

        max_context_requests = self._ctx.runner_config.max_batch_size - num_gen_requests
        if max_context_requests * max_seq_len < num_ctx_tokens:
            return None

        if num_ctx_tokens > 0:
            if least_requests:
                num_full_seqs = num_ctx_tokens // max_seq_len
                num_left_over_tokens = num_ctx_tokens - num_full_seqs * max_seq_len

            else:
                max_bs = min(num_ctx_tokens, max_context_requests)
                if num_ctx_tokens % max_bs == 0:
                    num_full_seqs = max_bs
                else:
                    num_full_seqs = max_bs - 1
                max_seq_len = num_ctx_tokens // num_full_seqs
                num_left_over_tokens = num_ctx_tokens - max_seq_len * num_full_seqs
            num_ctx_requests = num_full_seqs + (1 if num_left_over_tokens > 0 else 0)

        if num_ctx_requests + num_gen_requests > self._ctx.runner_config.max_batch_size:
            return None  # Not enough batch size to fill the request

        # Mirror add_dummy_requests' actual allocation: on top of the raw
        # token count, every sequence gets num_extra_kv_tokens add_token
        # calls, and generation dummies additionally reserve
        # max_draft_loop_tokens for the draft loop.
        # In one-engine spec modes that is (max_draft_len - 1) extra KV
        # tokens plus max_draft_len draft-loop tokens per gen dummy, i.e.
        # 2 * max_draft_len - 1 on top of the single prompt token.
        # Under-counting these let warmup start an allocation that fails
        # midway and, before the partial-allocation cleanup existed,
        # permanently leaked most of the estimation-sized KV pool
        # (TRTLLM-14903).
        def blocks_for_seq(num_tokens: int) -> int:
            return math.ceil(num_tokens / kv_cache_manager.tokens_per_block)

        extra_ctx_tokens = getattr(kv_cache_manager, "num_extra_kv_tokens", 0) or 0
        extra_gen_tokens = extra_ctx_tokens + self._ctx.config.max_draft_loop_tokens
        blocks_to_use = num_full_seqs * blocks_for_seq(max_seq_len + extra_ctx_tokens)
        if num_left_over_tokens > 0:
            blocks_to_use += blocks_for_seq(num_left_over_tokens + extra_ctx_tokens)
        blocks_to_use += (
            num_gen_requests
            * self._ctx.runner_config.max_beam_width
            * blocks_for_seq(1 + extra_gen_tokens)
        )

        if blocks_to_use > available_blocks and isinstance(kv_cache_manager, KVCacheManager):
            return None

        if num_ctx_tokens > 0:
            ctx_token_nums = [max_seq_len] * num_full_seqs
            if num_left_over_tokens > 0:
                ctx_token_nums.append(num_left_over_tokens)

            ctx_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests)),
                token_nums=ctx_token_nums,
                is_gen=False,
                max_num_draft_tokens=self._ctx.config.max_total_draft_tokens,
                kv_reserve_draft_tokens=self._ctx.config.max_draft_loop_tokens,
                use_mrope=self._ctx.use_mrope,
                draft_kv_cache_manager=draft_kv_cache_manager,
            )

            if ctx_requests is None:
                return None

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(request_ids=list(range(num_ctx_requests)))

        if num_gen_requests > 0:
            gen_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests, num_ctx_requests + num_gen_requests)),
                token_nums=[1] * num_gen_requests,
                is_gen=True,
                max_num_draft_tokens=self._ctx.config.max_total_draft_tokens,
                kv_reserve_draft_tokens=self._ctx.config.max_draft_loop_tokens,
                use_mrope=self._ctx.use_mrope,
                max_beam_width=self._ctx.runner_config.max_beam_width,
                draft_kv_cache_manager=draft_kv_cache_manager,
            )

            if gen_requests is None:
                for r in ctx_requests:
                    kv_cache_manager.free_resources(r)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(r)
                return None

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(
                    request_ids=list(range(num_ctx_requests, num_ctx_requests + num_gen_requests))
                )

        result = ScheduledRequests()
        result.reset_context_requests(ctx_requests)
        result.generation_requests = gen_requests
        return result

    def _warmup_dg_paged_mqa_logits_metadata(self) -> None:
        """Pre-compile DeepGEMM's `get_paged_mqa_logits_metadata` helper for
        every 32-aligned batch bucket the runtime can produce.

        DSA's `Indexer.prepare_scheduler_metadata` calls
        `deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv,
        num_sms)` inside `_prepare_inputs` every iteration. The underlying
        kernel is templated on `<kAlignedBatchSize, split_kv, num_sms>`
        where `kAlignedBatchSize = align(context_lens.size(0), 32)` and
        `split_kv` / `num_sms` are fixed for a given (block_kv, device).
        deep_gemm's Python-side JIT compiles a fresh cubin (spawning
        nvcc/cicc/ptxas, ~3s on GB300) the first time each `aligned_bs`
        is requested. CUDA-graph warmup exercises only the batch sizes in
        `cuda_graph_batch_sizes`, which round up to a subset of the 32-
        aligned buckets; every uncovered bucket that the inference
        workload later touches produces a 3s stall on that iteration.
        Pre-touching every bucket here funnels those compiles into the
        deterministic warmup phase.

        `context_lens.size(0)` is not always `num_generations`. For MTP
        with `use_expanded_buffers_for_mtp=True` the expanded call passes
        `num_generations * (1 + max_draft_tokens)`. For DSL expansion the
        call passes `num_generations * dsl_expand_factor`, where
        `dsl_expand_factor = next_n // eff` (`eff in kernel_atoms`, see
        `_pick_dsl_expand` in `dsa.py`); its worst case is
        `next_n = 1 + max_draft_tokens` when `eff == 1`. Reading the
        current `dsl_expand_factor` off the metadata would under-estimate
        the eventual max (it defaults to 1 before any prepare() has run,
        and per-iter picks can differ across iters when CUDA graph is
        off), so we use the static upper bound `1 + max_draft_tokens`
        for both expansion paths. Bucket range is also scaled by
        `max_beam_width` as a defense-in-depth ceiling for future beam
        support (no-op today — DSA does not use beam). No-op on non-DSA
        models.

        Best-effort: per-bucket JIT failures are logged and skipped so a
        single broken bucket does not abort PyExecutor startup.
        """
        attn_meta = getattr(self, "attn_metadata", None)
        if attn_meta is None:
            return
        try:
            from tensorrt_llm._torch.attention.backends.sparse.dsa import (
                _DG_SCHEDULE_BLOCK_KV,
                DSAtrtllmAttentionMetadata,
            )
        except ImportError:
            return
        if not isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            return
        try:
            from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
        except ImportError:
            logger.info(
                "[DG warmup] deep_gemm.get_paged_mqa_logits_metadata not "
                "available; skipping paged_mqa_logits_metadata prewarm."
            )
            return

        num_sms = attn_meta.num_sms
        max_bs = max(1, int(self._ctx.runner_config.max_batch_size))
        beam_width = max(1, int(getattr(self, "max_beam_width", 1) or 1))
        # Static upper bound on the row-count multiplier applied to
        # `context_lens`. Both MTP-expanded and DSL-expanded call sites
        # are bounded above by `(1 + max_draft_tokens)`; see the
        # docstring for why we don't read the runtime `dsl_expand_factor`
        # here.
        max_draft_tokens = int(getattr(attn_meta, "max_draft_tokens", 0) or 0)
        expands_batch = getattr(attn_meta, "use_expanded_buffers_for_mtp", False) or getattr(
            attn_meta, "expand_for_dsl", False
        )
        expand_factor = 1 + max_draft_tokens if expands_batch else 1
        max_aligned = ((max_bs * beam_width * expand_factor + 31) // 32) * 32
        buckets = list(range(32, max_aligned + 32, 32))
        logger.info(
            f"[DG warmup] Pre-compiling paged_mqa_logits_metadata for "
            f"{len(buckets)} aligned batch buckets up to {max_aligned} "
            f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}, "
            f"max_bs={max_bs}, beam_width={beam_width}, "
            f"expand_factor={expand_factor})"
        )
        for aligned_bs in buckets:
            # Kernel scans `context_lens` and prefix-sums schedules; a
            # zero-filled 2D tensor of shape (aligned_bs, 1) is enough to
            # trigger dispatch and compile — the metadata output is
            # discarded.
            dummy = torch.zeros(aligned_bs, 1, dtype=torch.int32, device="cuda")
            try:
                _ = get_paged_mqa_logits_metadata(dummy, _DG_SCHEDULE_BLOCK_KV, num_sms)
            except RuntimeError as e:
                # Narrow to RuntimeError so signature drifts in
                # get_paged_mqa_logits_metadata (TypeError / ValueError)
                # surface loudly instead of silently degrading perf.
                logger.warning(
                    f"[DG warmup] paged_mqa_logits_metadata prewarm failed "
                    f"for aligned_bs={aligned_bs} "
                    f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}); "
                    f"skipping bucket. {type(e).__name__}: {e}"
                )
        torch.cuda.synchronize()

    def _capture_prefill_cuda_graphs(self, resource_manager: ResourceManager):
        """Capture configured CUDA graphs for context/prefill steps."""
        if self._ctx.config.prefill_cuda_graph_backend == PrefillCudaGraphBackend.DISABLED or (
            self._ctx.config.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
            and not self._ctx.config.torch_compile_enabled
        ):
            return

        logger.info("Running prefill CUDA graph warmup...")
        prefill_cuda_graph_num_tokens = sorted(
            self._ctx.config.prefill_cuda_graph_num_tokens, reverse=True
        )

        capture_context = (
            capture_piecewise_cuda_graph(True)
            if self._ctx.config.torch_compile_piecewise_cuda_graph
            else contextlib.nullcontext()
        )
        with capture_context, self.no_cuda_graph():
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._create_warmup_request(resource_manager, num_tokens, 0)
                with self._release_batch_context(warmup_request, resource_manager) as batch:
                    self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                    if batch is None:
                        continue

                    logger.info(f"Run prefill CUDA graph capture for num tokens={num_tokens}")
                    if self._ctx.breakable_cuda_graph_runner is not None:
                        self._ctx.breakable_cuda_graph_runner.capture(
                            num_tokens, lambda: self._executor._run_batch(batch, resource_manager)
                        )
                    else:
                        # Run a few times to ensure torch.compile capture.
                        for _ in range(4):
                            self._executor._run_batch(batch, resource_manager)

        # The logits allocations grow with the number of requests and are not
        # part of the captured model body. Warm up the largest request count so
        # those allocations can be reused during stable inference.
        for num_tokens in prefill_cuda_graph_num_tokens:
            warmup_request = self._create_warmup_request(
                resource_manager, num_tokens, 0, least_requests=False
            )
            with self._release_batch_context(warmup_request, resource_manager) as batch:
                self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                if batch is None:
                    continue
                logger.info(
                    f"Run prefill CUDA graph warmup for num tokens={num_tokens} with most requests"
                )
                if self._ctx.breakable_cuda_graph_runner is not None:
                    with self.no_cuda_graph():
                        self._ctx.breakable_cuda_graph_runner.warmup(
                            lambda: self._executor._run_batch(batch, resource_manager), steps=1
                        )
                else:
                    self._executor._run_batch(batch, resource_manager)
                torch.cuda.synchronize()

    def _run_attention_warmup(
        self, resource_manager: ResourceManager, can_run_general_warmup: bool = True
    ) -> None:
        if not issubclass(
            self._ctx.runner_config.attention_backend.Metadata, TrtllmAttentionMetadata
        ):
            return

        @contextlib.contextmanager
        def trtllm_gen_fmha_jit_warmup():
            previous = self._ctx.state.trtllm_gen_jit_warmup
            self._ctx.state.trtllm_gen_jit_warmup = True
            try:
                yield
            finally:
                self._ctx.state.trtllm_gen_jit_warmup = previous

        logger.info("Running TRTLLM-Gen FMHA JIT warmup")

        warmup_requests_configs = []
        if not self._ctx.config.is_draft_model and self._ctx.guided_decoder is None:
            # doesn't support 2-model speculative draft and guided decoding
            warmup_requests_configs.append(
                (1 + self._ctx.config.max_total_draft_tokens, 1)
            )  # one generation request
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Gen kernels")

        if can_run_general_warmup:
            warmup_requests_configs.append((1, 0))  # one context token
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Ctx kernels")

        model_type = getattr(
            self._ctx.deps.model.model_config.pretrained_config, "model_type", None
        )
        if can_run_general_warmup and model_type in ("kimi_k3", "kimi_linear"):
            # Kimi's one-token context takes the NT < 4 FLA fallback and does
            # not compile the optimized single-sequence K123 variant. A
            # non-aligned five-chunk context enters the pure K123 path.
            _KIMI_KDA_PREFILL_WARMUP_TOKENS = 257
            logger.info(
                "Adding Kimi KDA pure-prefill warmup with "
                f"{_KIMI_KDA_PREFILL_WARMUP_TOKENS} context tokens"
            )
            warmup_requests_configs.append((_KIMI_KDA_PREFILL_WARMUP_TOKENS, 0))

        if (
            not self._ctx.config.is_draft_model
            and self._ctx.guided_decoder is None
            and can_run_general_warmup
        ):
            # The cute_dsl_mla FMHA lib now only support the generation-only batch, we need to warmup the TRTLLM-Gen
            #   FMHA lib for the mixed context+generation batch.
            # One MIXED context+generation batch (1 ctx token + 1 gen request).
            warmup_requests_configs.append((1 + self._ctx.config.max_total_draft_tokens + 1, 1))
        else:
            logger.debug(
                "Skipped TRTLLM-Gen flashinfer_trtllm_gen FMHA lib JIT warmup When enable cute_dsl_mla FMHA lib"
            )

        for num_tokens, num_gen_requests in warmup_requests_configs:
            warmup_request = self._create_warmup_request(
                resource_manager, num_tokens=num_tokens, num_gen_requests=num_gen_requests
            )

            with (
                self.no_cuda_graph(),
                self._release_batch_context(warmup_request, resource_manager) as batch,
            ):
                if not self._should_run_warmup_batch(
                    batch,
                    num_tokens,
                    f"attention, num_tokens={num_tokens}, num_gen_requests={num_gen_requests}",
                ):
                    continue
                with trtllm_gen_fmha_jit_warmup():
                    self._executor._run_batch(batch, resource_manager)
                torch.cuda.synchronize()

    def _should_run_warmup_batch(
        self, batch: Optional[ScheduledRequests], num_tokens: int, shape: str
    ) -> bool:
        """Decide whether this warmup shape runs, is skipped, or fails the rank.

        A rank that skips a shape its peers run leaves them blocked in that
        forward's collectives for the rest of the job. Skipping is therefore
        safe exactly when every rank in the forward group skips too, and an
        allgather establishes that before any rank enters the forward.

        The plan agreed by ``_agree_warmup_plan`` is what makes that allgather
        safe: every rank walks the same shape list, so this runs the same
        number of times everywhere.
        """
        if not self._is_distributed_forward():
            if batch is not None:
                return True
            # Safe to skip, but never silently: a skip during KV cache
            # estimation makes the profiling peak unrepresentative of
            # this shape.
            logger.warning(f"Skipping warmup shape ({shape}): not enough KV cache space.")
            return False

        allgather = self._warmup_agreement_allgather()
        if allgather is None:
            # No reachable group to agree with. Keep the TP-only check, which
            # still catches the attention-DP asymmetry it was written for.
            self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
            if batch is None:
                raise RuntimeError(
                    f"Warmup batch creation failed for shape ({shape}) on "
                    f"global_rank={global_mpi_rank()}, "
                    f"model_rank={self._ctx.deps.dist.rank}, and this topology offers no "
                    f"way to confirm that peers are skipping it too. They may "
                    f"already be inside the matching forward, so this rank "
                    f"cannot skip the shape without stranding them."
                )
            return True

        flags = list(allgather(int(batch is not None)))
        if all(flags):
            return True
        if not any(flags):
            # Every rank in the forward group is skipping, so none of them is
            # left inside a collective. This is the ordinary outcome for a
            # shape that does not fit the configuration at all, such as a
            # mixed context+generation shape under ``max_batch_size=1``.
            logger.warning(
                f"Skipping warmup shape ({shape}) on all "
                f"{len(flags)} ranks: not enough KV cache space."
            )
            return False

        all_tokens = list(allgather(num_tokens))
        failed_ranks = [i for i, flag in enumerate(flags) if not flag]
        raise RuntimeError(
            f"Warmup batch creation failed for shape ({shape}) on rank(s) "
            f"{failed_ranks} but succeeded on others, so entering this forward "
            f"would deadlock the ranks that still hold a batch. Per-rank "
            f"curr_max_num_tokens: {all_tokens}. This indicates asymmetric KV "
            f"cache capacity across ranks. Consider increasing "
            f"--kv_cache_free_gpu_mem_fraction."
        )

    def _get_graphs_to_capture(self, cuda_graph_batch_sizes: list[int]) -> list[tuple[int, int]]:
        """Determine which (batch_size, draft_len) graphs to capture.

        Returns:
            List of (batch_size, draft_len) tuples for CUDA graph capture.
        """
        # Case 1: Draft model (two-model speculative decoding)
        # Two-model path is deprecated and will be removed in the near future
        if self._ctx.config.is_draft_model:
            draft_len = self._ctx.config.max_total_draft_tokens
            return [(bs, draft_len) for bs in cuda_graph_batch_sizes]

        # Case 2: One-model with dynamic draft length
        if (
            self._ctx.config.spec_config is not None
            and self._ctx.config.spec_config.draft_len_schedule is not None
            and self._ctx.config.spec_config.spec_dec_mode.support_dynamic_draft_len()
        ):
            graphs = [
                (graph_bs, draft_len)
                for graph_bs, draft_len in self._ctx.config.dynamic_draft_len_mapping.items()
            ]
            # Workaround for dynamic draft length:
            # capture the maximum speculative graph shape up front. Dynamic draft length
            # breaks the previous assumption that attention workspace demand can be safely
            # ordered by batch size alone; a later graph shape may require a larger shared
            # graph workspace, and resizing that workspace can change its data_ptr and
            # invalidate pointers captured by earlier graphs, causing illegal memory access
            # on replay.
            #
            # This adds the overhead of one extra captured graph, and that graph is not
            # expected to be used by the normal schedule-driven dynamic draft-length path.
            #
            # Follow-up first-principles fix:
            # query or precompute the exact attention workspace requirement for all
            # reachable graph shapes, pre-size the shared graph workspace once without
            # capturing an extra graph, and avoid resizing it in graph mode afterward.
            max_spec_graph = (max(cuda_graph_batch_sizes), self._ctx.config.original_max_draft_len)
            if max_spec_graph not in graphs:
                graphs.append(max_spec_graph)
            logger.info(
                f"Dynamic draft length enabled for one-model path. "
                f"Capturing {len(graphs)} graphs: {graphs}"
            )
            return graphs

        # Case 3: Target model (two-model) or one-model without dynamic draft
        # Match the runtime_draft_len semantics enforced in _prepare_tp_inputs:
        # logical K for linear-tree modes, total tree tokens for tree decoding.
        # spec_config is None for non-spec models — fall back to max_draft_len (= 0).
        draft_lengths = [
            self._ctx.config.max_draft_len
            if (self._ctx.config.spec_config is None or self._ctx.config.spec_config.is_linear_tree)
            else self._ctx.config.max_total_draft_tokens
        ]
        should_capture_no_spec = (
            self._ctx.config.max_total_draft_tokens > 0
            and not self._ctx.config.spec_config.spec_dec_mode.use_one_engine()
            # Assume speculation is always on if no max_concurrency set (saves memory)
            and self._ctx.config.spec_config.max_concurrency is not None
        )
        if should_capture_no_spec:
            draft_lengths.append(0)
        return [(bs, draft_len) for bs in cuda_graph_batch_sizes for draft_len in draft_lengths]

    def _prewarm_cute_dsl_indexer_q(self) -> None:
        """Pre-compile the DSv4 indexer-Q CuTe DSL kernels, then barrier.

        Runs before any collective-bearing forward so this op's first-touch
        ``cute.compile`` is not charged against the MoE all-to-all
        completion-flag deadline. It is a partial mitigation only: other
        first-touch compiles remain inside collective-bearing forwards, and some
        sit on the all-to-all path itself and cannot be pre-compiled this way.
        The runtime budget (``moeA2AGetTimeoutCycles``) covers the general case.

        Only the fallback tactics are compiled -- what an eager, cache-miss
        forward selects. The runner's kernel cache key excludes m/n/k, so one
        compile per tactic covers every shape. Uses the real module and weights,
        so it cannot drift from what the model runs.

        No-op on non-DSA models. See nvbugs/6482566.
        """
        try:
            from .....attention.backends.sparse.deepseek_v4.deepseek_v4 import DeepseekV4Indexer
        except ImportError:
            return

        indexer = next(
            (
                m
                for m in self._ctx.deps.model.modules()
                if isinstance(m, DeepseekV4Indexer) and getattr(m, "wq_b", None) is not None
            ),
            None,
        )
        if indexer is None:
            return

        weight = indexer.wq_b.weight
        # _fallback_tactic() branches on m at 4 and 8, so these three token
        # counts cover every fallback tactic it can return.
        with torch.inference_mode():
            for num_tokens in (4, 8, 16):
                try:
                    qr = torch.zeros(
                        (num_tokens, weight.shape[1]), dtype=torch.bfloat16, device=weight.device
                    )
                    position_ids = torch.zeros(
                        (num_tokens,), dtype=torch.int32, device=weight.device
                    )
                    indexer._project_and_quantize_q(qr, position_ids)
                except Exception as e:
                    # Never fail startup for a prewarm miss; the kernel would
                    # simply be compiled later, as it is today.
                    logger.warning(
                        f"indexer-Q CuTe DSL prewarm skipped for {num_tokens} "
                        f"tokens. {type(e).__name__}: {e}"
                    )
        torch.cuda.synchronize()

        # Hold every rank here until the slowest has finished compiling, so the
        # first MoE all-to-all dispatch is entered without JIT skew.
        if self._ctx.deps.mapping.tp_size > 1 and self._ctx.deps.dist is not None:
            self._ctx.deps.dist.tp_allgather(1)
        logger.info("indexer-Q CuTe DSL prewarm complete")

    def _general_warmup_impl(
        self, resource_manager: ResourceManager, warmup_requests_configs: List[Tuple[int, int]]
    ) -> None:
        for num_tokens, num_gen_tokens in warmup_requests_configs:
            # Helix CP does not support warmup with context requests.
            if self._ctx.deps.mapping.has_cp_helix() and num_tokens != num_gen_tokens:
                continue
            try:
                with self._release_batch_context(
                    self._create_warmup_request(resource_manager, num_tokens, num_gen_tokens),
                    resource_manager,
                ) as batch:
                    if not self._should_run_warmup_batch(
                        batch,
                        num_tokens,
                        f"general, num_tokens={num_tokens}, num_gen_tokens={num_gen_tokens}",
                    ):
                        continue
                    logger.info(
                        f"Run warmup with {num_tokens} tokens, include {num_gen_tokens} generation tokens"
                    )
                    self._executor._run_batch(batch, resource_manager)
                    torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                if self._is_distributed_forward():
                    # Peers are inside the same forward's collectives and
                    # cannot follow a rank-local skip.
                    raise
                logger.warning(
                    f"OOM during general warmup with {num_tokens} tokens, "
                    f"{num_gen_tokens} generation tokens. Skipping."
                )
                # If the OOM aborted the forward between dispatch() and
                # combine(), the MoE A2A state machines are stuck in
                # ``dispatched`` and the next warmup will hit
                # ``dispatch called twice``. Reset them before retrying a
                # smaller shape.
                self._reset_moe_alltoall_state()
                torch.cuda.empty_cache()

    def _run_cuda_graph_warmup(self, resource_manager: ResourceManager):
        """Warm up or capture CUDA graphs for the configured graph shapes."""
        if not (
            self._ctx.cuda_graph_runner.enabled
            or self._ctx.config.prefill_cuda_graph_backend != PrefillCudaGraphBackend.DISABLED
        ):
            return

        from .....modules.linear import (
            MXFP8LinearMethod,
            flashinfer_mxfp8_autotune,
            flashinfer_mxfp8_decode_graph_capture,
        )

        # The automatic MiniMax-M3 MXFP8 selection is decode-graph-only.
        # Tune every generation graph shape during the warmup-only pass. Keep
        # piecewise context/prefill graph capture on the native backend.
        flashinfer_methods = [
            quant_method
            for module in self._ctx.deps.model.modules()
            if isinstance(
                (quant_method := getattr(module, "quant_method", None)), MXFP8LinearMethod
            )
            and quant_method.needs_flashinfer_autotune
        ]
        flashinfer_autotune_context = (
            flashinfer_mxfp8_autotune()
            if self._ctx.cuda_graph_runner.is_warmup_only and flashinfer_methods
            else contextlib.nullcontext()
        )
        with flashinfer_autotune_context, flashinfer_mxfp8_decode_graph_capture():
            self._capture_generation_cuda_graphs(resource_manager)
        self._capture_mixed_cuda_graphs(resource_manager)
        # Piecewise graphs have separate capture machinery and do not use the
        # whole-model attention workspace. Capture them only on the second pass.
        if not self._ctx.cuda_graph_runner.is_warmup_only:
            self._capture_prefill_cuda_graphs(resource_manager)

    def _assert_all_tp_ranks_have_warmup_batch(self, batch, num_tokens: int) -> None:
        """Assert every TP rank has a valid warmup batch, or raise with diagnostics.

        Under attention-DP, each rank's KV cache available capacity can differ at
        runtime, causing _create_warmup_request to return None on some ranks while
        others proceed into forward() with tp_comm collectives — deadlocking the
        job. This check prevents the deadlock by failing early with diagnostic info.

        ``tp_size`` alone does not establish that peers are reachable: ``dist``
        is optional, and without a communicator there is no tp_comm collective
        to deadlock in.
        """
        if self._ctx.deps.mapping.tp_size <= 1 or self._ctx.deps.dist is None:
            return
        has_batch = int(batch is not None)
        all_flags = list(self._ctx.deps.dist.tp_allgather(has_batch))
        if any(all_flags) and not all(all_flags):
            # Gather token counts for diagnostics
            all_tokens = list(self._ctx.deps.dist.tp_allgather(num_tokens))
            failed_ranks = [i for i, f in enumerate(all_flags) if not f]
            raise RuntimeError(
                f"Warmup batch creation failed on TP rank(s) {failed_ranks} "
                f"but succeeded on others. This would cause a collective "
                f"deadlock. Per-rank curr_max_num_tokens: {all_tokens}. "
                f"This indicates asymmetric KV cache capacity across TP ranks. "
                f"Consider increasing --kv_cache_free_gpu_mem_fraction."
            )

    def _agree_warmup_shapes(self, configs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Reduce warmup shapes to the ones every rank in the group proposed.

        ``_get_max_shape_warmup_requests`` derives shapes from this rank's free
        KV capacity, so under attention-DP the values -- and, once
        ``dict.fromkeys`` drops a collision, the length -- differ per rank.
        Ranks would then walk different loops and meet in different forwards.

        The intersection keeps rank 0's ordering, which matters because the
        general warmup list is ordered for torch.compile specialization.
        """
        allgather = self._warmup_agreement_allgather()
        if allgather is None:
            return configs
        per_rank = allgather([list(config) for config in configs])
        shared = set.intersection(
            *({tuple(config) for config in rank_configs} for rank_configs in per_rank)
        )
        agreed = [config for config in configs if config in shared]
        dropped = [config for config in configs if config not in shared]
        if dropped:
            logger.warning(
                f"Dropping warmup shapes {dropped} that not every rank could "
                f"propose; per-rank KV capacity differs. Remaining: {agreed}."
            )
        return agreed

    def _warmup_cute_dsl_radix_topk(self) -> None:
        """Pre-compile the DSA radix-filter CuTe DSL decode top-k for every
        cluster_size band during warmup, before serving.

        Captured geometries are already compiled by the warmup-step forwards;
        this fills in the bands the eager (non-captured) decode path can still
        hit (mixed prefill+decode batch, or cuda_graph disabled) so they do
        not pay a first-touch JIT stall on a live request. DSA-specific params
        live on the metadata, so delegate to it. No-op on non-DSA models.
        """
        attn_meta = getattr(self, "attn_metadata", None)
        if attn_meta is None:
            return
        try:
            from .....attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
        except ImportError:
            return
        if isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            next_n = 1 + self._ctx.config.original_max_draft_len
            attn_meta.warmup_cute_dsl_radix_topk(next_n)
            if hasattr(attn_meta, "warmup_selfsampling_topk"):
                attn_meta.warmup_selfsampling_topk(
                    next_n, batch_sizes=self._ctx.config.decoder_cuda_graph_batch_sizes
                )

    def _get_max_shape_warmup_requests(
        self, resource_manager: ResourceManager
    ) -> List[Tuple[int, int]]:
        """
        Returns warmup configs covering the maximum context and generation shapes.
        """

        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        token_num_upper_bound = min(
            self._ctx.runner_config.max_num_tokens,
            self._ctx.runner_config.max_batch_size * (self._ctx.runner_config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._ctx.config.original_max_draft_len,
        )
        max_batch_size = min(
            self._ctx.runner_config.max_batch_size,
            curr_max_num_tokens
            // (1 + self._ctx.config.max_draft_loop_tokens)
            // self._ctx.runner_config.max_beam_width,
        )

        warmup_requests_configs = [
            (curr_max_num_tokens, 0),  # max_num_tokens, pure context
            (max_batch_size, max_batch_size),  # max_batch_size, pure generation
        ]

        return warmup_requests_configs

    def _ensure_dsa_attn_metadata_for_warmup(self, resource_manager: ResourceManager) -> None:
        """Build the DSA attention metadata if no warmup forward created it, so
        the top-K pre-compile hooks still run (draft engine, guided decoder or
        context-only server without general warmup). No-op unless DSA."""
        if getattr(self, "attn_metadata", None) is not None:
            return
        try:
            from .....attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
        except ImportError:
            return
        metadata_cls = getattr(self._ctx.runner_config.attention_backend, "Metadata", None)
        if metadata_cls is None or not issubclass(metadata_cls, DSAtrtllmAttentionMetadata):
            return
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        if kv_cache_manager is None:
            return
        self._ctx.set_up_attn_metadata(
            kv_cache_manager, self._ctx.get_draft_kv_cache_manager(resource_manager)
        )

    @contextlib.contextmanager
    def _release_batch_context(
        self, batch: Optional[ScheduledRequests], resource_manager: ResourceManager
    ):
        """A context manager to automatically free resources of a dummy batch."""
        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        draft_kv_cache_manager = self._ctx.get_draft_kv_cache_manager(resource_manager)
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER
        )
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        try:
            yield batch
        finally:
            if batch is not None and kv_cache_manager is not None:
                for req in batch.all_requests():
                    kv_cache_manager.free_resources(req)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(req)
                    if cross_kv_cache_manager is not None:
                        cross_kv_cache_manager.free_resources(req)
                    if spec_resource_manager is not None:
                        spec_resource_manager.free_resources(req)

    def _reset_moe_alltoall_state(self) -> None:
        """Reset all MoE all-to-all state machines reachable from the model.

        Each MoE backend keeps a small dispatch/combine phase state per layer
        (``MoeAlltoAll`` or ``NVLinkOneSided``). A forward that calls
        ``dispatch`` but raises before reaching ``combine`` (e.g., a warmup
        OOM mid-MoE) leaves that state in ``dispatched``, which fails the
        invariant on the next ``dispatch`` call. This helper walks the model
        and resets any A2A state found, so subsequent forwards start clean.
        """
        for module in self._ctx.deps.model.modules():
            for attr_name in ("moe_a2a", "comm"):
                obj = getattr(module, attr_name, None)
                reset = getattr(obj, "reset_state", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"Failed to reset MoE A2A state on {type(module).__name__}.{attr_name}: {e}"
                        )

    def _get_full_general_warmup_requests(
        self, resource_manager: ResourceManager
    ) -> List[Tuple[int, int]]:
        """
        Returns the ordered warmup configs for torch.compile specialization.

        Covers 1-token (0-1 graph specialization), max-shape (best triton autotuning),
        and small-context (2-token path) cases.
        """
        max_configs = self._get_max_shape_warmup_requests(resource_manager)
        # Specialize for 1 token pure ctx and pure gen
        one_token_configs = [(1, 0), (1, 1)]
        # Small ctx specialization
        small_ctx_configs = [(2, 0)]

        # Ordering matters for torch.compile graph specialization:
        # 1-token first to capture the 0→1 transition graph; max-shape next to seed
        # triton autotuning with the largest inputs; 2-token last for the small-ctx path.
        warmup_configs = one_token_configs + max_configs + small_ctx_configs
        # Deduplicate the warmup_configs while keeping the order.
        return list(dict.fromkeys(warmup_configs))

    @contextmanager
    def maybe_autotune_lora(self):
        """Enable autotuning while warming up CUDA-graph LoRA kernels."""
        if not (
            self._ctx.runner_config.enable_autotuner
            and self._ctx.state.cuda_graph_lora_manager is not None
        ):
            yield
            return

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        with autotune(cache_path=cache_path):
            try:
                yield
            finally:
                # Complete the PP cache hand-off even on ranks without a
                # CUDA-graph-only tunable op.
                autotuner = AutoTuner.get()
                autotuner.cache_pp_recv()
                autotuner.cache_pp_send()
                autotuner.clean_pp_flag()

    def _warmup_agreement_allgather(self) -> Optional[Callable[[int], List[int]]]:
        """Return an allgather over the ranks this warmup forward synchronizes with.

        ``None`` means agreement cannot be established here, so a missing batch
        stays fatal rather than being skipped unilaterally.

        DWDP is the case that cannot be answered: its peers are reached through
        a ``COMM_WORLD``-derived subgroup built in ``dwdp.py``, not through
        ``self._ctx.deps.dist``, so a ``self._ctx.deps.dist`` allgather would report a unanimity it
        never observed.
        """
        if self._ctx.deps.dist is None or self._ctx.deps.mapping.dwdp_enabled:
            return None
        if self._ctx.deps.dist.world_size <= 1:
            return None
        return self._ctx.deps.dist.allgather

    def _agree_warmup_flag(self, flag: bool) -> bool:
        """Reduce a phase-entry decision to one the whole forward group shares.

        Several predicates that gate a warmup phase are rank-local. The
        capturable guided decoder is installed only on the last pipeline rank,
        so ``guided_decoder is None`` -- and through it
        ``can_run_general_warmup`` -- differs across pipeline stages. Mamba's
        entry test reads this rank's free KV capacity. Letting one rank enter a
        phase its peers skip strands whoever enters the collective, and it also
        unbalances the per-shape agreement below.

        Any rank opting out takes the whole group out with it.
        """
        allgather = self._warmup_agreement_allgather()
        if allgather is None:
            return flag
        return all(allgather(int(flag)))

    def _update_draft_inference_state_for_warmup(
        self, batch: ScheduledRequests, is_first_draft: bool, resource_manager: ResourceManager
    ):
        """Updates request states for specific draft model warmups like Eagle3."""
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        if self._ctx.config.is_draft_model and isinstance(
            spec_resource_manager, Eagle3ResourceManager
        ):
            spec_resource_manager.is_first_draft = is_first_draft
            if is_first_draft:
                for req in batch.generation_requests:
                    req.py_is_first_draft = True
                    req.py_draft_tokens = []

    def _general_warmup(
        self, resource_manager: ResourceManager, warmup_requests_configs: List[Tuple[int, int]]
    ):
        """
        Runs forward passes for each config in warmup_requests_configs.

        Serves both torch.compile graph specialization and memory pool pre-population.
        """
        # Disable CUDA graph replay during general warmup to avoid replaying
        # graphs with stale KV cache block offsets from capture time.
        with self.no_cuda_graph():
            self._general_warmup_impl(resource_manager, warmup_requests_configs)

    @contextlib.contextmanager
    def no_cuda_graph(self):
        if self._ctx.cuda_graph_runner is None:
            yield
            return
        _run_cuda_graphs = self._ctx.cuda_graph_runner.enabled
        self._ctx.cuda_graph_runner.enabled = False
        try:
            yield
        finally:
            self._ctx.cuda_graph_runner.enabled = _run_cuda_graphs

    def _is_distributed_forward(self) -> bool:
        """Return whether model forward can communicate with peer workers.

        ``dist`` is optional. An engine built without a communicator cannot
        enter a collective at all, so it has no peers to strand and every
        warmup failure stays rank-local.
        """
        if self._ctx.deps.dist is None:
            return False
        return self._ctx.deps.dist.world_size > 1 or self._ctx.deps.mapping.dwdp_enabled

    @staticmethod
    def _release_megamoe_profiling_scratch():
        # MegaMoE tuning resources are shared across layers, so only the engine
        # can release them after its full autotune warmup and before graph
        # capture. Later eviction could invalidate a captured workspace pointer.
        from .....moe.custom_ops import cute_dsl_megamoe_custom_op as _megamoe_op

        release_megamoe_scratch = getattr(_megamoe_op, "release_megamoe_profiling_scratch", None)
        if release_megamoe_scratch is not None:
            release_megamoe_scratch()

    @contextmanager
    def _spec_decode_override(self, *, enable: bool, draft_len: int):
        saved = (self._ctx.state.enable_spec_decode, self._ctx.state.runtime_draft_len)
        self._ctx.state.enable_spec_decode = enable
        self._ctx.state.runtime_draft_len = draft_len
        try:
            yield
        finally:
            self._ctx.state.enable_spec_decode, self._ctx.state.runtime_draft_len = saved

    # ---- where encoder-decoder specializes warmup and capture ----
    def _max_encoder_output_len(self, resource_manager):
        """Decoder-only models have no encoder output."""
        return None

    def _enc_dec_hidden_size(self) -> int:
        return 0

    def _add_cross_dummy_requests(self, requests, resource_manager) -> bool:
        """No cross-KV cache to seed."""
        return True

    def _populate_cross_kv_cache(self, projection_inputs) -> None:
        return None

    def _capture_mixed_cuda_graphs(self, resource_manager) -> None:
        """No mixed encoder/decoder shapes to capture."""
        return None
