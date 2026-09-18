# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import os
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._utils import prefer_pinned, release_gc
from tensorrt_llm.inputs.registry import (BaseMultimodalInputProcessor,
                                          create_input_processor)
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, DecodingBaseConfig,
                                          EncodeCudaGraphConfig,
                                          PrefillCudaGraphBackend,
                                          TorchCompileConfig, TorchLlmArgs)

# isort: split
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention.backends.interface import AttentionRuntimeFeatures
from ..attention.backends.trtllm import TrtllmAttentionMetadata
from ..attention.backends.utils import get_attention_backend
from ..compilation.backend import Backend
from ..distributed import Distributed
from ..distributed.communicator import init_pp_comm
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_multimodal_mixin import MultimodalModelMixin
from ..models.modeling_utils import DecoderModelForCausalLM
from ..moe.expert_statistic import ExpertStatistic
from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from ..peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from ..speculative import update_spec_config_from_loaded_model
from ..utils import set_torch_compiling, with_model_extra_attrs
from .breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from .cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from .engine.cuda_graph import filter_cuda_graph_batch_sizes
from .engine.lora import (LoraParamBuilder, make_cuda_graph_lora_manager,
                          make_lora_model_config)
from .engine.model_call import ModelCaller
from .engine.multimodal import (MultimodalItemScheduler, is_multimodal,
                                mm_encoder_cache_enabled,
                                setup_mm_encoder_attn_metadata)
from .engine.runners import resolve_runner_type
from .engine.runners.common import _set_moe_a2a_warmup
from .engine.runners.decoder import (DecoderBuffers, DecoderRunner,
                                     DecoderRunnerConfig)
from .engine.runners.encoder import EncoderRunner, EncoderRunnerConfig
from .engine.runners.encoder_decoder import (EncoderDecoderRunner,
                                             EncoderDecoderRunnerConfig)
from .engine.runners.interface import (ModelRunner, PackedEncoderBatch,
                                       PackedModelRunner, RunnerDeps)
from .engine.runners.no_kv_cache import NoKVCacheRunner, NoKVCacheRunnerConfig
from .engine.spec_decode import SpecMetadataBuilder
from .guided_decoder import CapturableGuidedDecoder
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .llm_request import LlmRequest
from .model_loader import ModelLoader, _construct_checkpoint_loader
from .resource_manager import ResourceManager, ResourceManagerType
from .sampler import SampleStateTensors
from .sampler.sampler_common import SampleType
from .scheduler import ScheduledRequests


class ModelEngine(ABC):

    @abstractmethod
    def get_max_num_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: Optional[ResourceManager],
                new_tensors_device: Optional[SampleStateTensors],
                gather_context_logits: bool = False,
                cache_indirection_buffer: Optional[torch.Tensor] = None,
                num_accepted_tokens_device: Optional[torch.Tensor] = None):
        raise NotImplementedError

    def warmup(self, resource_manager: Optional[ResourceManager]) -> None:
        """
        This method is called after runtime resources are initialized. The
        resource manager is absent for drivers that allocate none. Override to
        perform warmup actions: instantiating CUDA graphs, torch.compile, etc.
        """
        return


def _filter_piecewise_capture_num_tokens(
    candidate_num_tokens: list[int],
    max_num_tokens: int,
    max_batch_size: int,
    max_seq_len: int,
) -> Tuple[list[int], list[int]]:
    """Cap piecewise CUDA graph capture candidates at the engine's reachable
    `num_tokens` ceiling `max_batch_size * (max_seq_len - 1)`
    clamping user-requested sizes above it down to the ceiling.

    Each in-flight request must leave room for at least one decode token,
    so the ceiling is the largest forward-pass `num_tokens` the warmup
    builder can construct. Candidates above the ceiling cannot be
    recorded; clamping them down to the ceiling preserves the user's
    intent (a requested 128 becomes 127 when only 127 is recordable)
    without inventing capture sizes the user never asked
    for. Appending sizes beyond the user's list is harmful: runtime
    padding rounds iterations up to the nearest captured size, so a far
    appended ceiling (e.g. 65536 over a list topping at 13914) would
    make every iteration in the gap execute the full ceiling shape.

    Returns `(kept, unrecordable)` where `kept` is sorted ascending and
    deduped, with above-ceiling candidates clamped to the ceiling.
    `unrecordable` is the sorted unique set of input entries above the
    ceiling but within `max_num_tokens` (the clamped ones, reported so
    the caller's warning fires).
    """
    max_capturable_num_tokens = max(0, max_batch_size * (max_seq_len - 1))
    piecewise_capacity_limit = min(max_num_tokens, max_capturable_num_tokens)
    if piecewise_capacity_limit > 0:
        kept = sorted({
            min(i, piecewise_capacity_limit)
            for i in candidate_num_tokens if 0 < i <= max_num_tokens
        })
    else:
        kept = []
    unrecordable = sorted({
        i
        for i in candidate_num_tokens
        if max_capturable_num_tokens < i <= max_num_tokens
    })
    return kept, unrecordable


# BCG uses the same capture-bucket filtering semantics as PCG.
_filter_prefill_capture_num_tokens = _filter_piecewise_capture_num_tokens

_DEEP_GEMM_PDL_CONFIGURED = False

# Arbitrary non-greedy params used to force the advanced-sampling CUDA graph
# warmup capture path.


def _configure_deep_gemm_pdl() -> None:
    global _DEEP_GEMM_PDL_CONFIGURED
    if _DEEP_GEMM_PDL_CONFIGURED:
        return

    from tensorrt_llm import deep_gemm

    deep_gemm.set_pdl(os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
    _DEEP_GEMM_PDL_CONFIGURED = True


class PyTorchModelEngine(ModelEngine):

    def __init__(
        self,
        *,
        model_path: str,
        llm_args: TorchLlmArgs,
        mapping: Optional[Mapping] = None,
        attn_runtime_features: Optional[AttentionRuntimeFeatures] = None,
        dist: Optional[Distributed] = None,
        spec_config: Optional[DecodingBaseConfig] = None,
        is_draft_model: bool = False,
        model: Optional[torch.nn.Module] = None,
        checkpoint_loader: Optional[BaseCheckpointLoader] = None,
        model_weights_memory_tag: Optional[str] = None,
        model_weights_restore_mode=None,
    ):
        _configure_deep_gemm_pdl()

        self._cleanup_done = False
        self._runner: Optional[Union[ModelRunner, PackedModelRunner]] = None
        # Transitional snapshot for decoder capture and encoder scheduling.
        # Encoder graph resources and lifecycle remain entirely runner-owned.
        self._runner: Optional[Union[ModelRunner, PackedModelRunner]] = None
        # Optional in-graph sampling hook, registered by PyExecutor. Called at
        # the tail of _forward_step -- i.e. right after the LM head, and inside
        # capture_forward_fn -- so that for graph-capturable sampling tiers the
        # sampling lands at the end of the captured forward graph instead of
        # costing a separate launch. Left None when the sampler does not opt in.
        # Stages (or clears) in-graph sampling state once the batch is settled.
        if llm_args.encode_only and llm_args.mm_encoder_only:
            raise ValueError(
                "encode_only and mm_encoder_only are mutually exclusive.")
        (
            max_beam_width,
            max_num_tokens,
            max_seq_len,
            max_batch_size,
        ) = llm_args.get_runtime_sizes()

        self.batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.max_beam_width = max_beam_width
        self.encoder_batch_size = (llm_args.encoder_max_batch_size
                                   if llm_args.encoder_max_batch_size
                                   is not None else self.batch_size)
        # The multimodal encoder token budget falls back to the LLM-side value
        # when unset. It may be raised after model load because atomic MM items
        # cannot be split.
        self.encoder_max_num_tokens = (llm_args.encoder_max_num_tokens
                                       if llm_args.encoder_max_num_tokens
                                       is not None else self.max_num_tokens)

        if checkpoint_loader is None:
            checkpoint_loader = _construct_checkpoint_loader(
                llm_args.backend,
                llm_args.checkpoint_loader,
                llm_args.checkpoint_format,
                mx_config=llm_args.mx_config,
                checkpoint_io_policy=llm_args.checkpoint_io_policy,
                load_format=llm_args.load_format,
                partial_model_loading=llm_args.is_partial_model_loading,
            )

        self.mapping = mapping
        if mapping.has_pp():
            init_pp_comm(mapping)
        # Disaggregated attention-DP can backfill a batch before the overlap
        # scheduler releases the previous batch's terminal sequence slots.
        from ._util import (compute_max_num_sequences,
                            should_enable_adp_dummy_fixes,
                            should_enable_disagg_adp_overlap_headroom,
                            should_enable_non_overlap_adp_forward_intent,
                            should_enable_scheduler_aware_adp_dummy)
        self._enable_disagg_adp_overlap_headroom = (
            should_enable_disagg_adp_overlap_headroom(
                mapping, llm_args.cache_transceiver_config,
                llm_args.disable_overlap_scheduler))
        self._enable_adp_dummy_fixes = should_enable_adp_dummy_fixes(mapping)
        self.max_num_seq_slots = compute_max_num_sequences(
            mapping,
            self.batch_size,
            llm_args.disable_overlap_scheduler,
            enable_overlap_headroom=self._enable_disagg_adp_overlap_headroom,
        )
        self.dist = dist
        if dist is not None:
            ExpertStatistic.create(self.dist.rank)
        self.llm_args = llm_args
        # Opt-in tiered sampling captured into the forward graph. Off by
        # default: it captures one extra graph per enabled tier, which costs
        # startup time and memory that deployments not bound by sampling
        # overhead should not pay.
        self.enable_in_graph_sampling = bool(
            getattr(llm_args, "enable_in_graph_sampling", False))
        self.original_max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.original_max_total_draft_tokens = (
            spec_config.tokens_per_gen_step -
            1) if spec_config is not None else 0
        # Saved before zeroing for draft models; used by update_spec_dec_param.
        self._spec_dec_max_total_draft_tokens = (
            spec_config.max_total_draft_tokens
            if spec_config is not None else 0)

        # Dynamic tree draft loop produces up to K * max_draft_len tokens,
        # which may exceed max_total_draft_tokens. Use the larger value for
        # KV cache reservation only; verify/tree output stays at max_total_draft_tokens.
        if (spec_config is not None
                and getattr(spec_config, 'use_dynamic_tree', False)
                and getattr(spec_config, 'dynamic_tree_max_topK', 0) > 0):
            self.max_draft_loop_tokens = max(
                self.original_max_total_draft_tokens,
                spec_config.dynamic_tree_max_topK * spec_config.max_draft_len)
        else:
            self.max_draft_loop_tokens = self.original_max_total_draft_tokens

        # The draft model won't have any draft tokens attached to
        # generation requests when we invoke it autoregressively
        if spec_config is not None and is_draft_model:
            spec_config.max_draft_len = 0
            spec_config.max_total_draft_tokens = 0
        self.spec_config = spec_config
        self.is_spec_decode = spec_config is not None
        self.sparse_attention_config = None if is_draft_model else llm_args.sparse_attention_config
        self.is_draft_model = is_draft_model

        self.attn_runtime_features = attn_runtime_features or AttentionRuntimeFeatures(
        )

        input_processor_kwargs = {}
        video_pruning_rate = llm_args.multimodal_config.video_pruning_rate
        if video_pruning_rate is not None:
            input_processor_kwargs['video_pruning_rate'] = video_pruning_rate
        self.input_processor = create_input_processor(
            model_path,
            tokenizer=None,
            checkpoint_format=llm_args.checkpoint_format,
            trust_remote_code=llm_args.trust_remote_code,
            **input_processor_kwargs)

        self.moe_load_balancer: Optional[MoeLoadBalancer] = None
        self.model_loader: Optional[ModelLoader] = None
        if model is None:
            lora_config: Optional[
                LoraConfig] = None if is_draft_model else llm_args.lora_config
            # Keep the model_loader to support reloading the model weights later
            self.model_loader = ModelLoader(
                llm_args=llm_args,
                mapping=self.mapping,
                spec_config=self.spec_config,
                sparse_attention_config=self.sparse_attention_config,
                max_num_tokens=self.max_num_tokens,
                max_seq_len=self.max_seq_len,
                lora_config=lora_config,
                model_weights_memory_tag=model_weights_memory_tag,
                model_weights_restore_mode=model_weights_restore_mode,
            )
            # Open checkpoint and load the LLM module object.
            self.model, moe_load_balancer = self.model_loader.load(
                checkpoint_dir=model_path, checkpoint_loader=checkpoint_loader)
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                self.moe_load_balancer = moe_load_balancer
        else:
            self.model = model
        self._validate_breakable_cuda_graph_compatibility()
        # In-graph sampling needs the full vocabulary: top-k / top-p over a
        # tensor-parallel shard would rank against a slice of the logits and
        # emit the wrong token, silently. The LM head gathers by default, but
        # the draft models of some speculation modes turn that off, so refuse
        # the fast path rather than sample a sharded row.
        if self.enable_in_graph_sampling and not getattr(
                self.model.model_config, "lm_head_gather_output", True):
            logger.warning(
                "Disabling enable_in_graph_sampling: this model's LM head does not "
                "gather its output, so the logits are sharded across tensor "
                "parallel ranks and cannot be sampled in-graph.")
            self.enable_in_graph_sampling = False
        pretrained_config = self.model.model_config.pretrained_config
        model_type = getattr(pretrained_config, "model_type", None)
        self._enable_scheduler_aware_adp_dummy = (
            should_enable_scheduler_aware_adp_dummy(
                model_type, mapping, llm_args.disable_overlap_scheduler))
        self._enable_non_overlap_adp_forward_intent = (
            should_enable_non_overlap_adp_forward_intent(
                mapping, llm_args.disable_overlap_scheduler))
        self.sparse_attention_config = self.model.model_config.sparse_attention_config
        # In case that some tests use stub models and override `_load_model`.
        if not hasattr(self.model, 'extra_attrs'):
            self.model.extra_attrs = {}
        # Every MM item-scheduling decision -- policy, capability, feature
        # validation, budget resolution -- lives in engine/multimodal.py; the
        # engine only copies back the three budgets that are external contract.
        mm_item_scheduler = MultimodalItemScheduler.maybe_create(
            llm_args=self.llm_args,
            model=self.model,
            input_processor=self.input_processor,
            encoder_max_num_tokens=self.encoder_max_num_tokens)
        self._mm_item_scheduler = mm_item_scheduler
        # `getattr`-read by py_executor.py and _util.py.
        self.mm_encoder_item_scheduling_enabled = mm_item_scheduler is not None
        self.mm_encoder_output_budget_bytes: Optional[int] = None
        if mm_item_scheduler is not None:
            # The raised encoder token budget is read back off the engine by
            # `_util.py`, and sizes the encoder metadata set up below.
            self.encoder_max_num_tokens = mm_item_scheduler.encoder_max_num_tokens
            self.mm_encoder_output_budget_bytes = mm_item_scheduler.output_budget_bytes
            # Absent, not None, when item scheduling is off: external readers
            # rely on the `getattr` default.
            self.bytes_per_mm_encoder_embedding = mm_item_scheduler.bytes_per_embedding
        setup_mm_encoder_attn_metadata(
            self.model, self.input_processor, self.encoder_max_num_tokens,
            mm_item_scheduler.attention_metadata_capacity
            if mm_item_scheduler is not None else None)
        if self.llm_args.enable_layerwise_nvtx_marker:
            layerwise_nvtx_marker = LayerwiseNvtxMarker()
            module_prefix = 'Model'
            if self.model.model_config and self.model.model_config.pretrained_config and self.model.model_config.pretrained_config.architectures:
                module_prefix = '|'.join(
                    self.model.model_config.pretrained_config.architectures)
            layerwise_nvtx_marker.register_hooks(self.model, module_prefix)

        self.enable_attention_dp = self.model.model_config.mapping.enable_attention_dp
        self._disable_overlap_scheduler = self.llm_args.disable_overlap_scheduler
        self._torch_compile_backend = None
        self.dtype = self.model.config.torch_dtype
        self._init_model_capacity()

        self.cuda_graph_config = self.llm_args.cuda_graph_config
        self._is_encode_only = self.llm_args.encode_only

        if (isinstance(self.cuda_graph_config, EncodeCudaGraphConfig)
                and self._is_encoder_decoder_model()):
            logger.warning(
                "EncodeCudaGraphConfig is not supported for encoder-decoder "
                "models through cuda_graph_config. Use DecodeCudaGraphConfig "
                "for cuda_graph_config and configure encoder graphs through "
                "encoder_cuda_graph_config. Decoder CUDA graphs will be "
                "disabled.")
            self.cuda_graph_config = None

        cuda_graph_batch_sizes = self.cuda_graph_config.batch_sizes if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'batch_sizes'].default
        cuda_graph_padding_enabled = self.cuda_graph_config.enable_padding if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'enable_padding'].default

        self._cuda_graph_padding_enabled = cuda_graph_padding_enabled

        decode_tokens_per_request = 1 + self.original_max_total_draft_tokens
        self._cuda_graph_batch_sizes = filter_cuda_graph_batch_sizes(
            cuda_graph_batch_sizes, self.batch_size, self.max_num_tokens,
            decode_tokens_per_request,
            self._cuda_graph_padding_enabled) if cuda_graph_batch_sizes else []

        self._max_cuda_graph_batch_size = (self._cuda_graph_batch_sizes[-1] if
                                           self._cuda_graph_batch_sizes else 0)

        self.torch_compile_config = self.llm_args.torch_compile_config
        self.prefill_cuda_graph_backend = self.llm_args.prefill_cuda_graph_backend
        torch_compile_enabled = bool(self.torch_compile_config is not None)
        torch_compile_fullgraph = self.torch_compile_config.enable_fullgraph if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_fullgraph'].default
        torch_compile_inductor_enabled = self.torch_compile_config.enable_inductor if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_inductor'].default
        torch_compile_piecewise_cuda_graph = (self.prefill_cuda_graph_backend ==
                                              PrefillCudaGraphBackend.PIECEWISE)
        torch_compile_enable_userbuffers = self.torch_compile_config.enable_userbuffers if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_userbuffers'].default
        torch_compile_max_num_streams = self.torch_compile_config.max_num_streams if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'max_num_streams'].default

        self._torch_compile_enabled = torch_compile_enabled
        self._torch_compile_piecewise_cuda_graph = torch_compile_piecewise_cuda_graph

        prefill_cuda_graph_num_tokens = self.llm_args.prefill_capture_num_tokens
        if prefill_cuda_graph_num_tokens is None:
            prefill_cuda_graph_num_tokens = cuda_graph_batch_sizes or []

        self._prefill_cuda_graph_num_tokens, unrecordable = (
            _filter_prefill_capture_num_tokens(
                prefill_cuda_graph_num_tokens,
                max_num_tokens=self.max_num_tokens,
                max_batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
            ))
        if unrecordable:
            logger.warning(
                f"Skipping prefill CUDA graph capture for num_tokens="
                f"{unrecordable}: exceeds reachable ceiling "
                f"max_batch_size*(max_seq_len-1)="
                f"{max(0, self.batch_size * (self.max_seq_len - 1))}. "
                f"Clamping them to the ceiling; raise max_seq_len for larger graphs."
            )

        try:
            use_ub_for_nccl = (
                self.llm_args.allreduce_strategy == "NCCL_SYMMETRIC"
                and self._init_userbuffers(self.model.config.hidden_size))
            if self._torch_compile_enabled:
                set_torch_compiling(True)
                use_ub = not use_ub_for_nccl and (
                    torch_compile_enable_userbuffers
                    and self._init_userbuffers(self.model.config.hidden_size))
                self.backend_num_streams = Backend.Streams([
                    torch.cuda.Stream()
                    for _ in range(torch_compile_max_num_streams - 1)
                ])
                self._torch_compile_backend = Backend(
                    torch_compile_inductor_enabled,
                    enable_userbuffers=use_ub,
                    enable_piecewise_cuda_graph=self.
                    _torch_compile_piecewise_cuda_graph,
                    capture_num_tokens=self._prefill_cuda_graph_num_tokens,
                    max_num_streams=torch_compile_max_num_streams,
                    mapping=self.mapping)
                apply_llm_torch_compile = getattr(self.model,
                                                  "apply_llm_torch_compile",
                                                  None)
                if isinstance(self.model, DecoderModelForCausalLM):
                    self.model.model = torch.compile(
                        self.model.model,
                        backend=self._torch_compile_backend,
                        fullgraph=torch_compile_fullgraph)
                elif callable(apply_llm_torch_compile):
                    # TODO: Move this contract to MultimodalModelMixin once
                    # multimodal models consistently expose their LLM compile
                    # scope through the mixin.
                    apply_llm_torch_compile(backend=self._torch_compile_backend,
                                            fullgraph=torch_compile_fullgraph)
                else:
                    self.model = torch.compile(
                        self.model,
                        backend=self._torch_compile_backend,
                        fullgraph=torch_compile_fullgraph)
                torch._dynamo.config.cache_size_limit = 16
            else:
                set_torch_compiling(False)
        except Exception as e:
            import traceback
            traceback.print_exception(Exception, e, e.__traceback__)
            raise e

        self.is_warmup = False
        self.previous_request_ids = []

        self._encoder_decoder_host_buffer_pool: List[Dict[str, Any]] = []
        self._encoder_decoder_input_fast_path_static_eligible: Optional[
            bool] = None
        self._encoder_decoder_position_id_offset: Optional[int] = None

        sparse_params = (self.sparse_attention_config.to_sparse_params(
            pretrained_config=self.model.model_config.pretrained_config)
                         if self.sparse_attention_config is not None else None)
        self.attn_backend = get_attention_backend(self.llm_args.attn_backend,
                                                  sparse_params=sparse_params)

        self.get_runtime_tokens_per_gen_step = spec_config.get_runtime_tokens_per_gen_step if spec_config is not None else lambda runtime_draft_len: 1

        if self.is_spec_decode:
            if not self.is_draft_model:
                update_spec_config_from_loaded_model(self.spec_config,
                                                     self.model)
            max_num_draft_tokens = self.max_draft_loop_tokens * self.batch_size
            self.draft_tokens_cuda = torch.empty((max_num_draft_tokens, ),
                                                 dtype=torch.int,
                                                 device='cuda')
            self.gather_ids_cuda = torch.empty((self.max_num_tokens, ),
                                               dtype=torch.int,
                                               device='cuda')
            self.num_accepted_draft_tokens_cuda = torch.empty(
                (self.batch_size, ), dtype=torch.int, device='cuda')
            self.previous_pos_indices_cuda = torch.empty(
                (self.max_num_tokens, ), dtype=torch.int, device='cuda')
            self.previous_pos_id_offsets_cuda = torch.zeros(
                (self.max_num_tokens, ), dtype=torch.int, device='cuda')
            self.previous_kv_lens_offsets_cuda = torch.zeros(
                (self.batch_size, ), dtype=torch.int, device='cuda')
            self.without_logits = self.spec_config.spec_dec_mode.without_logits(
            )
            self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
            self.max_draft_len = spec_config.max_draft_len
            # Mutable per-iteration draft length (updated each iteration when
            # dynamic draft length is enabled; otherwise stays fixed).  Tree
            # modes verify all tree nodes per step, which can be wider than the
            # tree depth used by the drafter loop.
            self._initial_runtime_draft_len = (self.max_total_draft_tokens
                                               if not spec_config.is_linear_tree
                                               else self.max_draft_len)

        else:
            self.without_logits = False
            self.max_draft_len = 0
            self._initial_runtime_draft_len = 0
            self.max_total_draft_tokens = 0

        # This field is initialized lazily on the first forward pass.
        # This is convenient because:
        # 1) The attention metadata depends on the KV cache manager.
        # 2) The KV cache manager depends on the model configuration.
        # 3) The model configuration is not loaded until the model engine
        # is initialized.
        #
        # NOTE: This can be simplified by decoupling the model config loading and
        # the model engine.
        # Let the first CUDA graph capture create its private pool. Piecewise
        # CUDA graphs use a separate pool owned by their runners, so sharing a
        # pre-created pool handle with the outer graph runner is unnecessary.
        self._cuda_graph_mem_pool = None

        self._dynamic_draft_len_mapping = self._compute_dynamic_draft_len_mapping(
        )

        self.previous_batch_indices_cuda = torch.empty((self.max_num_tokens, ),
                                                       dtype=torch.int,
                                                       device='cuda')
        self._encoder_decoder_staged_request_ids: Optional[List[int]] = None
        self.input_ids_cuda = torch.empty((self.max_num_tokens, ),
                                          dtype=torch.int,
                                          device='cuda')
        self.position_ids_cuda = torch.empty((self.max_num_tokens, ),
                                             dtype=torch.int,
                                             device='cuda')
        # Steady-state generation-only prepare cache (non-speculative overlap
        # decode). Holds the per-request lists that are invariant while the
        # scheduled generation batch keeps the same composition, plus a pinned
        # cached-token counter advanced by one per step (host-side bookkeeping
        # only; the device position buffer is advanced in place and this
        # buffer is never the source of an async H2D). Invalidated (set to
        # None) by every full _prepare_tp_inputs pass.
        self._steady_gen_cache: Optional[Dict[str, Any]] = None
        self._steady_gen_positions_pinned = torch.empty(
            (self.max_num_tokens, ),
            dtype=torch.int,
            pin_memory=prefer_pinned())
        if self.use_mrope:
            self.mrope_position_ids_cuda = torch.empty(
                (3, 1, self.max_num_tokens), dtype=torch.int, device='cuda')

        # Pre-allocated buffers for draft model to avoid implicit synchronization
        # These are used to build index tensors without creating tensors from Python lists
        max_first_draft_tokens = self.batch_size * (
            self.original_max_total_draft_tokens +
            1) if spec_config else self.batch_size
        tokens_per_draft = self.original_max_total_draft_tokens + 1
        self.idx_accepted_tokens_cache = None
        self.draft_token_positions_cache = None
        if spec_config:
            # Cache for idx_accepted_tokens (pattern: 0,0,0...1,1,1...2,2,2...)
            self.idx_accepted_tokens_cache = torch.arange(
                max_first_draft_tokens, dtype=torch.long,
                device='cuda') // tokens_per_draft

        if self.is_draft_model:
            self.draft_ctx_token_indices_cuda = torch.empty((self.batch_size, ),
                                                            dtype=torch.long,
                                                            device='cuda')
            self.draft_ctx_seq_slots_cuda = torch.empty((self.batch_size, ),
                                                        dtype=torch.long,
                                                        device='cuda')
            # Buffers for first_draft requests (max_draft_len+1 tokens per request)
            self.draft_first_draft_indices_cuda = torch.empty(
                (max_first_draft_tokens, ), dtype=torch.long, device='cuda')
            self.draft_first_draft_seq_slots_cuda = torch.empty(
                (max_first_draft_tokens, ), dtype=torch.long, device='cuda')
            # Buffers for seq_slots and request indices
            self.draft_seq_slots_buffer_cuda = torch.empty((self.batch_size, ),
                                                           dtype=torch.int,
                                                           device='cuda')
            self.draft_request_indices_buffer_cuda = torch.empty(
                (self.batch_size, ), dtype=torch.int, device='cuda')

            # Pre-computed constant tensors for incremental update optimization
            # Cache for token_positions (pattern: 0,1,2...N repeated)
            self.draft_token_positions_cache = torch.arange(tokens_per_draft,
                                                            dtype=torch.long,
                                                            device='cuda')

        # We look up this key in resource_manager during forward to find the
        # kv cache manager. Can be changed to support multiple model engines
        # with different KV cache managers.
        self.kv_cache_manager_key = ResourceManagerType.DRAFT_KV_CACHE_MANAGER if is_draft_model else ResourceManagerType.KV_CACHE_MANAGER
        self.lora_model_config: Optional[LoraModelConfig] = None
        self._trtllm_gen_jit_warmup = False

        self.cuda_graph_lora_manager: Optional[CudaGraphLoraManager] = None
        self._force_lora_graph_for_capture: Optional[bool] = None
        self._lora = LoraParamBuilder(spec_config=self.spec_config,
                                      attn_backend=self.attn_backend)
        self._model_caller = ModelCaller(
            self.model,
            torch_compile_backend=self._torch_compile_backend,
            backend_num_streams=getattr(self, "backend_num_streams", None))
        self._spec = SpecMetadataBuilder(
            spec_config=self.spec_config,
            model_config=self.model.config,
            is_draft_model=self.is_draft_model,
            original_max_draft_len=self.original_max_draft_len,
            original_max_total_draft_tokens=self.
            original_max_total_draft_tokens,
            spec_dec_max_total_draft_tokens=self.
            _spec_dec_max_total_draft_tokens,
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            # The disaggregated attention-DP overlap path opts into larger
            # metadata buffers. None keeps the established max_num_requests
            # fallback for other configurations, including PP.
            num_seq_slots=(self.max_num_seq_slots if
                           self._enable_disagg_adp_overlap_headroom else None),
            attn_backend=self.attn_backend)
        # Sampling tier pinned during an in-graph sampling capture pass; None outside
        # capture, where the tier comes from the batch instead.
        self._capture_sample_type: Optional[SampleType] = None

        # Setup the local cache indirection buffer only once and reuse it.
        # This way it can also be used for CUDA graphs.
        if self.use_beam_search:
            self.cache_indirection_attention = torch.zeros(
                (self.batch_size, self.max_beam_width, self.max_seq_len),
                device="cuda",
                dtype=torch.int32)
        else:
            self.cache_indirection_attention = None

        runner_cls = resolve_runner_type(self.model, self.llm_args)
        self._runner = self._initialize_runner(runner_cls)

        self.cuda_graph_runner = self._initialize_cuda_graph_runner()
        self.breakable_cuda_graph_runner = \
            self._initialize_breakable_cuda_graph_runner()
        self._install_runner_collaborators()

        self.kv_cache_dtype_byte_size = self.get_kv_cache_dtype_byte_size()

        self._prepare_inputs_event: Optional[torch.cuda.Event] = None

        # Cache for enc-dec cross-attention stable generation steps.
        # Populated on the first CUDA-graph generation step; cleared whenever
        # the batch composition changes (new encoder request arrives).
        self._cross_attn_stable_cached_tokens: Optional[List[int]] = None
        self._cross_attn_stable_request_ids: Optional[List[int]] = None

    def _initialize_cuda_graph_runner(self) -> Optional[CUDAGraphRunner]:
        is_encoder_decoder = self._is_encoder_decoder_model()
        if not isinstance(self._runner, (DecoderRunner, EncoderDecoderRunner)):
            return None

        enable_encoder_decoder_mixed_cuda_graph = (
            is_encoder_decoder and bool(self._encoder_graph_shapes)
            and self.cuda_graph_config is not None
            and self.llm_args.enable_encoder_decoder_mixed_cuda_graph)

        config = CUDAGraphRunnerConfig(
            use_cuda_graph=(not self._is_encode_only
                            and self.cuda_graph_config is not None),
            cuda_graph_padding_enabled=self._cuda_graph_padding_enabled,
            cuda_graph_batch_sizes=self._cuda_graph_batch_sizes,
            max_cuda_graph_batch_size=self._max_cuda_graph_batch_size,
            max_beam_width=self.max_beam_width,
            spec_config=self.spec_config,
            cuda_graph_mem_pool=self._cuda_graph_mem_pool,
            dynamic_draft_len_mapping=self._dynamic_draft_len_mapping,
            max_num_tokens=self.max_num_tokens,
            use_mrope=self.use_mrope,
            original_max_draft_len=self.original_max_draft_len,
            original_max_total_draft_tokens=self.
            original_max_total_draft_tokens,
            is_draft_model=self.is_draft_model,
            enable_attention_dp=self.enable_attention_dp,
            is_encoder_decoder=is_encoder_decoder,
            batch_size=self.batch_size,
            mapping=self.mapping,
            dist=self.dist,
            kv_cache_manager_key=self.kv_cache_manager_key,
            sparse_attention_config=self.sparse_attention_config,
            enable_encoder_decoder_mixed_cuda_graph=(
                enable_encoder_decoder_mixed_cuda_graph),
            enable_in_graph_sampling=self.enable_in_graph_sampling,
        )
        return CUDAGraphRunner(config)

    def _initialize_breakable_cuda_graph_runner(
            self) -> Optional[BreakableCUDAGraphRunner]:
        if (self.cuda_graph_runner is None or self.prefill_cuda_graph_backend
                != PrefillCudaGraphBackend.BREAKABLE):
            return None

        decoder_model = (self.model if isinstance(
            self.model, DecoderModelForCausalLM) else getattr(
                self.model, "llm", None))
        if not isinstance(decoder_model, DecoderModelForCausalLM):
            raise ValueError(
                "breakable prefill CUDA graph requires a decoder model body")
        return BreakableCUDAGraphRunner(decoder_model.model)

    @property
    def _encoder_graph_shapes(self) -> frozenset:
        return getattr(self._runner, "_encoder_graph_shapes", frozenset())

    @property
    def _encoder_graph_batch_sizes(self) -> tuple:
        return getattr(self._runner, "_encoder_graph_batch_sizes", ())

    @property
    def _encoder_graph_pad_to_limit(self) -> bool:
        return getattr(self._runner, "_encoder_graph_pad_to_limit", False)

    @property
    def forward_pass_callable(self):
        return self._runner.forward_pass_callable

    @forward_pass_callable.setter
    def forward_pass_callable(self, value) -> None:
        self._runner.forward_pass_callable = value

    @property
    def sample_in_graph_callable(self):
        return self._runner.sample_in_graph_callable

    @sample_in_graph_callable.setter
    def sample_in_graph_callable(self, value) -> None:
        self._runner.sample_in_graph_callable = value

    @property
    def guided_decoder(self):
        return self._runner.guided_decoder

    @guided_decoder.setter
    def guided_decoder(self, value) -> None:
        self._runner.guided_decoder = value

    @property
    def iter_states(self):
        return self._runner.iter_states

    @iter_states.setter
    def iter_states(self, value) -> None:
        self._runner.iter_states = value

    @property
    def enable_spec_decode(self) -> bool:
        return self._runner._ctx.state.enable_spec_decode

    @enable_spec_decode.setter
    def enable_spec_decode(self, value: bool) -> None:
        self._runner._ctx.state.enable_spec_decode = value

    @property
    def runtime_draft_len(self) -> int:
        return self._runner._ctx.state.runtime_draft_len

    @runtime_draft_len.setter
    def runtime_draft_len(self, value: int) -> None:
        self._runner._ctx.state.runtime_draft_len = value

    @property
    def attn_metadata(self):
        return self._runner.attn_metadata

    @attn_metadata.setter
    def attn_metadata(self, value) -> None:
        self._runner.attn_metadata = value

    @property
    def spec_metadata(self):
        return self._runner.spec_metadata

    @spec_metadata.setter
    def spec_metadata(self, value) -> None:
        self._runner.spec_metadata = value

    def _initialize_runner(
        self, runner_cls: Optional[Type[Union[ModelRunner, PackedModelRunner]]]
    ) -> Optional[Union[ModelRunner, PackedModelRunner]]:
        if issubclass(runner_cls, EncoderRunner):
            return self._initialize_encoder_runner(runner_cls)
        if issubclass(runner_cls, EncoderDecoderRunner):
            return self._initialize_encoder_decoder_runner(runner_cls)
        if issubclass(runner_cls, NoKVCacheRunner):
            return self._initialize_no_kv_cache_runner(runner_cls)
        if issubclass(runner_cls, DecoderRunner):
            return self._initialize_decoder_runner(runner_cls)
        raise TypeError(f"No runner initializer registered for "
                        f"{runner_cls.__module__}.{runner_cls.__qualname__}")

    def _initialize_encoder_runner(
            self, runner_cls: Type[EncoderRunner]) -> EncoderRunner:
        runner_config = EncoderRunnerConfig.create(
            model=self.model,
            mapping=self.mapping,
            graph_config=(self.cuda_graph_config if isinstance(
                self.cuda_graph_config, EncodeCudaGraphConfig) else None),
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            attention_backend=self.attn_backend,
            attention_runtime_features=self.attn_runtime_features,
            enable_autotuner=self.llm_args.enable_autotuner,
            draft_model=self.is_draft_model,
        )
        return runner_cls(
            self.model,
            self._create_runner_deps(),
            runner_config,
        )

    def _decoder_runner_buffers(self) -> DecoderBuffers:
        """Hand the family the buffers it writes; the engine sizes them."""
        names = (
            "gather_ids_cuda",
            "draft_tokens_cuda",
            "mrope_position_ids_cuda",
            "num_accepted_draft_tokens_cuda",
            "previous_batch_indices_cuda",
            "previous_kv_lens_offsets_cuda",
            "previous_pos_id_offsets_cuda",
            "previous_pos_indices_cuda",
            "draft_ctx_seq_slots_cuda",
            "draft_ctx_token_indices_cuda",
            "draft_first_draft_indices_cuda",
            "draft_first_draft_seq_slots_cuda",
            "draft_request_indices_buffer_cuda",
            "draft_seq_slots_buffer_cuda",
        )
        return DecoderBuffers(
            input_ids_cuda=self.input_ids_cuda,
            position_ids_cuda=self.position_ids_cuda,
            **{name: getattr(self, name, None)
               for name in names},
        )

    def _decoder_settings(self) -> Dict[str, Any]:
        """What the decoder half is configured with, for either composition."""
        return dict(
            spec_config=self.spec_config,
            is_draft_model=self.is_draft_model,
            original_max_draft_len=self.original_max_draft_len,
            dtype=self.dtype,
            kv_cache_manager_key=self.kv_cache_manager_key,
            cuda_graph_specialize_lora=(
                self.llm_args.lora_config.cuda_graph_specialize_lora),
            input_processor=getattr(self, "input_processor", None),
            lora_model_config=getattr(self, "lora_model_config", None),
            initial_runtime_draft_len=self._initial_runtime_draft_len,
            max_total_draft_tokens=self.max_total_draft_tokens,
            max_draft_len=self.max_draft_len,
            max_draft_loop_tokens=getattr(self, "max_draft_loop_tokens", 0),
            enable_attention_dp=self.enable_attention_dp,
            disable_overlap_scheduler=self._disable_overlap_scheduler,
            enable_in_graph_sampling=self.enable_in_graph_sampling,
            sparse_attention_config=self.sparse_attention_config,
            prefill_cuda_graph_backend=self.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._prefill_cuda_graph_num_tokens,
            decoder_cuda_graph_batch_sizes=self._cuda_graph_batch_sizes,
            dynamic_draft_len_mapping=self._dynamic_draft_len_mapping,
            steady_gen_positions_pinned=getattr(self,
                                                "_steady_gen_positions_pinned",
                                                None),
            torch_compile_enabled=self._torch_compile_enabled,
            torch_compile_piecewise_cuda_graph=(
                self._torch_compile_piecewise_cuda_graph),
            cache_indirection_attention=self.cache_indirection_attention)

    def _decoder_runner_config(self) -> DecoderRunnerConfig:
        return DecoderRunnerConfig(
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            attention_backend=self.attn_backend,
            attention_runtime_features=self.attn_runtime_features,
            enable_autotuner=self.llm_args.enable_autotuner,
            **self._decoder_settings(),
        )

    def _initialize_decoder_runner(
            self, runner_cls: Type[DecoderRunner]) -> DecoderRunner:
        # Do not retain the engine through the flag's closure: the runner is
        # reachable from the engine, and `cleanup` only runs from `__del__`.
        engine = weakref.proxy(self)
        return runner_cls(
            self.model,
            self._create_runner_deps(),
            self._decoder_runner_config(),
            self._decoder_runner_buffers(),
            warmup_flag=lambda: engine.is_warmup,
        )

    def _install_runner_collaborators(self) -> None:
        """Hand the family the graph runners the engine built for it."""
        self._runner.cuda_graph_runner = self.cuda_graph_runner
        self._runner.breakable_cuda_graph_runner = (
            self.breakable_cuda_graph_runner)

    def _initialize_encoder_decoder_runner(
            self,
            runner_cls: Type[EncoderDecoderRunner]) -> EncoderDecoderRunner:
        engine = weakref.proxy(self)
        runner_config = EncoderDecoderRunnerConfig(
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            **EncoderDecoderRunnerConfig.encoder_fields(
                model=self.model,
                mapping=self.mapping,
                graph_config=self.llm_args.encoder_cuda_graph_config,
                encoder_max_batch_size=self.encoder_batch_size,
                encoder_max_num_tokens=self.encoder_max_num_tokens,
                max_seq_len=self.max_seq_len,
                max_beam_width=self.max_beam_width,
                without_logits=self.without_logits,
                attention_backend=self.attn_backend,
                attention_runtime_features=self.attn_runtime_features,
                enable_autotuner=self.llm_args.enable_autotuner,
                is_encoder_decoder=True,
                draft_model=self.is_draft_model,
            ),
            **self._decoder_settings(),
        )
        return runner_cls(
            self.model,
            self._create_runner_deps(),
            runner_config,
            self._decoder_runner_buffers(),
            warmup_flag=lambda: engine.is_warmup,
        )

    def _initialize_no_kv_cache_runner(
            self, runner_cls: Type[NoKVCacheRunner]) -> NoKVCacheRunner:
        runner_config = NoKVCacheRunnerConfig(
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            enable_attention_dp=self.enable_attention_dp,
            prefill_cuda_graph_backend=self.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._prefill_cuda_graph_num_tokens,
            attention_backend=self.attn_backend,
            attention_runtime_features=self.attn_runtime_features,
            enable_autotuner=self.llm_args.enable_autotuner,
            mm_encoder_cache_enabled=self._mm_encoder_cache_enabled,
        )
        return runner_cls(self.model, self._create_runner_deps(), runner_config)

    def _create_runner_deps(self) -> RunnerDeps:
        return RunnerDeps(
            model=self.model,
            dist=self.dist,
            mapping=self.mapping,
            input_ids_cuda=self.input_ids_cuda,
            position_ids_cuda=self.position_ids_cuda,
            gather_ids_cuda=getattr(self, "gather_ids_cuda", None),
            draft_tokens_cuda=getattr(self, "draft_tokens_cuda", None),
            cache_indirection=(self.cache_indirection_attention
                               if self.attn_backend.Metadata
                               is TrtllmAttentionMetadata else None),
            lora=self._lora,
            spec=self._spec,
            moe_load_balancer=self.moe_load_balancer,
            model_forward=self._model_caller,
        )

    def register_forward_pass_callable(self, callable: Callable):
        self.forward_pass_callable = callable

    def register_sample_in_graph_callable(self, callable: Optional[Callable]):
        """Register the hook that samples at the tail of the forward graph."""
        self.sample_in_graph_callable = callable

    def register_sample_type_resolver(self,
                                      resolver: Optional[Callable],
                                      stage: Optional[Callable] = None):
        """Register how a batch maps to its sampling tier, for the graph key."""
        self.cuda_graph_runner.register_sample_type_resolver(resolver)
        self._runner.stage_in_graph_sampling = stage

    def get_kv_cache_dtype_byte_size(self) -> float:
        """
        Returns the size (in bytes) occupied by kv cache type.
        """
        layer_quant_mode = self.model.model_config.quant_config.layer_quant_mode
        if layer_quant_mode.has_fp4_kv_cache():
            return 1 / 2
        elif layer_quant_mode.has_fp8_kv_cache(
        ) or layer_quant_mode.has_int8_kv_cache():
            return 1
        else:
            return 2

    def set_lora_model_config(self,
                              lora_target_modules: list[str],
                              trtllm_modules_to_hf_modules: dict[str, str],
                              swap_gate_up_proj_lora_b_weight: bool = True):
        # Called by `_util.py` after the engine exists. Both LoRA handles stay
        # engine state: warmup, capture and the enc-dec fast path read them.
        self.lora_model_config = make_lora_model_config(
            self.model, lora_target_modules, trtllm_modules_to_hf_modules,
            swap_gate_up_proj_lora_b_weight)

    def _init_cuda_graph_lora_manager(self, lora_config: LoraConfig):
        """Initialize CUDA Graph LoRA manager with model configuration."""
        if (self.cuda_graph_runner is not None
                and self.cuda_graph_runner.enabled):
            # For spec decode, each generation request contributes
            # max_draft_len + 1 tokens per forward pass.
            max_tokens_per_seq = (self.original_max_draft_len +
                                  1) if self.is_spec_decode else 1
            self.cuda_graph_lora_manager = make_cuda_graph_lora_manager(
                self.model,
                lora_config,
                self.lora_model_config,
                self.batch_size,  # Use engine's max batch size
                max_tokens_per_seq,
                self.max_num_tokens)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.model, "set_guided_decoder"):
            success = self.model.set_guided_decoder(guided_decoder)
            if success:
                self.guided_decoder = guided_decoder
            return success
        return False

    @property
    def use_mrope(self):
        use_mrope = False
        try:
            use_mrope = self.model.model_config.pretrained_config.rope_scaling[
                'type'] == 'mrope'
        except Exception:
            pass
        logger.debug(f"Detected use_mrope: {use_mrope}")
        return use_mrope

    @functools.cached_property
    def _mm_encoder_cache_enabled(self) -> bool:
        """Whether the multimodal encoder cache is active for this model."""
        return mm_encoder_cache_enabled(self.model)

    @property
    def is_warmup(self):
        return getattr(self, "_is_warmup", False)

    @is_warmup.setter
    def is_warmup(self, value: bool):
        self._is_warmup = value

        # This setter is the one choke point every warmup transition passes
        # through, including PyExecutor's, so select the MoE all-to-all budget
        # here rather than in set_warmup_flag().
        _set_moe_a2a_warmup(value)

        self._runner.moe_load_balancer_iter_info = (not value, not value)

    @property
    def use_beam_search(self):
        return self.max_beam_width > 1

    @property
    def _is_packed_runner(self) -> bool:
        """Which of the two runner contracts `_runner` implements."""
        return isinstance(self._runner, EncoderRunner)

    @contextmanager
    def set_warmup_flag(self):
        prev_is_warmup = self.is_warmup
        self.is_warmup = True
        try:
            yield
        finally:
            self.is_warmup = prev_is_warmup

    @staticmethod
    def with_warmup_flag(method):

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            with self.set_warmup_flag():
                return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def warmup_with_kv_cache_cleanup(method):
        """
        Decorator for warmup methods that cleans up NaNs/Infs in KV Cache after warmup execution.

        Why this is needed:
        - Our attention kernel uses multiplication by zero to mask out invalid tokens within
          the same page. Since NaN/Inf * 0 = NaN, any NaNs/Infs in these invalid KV areas
          will persist after masking.
        - These NaNs/Infs propagate to outputs and subsequent KV Cache entries, corrupting
          future computations with higher probability.
        - During warmup, we execute with placeholder data rather than actual valid inputs,
          which can introduce NaNs/Infs into KV Cache pages and cause random, hard-to-debug
          accuracy issues.
        """

        @functools.wraps(method)
        def wrapper(self,
                    resource_manager: Optional[ResourceManager] = None,
                    *args,
                    **kwargs):
            result = method(self, resource_manager, *args, **kwargs)
            kv_cache_manager = (resource_manager.get_resource_manager(
                self.kv_cache_manager_key)
                                if resource_manager is not None else None)
            if kv_cache_manager is not None:
                has_invalid_values = kv_cache_manager.check_invalid_values_in_kv_cache(
                    fill_with_zero=True)
                if has_invalid_values:
                    logger.warning(
                        "NaNs/Infs have been introduced to KVCache during warmup, KVCache was filled with zeros to avoid potential issues"
                    )
            return result

        return wrapper

    @with_warmup_flag
    def _warmup_encoder_cuda_graphs_enc_dec(
        self,
        resource_manager: ResourceManager,
    ) -> None:
        """Reached by name, by ``getattr``, from the encoder launch thread.

        The name has to resolve here for the probe to find it, and the warmup
        flag has to be bound to this object; the work is the family's.
        """
        self._runner.warmup_encoder_graphs(resource_manager)

    def _get_encoder_cuda_graph_batch_sizes(
            self, max_batch_size: int) -> tuple[int, ...]:
        """Reached by name from PyExecutor, which still schedules the encoder."""
        return self._runner.encoder_graph_batch_sizes(max_batch_size)

    def forward_encoder(
        self,
        encoder_requests: List[LlmRequest],
        resource_manager: Optional[ResourceManager] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Reached by name from PyExecutor's encoder step."""
        assert resource_manager is not None, (
            "the encoder phase requires a resource manager")
        return self._runner.forward_encoder(encoder_requests, resource_manager)

    @with_warmup_flag
    @warmup_with_kv_cache_cleanup
    def warmup(self,
               resource_manager: Optional[ResourceManager] = None) -> None:
        """Hand warmup to the runner that owns this model's family."""
        if self._is_packed_runner:
            packed_runner = cast(PackedModelRunner, self._runner)
            packed_runner.warmup()
            packed_runner.capture_graphs()
            return
        assert resource_manager is not None, (
            "scheduled warmup requires a resource manager")
        runner = cast(ModelRunner, self._runner)
        runner.warmup(resource_manager)
        runner.capture_graphs(resource_manager)

    def _compute_dynamic_draft_len_mapping(self) -> Optional[Dict[int, int]]:
        """Compute graph_bs → draft_len mapping for dynamic draft length feature.

        Example: draft_len_schedule = {4:4, 8:2, 32:1}, cuda_graph_batch_sizes = [1,2,3,4,5,6,7,8,16,24,32,64]
        - Batch sizes 1-4:   use draft_len=4 (up to key 4)
        - Batch sizes 5-8:   use draft_len=2 (up to key 8)
        - Batch sizes 9-32:  use draft_len=1 (up to key 32)
        - Batch sizes 33+:   use draft_len=0 (implicit, speculation disabled)

        Returns: {1:4, 2:4, 3:4, 4:4, 5:2, 6:2, 7:2, 8:2, 16:1, 24:1, 32:1, 64:0}
        """
        # Dynamic draft length for CUDA graphs is only supported for one-model path
        if (not self.spec_config or not self.spec_config.draft_len_schedule or
                not self.spec_config.spec_dec_mode.support_dynamic_draft_len()):
            return None

        schedule = self.spec_config.draft_len_schedule
        schedule_keys = list(schedule.keys())

        mapping = {}
        key_idx = 0
        for graph_bs in self._cuda_graph_batch_sizes:
            while key_idx < len(
                    schedule_keys) and schedule_keys[key_idx] < graph_bs:
                key_idx += 1
            if key_idx < len(schedule_keys):
                draft_len = schedule[schedule_keys[key_idx]]
            else:
                draft_len = 0
            mapping[graph_bs] = draft_len
        return mapping

    ### Helper methods promoted from the original warmup method ###

    @property
    def is_multimodal(self) -> bool:
        """True iff this engine drives a multimodal model."""
        return is_multimodal(self.model, self.input_processor)

    def _validate_breakable_cuda_graph_compatibility(self) -> None:
        if self.llm_args.prefill_cuda_graph_backend != PrefillCudaGraphBackend.BREAKABLE:
            return

        if isinstance(self.model, DecoderModelForCausalLM):
            return
        decoder_model = getattr(self.model, "llm", None)
        if (self.llm_args.disable_mm_encoder
                and isinstance(decoder_model, DecoderModelForCausalLM)
                and getattr(self.model, "mm_encoder", None) is None):
            return
        if (isinstance(self.model, MultimodalModelMixin) or isinstance(
                self.input_processor, BaseMultimodalInputProcessor)):
            raise ValueError(
                "breakable prefill CUDA graph does not support multimodal models"
            )

    def forward_multimodal_encoder_items(
        self,
        requests: List[LlmRequest],
        scheduled_items: Dict[int, List[int]],
    ) -> None:
        """Forward selected MM encoder items and commit request-local outputs."""
        if not scheduled_items:
            return
        if self._mm_item_scheduler is None:
            raise TypeError(
                "Item-level MM scheduling requires MultimodalModelMixin")
        self._mm_item_scheduler.forward_items(requests, scheduled_items)

    def cleanup(self) -> None:
        """Release resources owned by this model engine.

        Tears down, in order:

        1. The optional ``ModelLoader`` (which in turn releases any
           GMS client; see :meth:`ModelLoader.cleanup`).
        2. CUDA Graph captures (via :meth:`_release_cuda_graphs`).
        3. The runner, MM item scheduler, and model module reference, which
           hold references to the model.
        4. Input processors.

        Idempotency:
            Subsequent calls are no-ops (guarded by ``_cleanup_done``).
            The flag is set only at the end, so a partial cleanup that
            raises mid-way will be retried on the next call.

        Called from:
            :meth:`__del__`, and only from there. ``PyExecutor.shutdown``
            deliberately does *not* call this: it is also invoked mid-init by
            ``configure_kv_cache_capacity``, which reads ``model`` right
            afterwards, so clearing ``model`` here would break it. That path
            calls :meth:`_release_cuda_graphs` and then drops its reference
            instead.
        """
        if self._cleanup_done:
            return

        # Cleanup is not truly atomic: released CUDA/GMS resources cannot be
        # rolled back.  Keep each handle live until its own release succeeds,
        # so a failed cleanup can be retried without double-freeing resources
        # that were already released.
        model_loader = self.model_loader
        if model_loader is not None:
            model_loader.cleanup()
            self.model_loader = None

        # Release runner-owned graphs before dropping the runner. Keep the
        # handle available if graph release fails and cleanup is retried.
        self._release_cuda_graphs()

        # The runner and scheduler keep their own references to the model, so
        # clearing the engine's attribute alone would leave the weights
        # reachable past `release_gc()` below.
        self._runner = None
        self._mm_item_scheduler = None
        self.model = None

        self.input_processor = None

        # Release model weights.
        release_gc()
        self._cleanup_done = True

    def __del__(self) -> None:
        """Best-effort cleanup during garbage collection.

        Delegates to :meth:`cleanup`. Catches ``RuntimeError`` (which a
        release step such as :meth:`_release_cuda_graphs` or
        ``ModelLoader.cleanup`` may raise) and ``AttributeError`` (typical
        on partially-initialized engines torn down during interpreter
        shutdown when module references have already been cleared); both
        are logged and swallowed because destructors cannot reliably
        surface exceptions.

        This is the only production caller of :meth:`cleanup` -- see the
        note there on why ``PyExecutor.shutdown`` must not call it.
        """
        try:
            self.cleanup()
        except (RuntimeError, AttributeError) as e:
            logger.warning(
                "PyTorchModelEngine cleanup failed during destruction: %s", e)

    def _init_max_seq_len(self):
        # Allow user to override the inferred max_seq_len with a warning.
        allow_long_max_model_len = os.getenv(
            "TLLM_ALLOW_LONG_MAX_MODEL_LEN",
            "0").lower() in ["1", "true", "yes", "y"]

        # Vision encoders may expose their own sequence-length inference.
        if hasattr(self.model, 'infer_max_seq_len'):
            inferred_max_seq_len = self.model.infer_max_seq_len()
        else:
            inferred_max_seq_len = self._infer_max_seq_len_from_config()

        if self.max_seq_len is None:
            logger.info(
                f"max_seq_len is not specified, using inferred value {inferred_max_seq_len}"
            )
            self.max_seq_len = inferred_max_seq_len
        elif inferred_max_seq_len < self.max_seq_len:
            if allow_long_max_model_len:
                logger.warning(
                    f"User specified max_seq_len is larger than the config in the model config file "
                    f"({inferred_max_seq_len}). Setting max_seq_len to user's specified value {self.max_seq_len}. "
                )
            else:
                # NOTE: py_executor_creator makes sure that the executor uses this
                # smaller value as its max_seq_len too.
                logger.warning(
                    f"Specified {self.max_seq_len=} is larger than what the model can support "
                    f"({inferred_max_seq_len}). Setting max_seq_len to {inferred_max_seq_len}. "
                )
                self.max_seq_len = inferred_max_seq_len

    def _infer_max_seq_len_from_config(self) -> int:

        if hasattr(self.model, 'model_config') and self.model.model_config:
            model_config = self.model.model_config.pretrained_config
            rope_scaling = getattr(model_config, 'rope_scaling', None)
            rope_factor = 1
            if rope_scaling is not None:
                rope_type = rope_scaling.get('type',
                                             rope_scaling.get('rope_type'))
                if rope_type not in ("su", "longrope", "llama3", "yarn"):
                    rope_factor = rope_scaling.get('factor', 1.0)

            # Step 1: Find the upper bound of max_seq_len
            inferred_max_seq_len = 2048
            max_position_embeddings = getattr(model_config,
                                              'max_position_embeddings', None)
            if max_position_embeddings is None and hasattr(
                    model_config, 'text_config'):
                max_position_embeddings = getattr(model_config.text_config,
                                                  'max_position_embeddings',
                                                  None)
            if max_position_embeddings is not None:
                inferred_max_seq_len = max_position_embeddings

            # Step 2: Scale max_seq_len with rotary scaling
            if rope_factor != 1:
                inferred_max_seq_len = int(
                    math.ceil(inferred_max_seq_len * rope_factor))
                logger.warning(
                    f'max_seq_len is scaled to {inferred_max_seq_len} by rope scaling {rope_factor}'
                )

            return inferred_max_seq_len

        default_max_seq_len = 8192
        logger.warning(
            f"Could not infer max_seq_len from model config, using default value: {default_max_seq_len}"
        )
        return default_max_seq_len

    def _init_max_num_tokens(self):
        # Modified from tensorrt_llm/_bootstrap.py check_max_num_tokens
        if self.max_num_tokens is None:
            self.max_num_tokens = self.max_seq_len * self.batch_size
        if self.max_num_tokens > self.max_seq_len * self.batch_size:
            logger.warning(
                f"max_num_tokens ({self.max_num_tokens}) shouldn't be greater than "
                f"max_seq_len * max_batch_size ({self.max_seq_len * self.batch_size}), "
                f"specifying to max_seq_len * max_batch_size ({self.max_seq_len * self.batch_size})."
            )
            self.max_num_tokens = self.max_seq_len * self.batch_size

    def _init_model_capacity(self):
        self._init_max_seq_len()
        self._init_max_num_tokens()

    def _release_cuda_graphs(self):
        self._runner.cleanup()
        if self._torch_compile_backend is not None:
            self._torch_compile_backend.clear_piecewise_cuda_graphs()
        if hasattr(self,
                   'cuda_graph_runner') and self.cuda_graph_runner is not None:
            self.cuda_graph_runner.clear()
        if (hasattr(self, 'breakable_cuda_graph_runner')
                and self.breakable_cuda_graph_runner is not None):
            self.breakable_cuda_graph_runner.clear()

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor needs this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

    def _is_encoder_decoder_model(self) -> bool:
        return bool(
            getattr(getattr(self.model, "model_config", None),
                    "is_encoder_decoder", False))

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self.model.extra_attrs)
    def forward(self,
                batch: Union[ScheduledRequests, PackedEncoderBatch],
                resource_manager: Optional[ResourceManager] = None,
                new_tensors_device: Optional[SampleStateTensors] = None,
                gather_context_logits: bool = False,
                cache_indirection_buffer: Optional[torch.Tensor] = None,
                num_accepted_tokens_device: Optional[torch.Tensor] = None,
                req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None):
        if isinstance(batch, PackedEncoderBatch):
            assert self._is_packed_runner, (
                "a packed batch requires a packed-batch runner")
            return cast(PackedModelRunner, self._runner).forward(
                batch, gather_context_logits=gather_context_logits)
        assert resource_manager is not None, (
            "scheduled execution requires a resource manager")
        return self._runner.forward(
            batch,
            resource_manager=resource_manager,
            new_tensors_device=new_tensors_device,
            gather_context_logits=gather_context_logits,
            cache_indirection_buffer=cache_indirection_buffer,
            num_accepted_tokens_device=num_accepted_tokens_device,
            req_id_to_old_request=req_id_to_old_request,
        )

    def _init_userbuffers(self, hidden_size):
        if self.mapping.tp_size <= 1 or self.mapping.pp_size > 1:
            return False

        # Disable UB for unsupported platforms
        if not ub.ub_supported():
            return False
        # NCCL_SYMMETRIC strategy no longer requires UserBuffer allocator initialization.
        # It uses NCCLWindowAllocator from ncclUtils directly.
        if self.llm_args.allreduce_strategy == "NCCL_SYMMETRIC":
            # Skip UB initialization for NCCL_SYMMETRIC - it uses NCCLWindowAllocator directly
            return False
        ub.initialize_userbuffers_manager(self.mapping.tp_size,
                                          self.mapping.pp_size,
                                          self.mapping.cp_size,
                                          self.mapping.rank,
                                          self.mapping.gpus_per_node,
                                          hidden_size * self.max_num_tokens * 2)

        return True

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        """
        When doing spec decode, sometimes draft models need to share certain weights
        with their target models. Here, we set up such weights by invoking
        self.model.load_weights_from_target_model if such a method exists.
        """
        loader = getattr(self.model, "load_weights_from_target_model", None)
        if callable(loader):
            loader(target_model)

        # logits_rows is a view into logits_tensor (narrow + view never
        # copy), so the processors already mutated it in place. Writing it
        # back would be a self-assignment, which torch rejects for the
        # non-contiguous slices a TP-padded vocab produces.

    def wait_for_input_copy(self):
        """
        Wait for input preparation and H2D copy of previous iteration before modifying host input,
        otherwise the input of previous iteration will be overwritten.
        """
        if self._prepare_inputs_event is not None:
            self._prepare_inputs_event.synchronize()
