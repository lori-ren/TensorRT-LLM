# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder-phase runner for encoder-decoder models."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from typing_extensions import Self

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.memory_buffer_utils import with_shared_pool
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.bindings.internal import batch_manager as batch_manager_bindings
from tensorrt_llm.llmapi.llm_args import EncodeCudaGraphConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ....attention.backends.interface import AttentionMetadata
from ....metadata import KVCacheParams
from ....utils import set_per_request_prefill_cuda_graph_flag
from ...kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from ...llm_request import LlmRequestState
from ...resource_manager import KVCacheManager, ResourceManagerType
from ..cuda_graph import resolve_cuda_graph_batch_sizes
from .common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_padding_params,
    get_position_id_offset,
    get_top_level_model,
)
from .decoder import DecoderBuffers, DecoderMixin
from .decoder.config import DecoderConfigMixin, DecoderRunnerConfig
from .decoder.forward import ForwardExecutor
from .decoder.prepare import InputPreparer
from .decoder.warmup import WarmupDriver
from .encoder import EncoderConfigMixin, EncoderMixin, EncoderPreparedInputs
from .interface import RunnerConfig, RunnerDeps


@dataclass(frozen=True, kw_only=True)
class EncoderDecoderRunnerConfig(EncoderConfigMixin, DecoderConfigMixin, RunnerConfig):
    """Configuration for the Encoder-Decoder model runner.

    Both halves: ``EncoderConfigMixin`` for the encoder phase,
    ``DecoderConfigMixin`` because the decoder half is the decoder family's.
    """

    @classmethod
    def create(
        cls,
        *,
        decoder: "DecoderRunnerConfig",
        model: nn.Module,
        mapping: Mapping,
        graph_config: EncodeCudaGraphConfig | None,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        max_beam_width: int,
        without_logits: bool,
        attention_backend: type[AttentionBackend],
        attention_runtime_features: AttentionRuntimeFeatures,
        enable_autotuner: bool,
        draft_model: bool,
    ) -> Self:
        """Compose both halves.

        The encoder arguments size the encoder phase; ``decoder`` carries the
        decoder half's settings verbatim, so neither mixin has to know that the
        other exists.
        """
        return cls(
            **cls.encoder_fields(
                model=model,
                mapping=mapping,
                graph_config=graph_config,
                max_batch_size=max_batch_size,
                max_num_tokens=max_num_tokens,
                max_seq_len=max_seq_len,
                max_beam_width=max_beam_width,
                without_logits=without_logits,
                attention_backend=attention_backend,
                attention_runtime_features=attention_runtime_features,
                enable_autotuner=enable_autotuner,
                is_encoder_decoder=True,
                draft_model=draft_model,
            ),
            **{
                field.name: getattr(decoder, field.name)
                for field in dataclasses.fields(DecoderConfigMixin)
            },
        )


class EncoderDecoderInputPreparer(InputPreparer):
    """Input preparation for encoder-decoder: it adds cross attention."""

    def _can_use_input_fast_path(
        self,
        scheduled_requests: ScheduledRequests,
        new_tokens_device: Optional[torch.Tensor],
        next_draft_tokens_device: Optional[torch.Tensor],
    ) -> bool:
        """Return whether the TRT-like persistent input path is sufficient."""
        static_eligible = self._ctx.state.encoder_decoder_input_fast_path_static_eligible
        if static_eligible is None:
            static_eligible = (
                hasattr(batch_manager_bindings, "prepare_encoder_decoder_inputs")
                and self._ctx.is_encoder_decoder
                and not self._ctx.config.is_draft_model
                and self._ctx.config.max_beam_width == 1
                and self._ctx.config.sparse_attention_config is None
                and not self._ctx.use_mrope
                and not self._ctx.config.enable_attention_dp
                and not self._ctx.deps.mapping.has_cp_helix()
                and not self.is_multimodal
                and not self._ctx.config.attention_runtime_features.chunked_prefill
                and not self._ctx.config.attention_runtime_features.cache_reuse
                and not self._ctx.config.attention_runtime_features.has_speculative_draft_tokens
            )
            self._ctx.state.encoder_decoder_input_fast_path_static_eligible = static_eligible
        if (
            not static_eligible
            or self._ctx.state.enable_spec_decode
            or self._ctx.config.lora_model_config is not None
            or new_tokens_device is None
            or next_draft_tokens_device is not None
            or self._ctx.state.guided_decoder is not None
        ):
            return False

        if scheduled_requests.batch_size == 0:
            return False
        for request in scheduled_requests.generation_requests:
            if request.py_batch_idx is None and not request.is_dummy:
                return False
        return True

    def _acquire_encoder_decoder_host_buffers(self) -> Dict[str, Any]:
        """Acquire pinned staging whose preceding asynchronous copies finished."""
        pool = self._ctx.state.encoder_decoder_host_buffer_pool
        for buffers in pool:
            event = buffers["event"]
            if event is None or event.query():
                return buffers

        buffers = {
            "input_ids": torch.empty(
                self._ctx.config.decoder_max_num_tokens, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "position_ids": torch.empty(
                self._ctx.config.decoder_max_num_tokens, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "sequence_lengths": torch.empty(
                self._ctx.config.batch_size, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "prompt_lengths": torch.empty(
                self._ctx.config.batch_size, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "cached_token_lengths": torch.empty(
                self._ctx.config.batch_size, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "kv_lengths": torch.empty(
                self._ctx.config.batch_size, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "encoder_kv_lengths": torch.empty(
                self._ctx.config.batch_size, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "previous_batch_indices": torch.empty(
                self._ctx.config.batch_size, dtype=torch.int, pin_memory=prefer_pinned()
            ),
            "event": None,
        }
        pool.append(buffers)
        return buffers

    @nvtx_range("_prepare_encoder_decoder_inputs_fast")
    def _prepare_inputs_fast(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        attn_metadata: AttentionMetadata,
        new_tokens_device: torch.Tensor,
        resource_manager: Optional[ResourceManager],
    ):
        """Prepare a simple BART batch with native collation and reused buffers."""
        buffers = self._acquire_encoder_decoder_host_buffers()
        position_id_offset = self._ctx.state.encoder_decoder_position_id_offset
        if position_id_offset is None:
            position_id_offset = get_position_id_offset(self._ctx.deps.model)
            self._ctx.state.encoder_decoder_position_id_offset = position_id_offset
        (
            request_ids,
            encoder_seq_lens,
            encoder_cached_token_lengths,
            total_num_tokens,
            num_context_tokens,
            num_previous_batch_requests,
            cached_kv_tokens,
            context_kv_tokens,
            generation_kv_tokens,
            max_kv_len,
            context_encoder_kv_tokens,
            generation_encoder_kv_tokens,
            max_encoder_kv_len,
        ) = batch_manager_bindings.prepare_encoder_decoder_inputs(
            scheduled_requests.context_requests,
            scheduled_requests.generation_requests,
            buffers["input_ids"],
            buffers["position_ids"],
            buffers["sequence_lengths"],
            buffers["prompt_lengths"],
            buffers["cached_token_lengths"],
            buffers["kv_lengths"],
            buffers["encoder_kv_lengths"],
            buffers["previous_batch_indices"],
            position_id_offset,
        )

        num_sequences = scheduled_requests.batch_size
        num_context_requests = scheduled_requests.num_context_requests
        num_generation_requests = scheduled_requests.num_generation_requests
        generation_request_ids = request_ids[num_context_requests:]
        if num_context_tokens:
            self._ctx.buffers.input_ids_cuda[:num_context_tokens].copy_(
                buffers["input_ids"][:num_context_tokens], non_blocking=True
            )
        if num_previous_batch_requests:
            previous_slots = self._ctx.buffers.previous_batch_indices_cuda[
                :num_previous_batch_requests
            ]
            staged_request_ids = generation_request_ids[:num_previous_batch_requests]
            # Sequence slots are stable for a request's lifetime, so the
            # device indices remain valid while this ordered batch does.
            if self._ctx.state.encoder_decoder_staged_request_ids != staged_request_ids:
                previous_slots.copy_(
                    buffers["previous_batch_indices"][:num_previous_batch_requests],
                    non_blocking=True,
                )
                self._ctx.state.encoder_decoder_staged_request_ids = staged_request_ids
            generation_begin = num_context_tokens
            generation_end = generation_begin + num_previous_batch_requests
            torch.index_select(
                new_tokens_device[0, :, 0],
                dim=0,
                index=previous_slots,
                out=self._ctx.buffers.input_ids_cuda[generation_begin:generation_end],
            )
        else:
            self._ctx.state.encoder_decoder_staged_request_ids = None
        dummy_begin = num_context_tokens + num_previous_batch_requests
        if dummy_begin < total_num_tokens:
            self._ctx.buffers.input_ids_cuda[dummy_begin:total_num_tokens].fill_(0)

        self._ctx.buffers.position_ids_cuda[:total_num_tokens].copy_(
            buffers["position_ids"][:total_num_tokens], non_blocking=True
        )
        final_position_ids = self._ctx.buffers.position_ids_cuda[:total_num_tokens].unsqueeze(0)

        sequence_lengths = buffers["sequence_lengths"][:num_sequences]
        attn_metadata._seq_lens = sequence_lengths
        if attn_metadata.is_cuda_graph and attn_metadata._seq_lens_cuda is not None:
            attn_metadata._seq_lens_cuda.copy_(sequence_lengths, non_blocking=True)
        else:
            attn_metadata._seq_lens_cuda = sequence_lengths.cuda(non_blocking=True)

        attn_metadata._num_contexts = scheduled_requests.num_context_requests
        attn_metadata._num_ctx_tokens = num_context_tokens
        attn_metadata._num_generations = num_generation_requests
        attn_metadata._num_tokens = total_num_tokens
        attn_metadata.beam_width = 1
        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = buffers["prompt_lengths"][:num_sequences]
        attn_metadata.num_chunked_ctx_requests = 0
        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=buffers["cached_token_lengths"][:num_sequences],
            num_extra_kv_tokens=0,
        )
        attn_metadata.kv_cache_manager = kv_cache_manager
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        attn_metadata.prepare_encoder_decoder_from_precomputed_lengths(
            prompt_lens=buffers["prompt_lengths"][:num_sequences],
            kv_lens=buffers["kv_lengths"][:num_sequences],
            context_kv_tokens=context_kv_tokens,
            generation_kv_tokens=generation_kv_tokens,
            max_kv_len=max_kv_len,
        )

        encoder_hidden_states = []
        for request in scheduled_requests.context_requests:
            encoder_output = request.py_encoder_output
            if encoder_output is None:
                raise RuntimeError(
                    f"Decoder context request {request.py_request_id} has no encoder output."
                )
            encoder_hidden_states.append(encoder_output)
            request.py_batch_idx = request.py_seq_slot

        cross_attention_inputs = self._prepare_cross_attn_inputs(
            encoder_hidden_states,
            encoder_seq_lens,
            encoder_cached_token_lengths,
            attn_metadata,
            resource_manager,
            encoder_kv_lens=buffers["encoder_kv_lengths"][:num_sequences],
            context_encoder_kv_tokens=context_encoder_kv_tokens,
            generation_encoder_kv_tokens=generation_encoder_kv_tokens,
            max_encoder_kv_len=max_encoder_kv_len,
        )

        attn_all_rank_num_tokens = get_all_rank_num_tokens(
            attn_metadata,
            enable_attention_dp=self._ctx.config.enable_attention_dp,
            mapping=self._ctx.deps.mapping,
            dist=self._ctx.deps.dist,
        )
        (padded_num_tokens, can_run_piecewise_cuda_graph, attn_all_rank_num_tokens) = (
            get_padding_params(
                total_num_tokens,
                scheduled_requests.num_context_requests,
                attn_all_rank_num_tokens,
                dist=self._ctx.deps.dist,
                enable_attention_dp=self._ctx.config.enable_attention_dp,
                prefill_cuda_graph_backend=self._ctx.config.prefill_cuda_graph_backend,
                prefill_cuda_graph_num_tokens=self._ctx.config.prefill_cuda_graph_num_tokens,
            )
        )
        set_per_request_prefill_cuda_graph_flag(can_run_piecewise_cuda_graph)
        attn_metadata.padded_num_tokens = (
            padded_num_tokens if padded_num_tokens != total_num_tokens else None
        )

        virtual_num_tokens = total_num_tokens
        if attn_metadata.padded_num_tokens is not None:
            self._ctx.buffers.input_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            self._ctx.buffers.position_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens
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
        inputs.update(cross_attention_inputs)

        self._ctx.state.iter_states["num_ctx_requests"] = scheduled_requests.num_context_requests
        self._ctx.state.iter_states["num_ctx_tokens"] = num_context_tokens
        self._ctx.state.iter_states["num_generation_tokens"] = num_generation_requests
        self._ctx.state.iter_states["cached_kv_tokens"] = cached_kv_tokens
        if not self._ctx.is_warmup:
            self._ctx.state.previous_request_ids = generation_request_ids

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        buffers["event"] = event
        return inputs, None

    def _prepare_cross_attn_inputs(
        self,
        encoder_hidden_states: List[torch.Tensor],
        encoder_seq_lens: List[int],
        encoder_num_cached_tokens_per_seq: List[int],
        attn_metadata: AttentionMetadata,
        resource_manager: Optional[ResourceManager],
        encoder_kv_lens: Optional[torch.Tensor] = None,
        context_encoder_kv_tokens: int = 0,
        generation_encoder_kv_tokens: int = 0,
        max_encoder_kv_len: int = 0,
    ) -> Dict[str, Any]:
        if not encoder_seq_lens:
            return {}

        if len(encoder_seq_lens) != attn_metadata.num_seqs:
            raise RuntimeError(
                "Cross-attention encoder lengths must align with decoder "
                f"sequences: got {len(encoder_seq_lens)} encoder lengths for "
                f"{attn_metadata.num_seqs} decoder sequences."
            )

        if resource_manager is None:
            raise RuntimeError(
                "Encoder-decoder decoder forward requires a resource manager "
                "with a cross-KV cache manager."
            )
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER
        )
        if cross_kv_cache_manager is None:
            raise RuntimeError(
                "Encoder-decoder decoder forward requires "
                "ResourceManagerType.CROSS_KV_CACHE_MANAGER."
            )

        new_encoder_tokens = sum(encoder_seq_lens)
        if encoder_hidden_states:
            packed_encoder_hidden_states = (
                encoder_hidden_states[0]
                if len(encoder_hidden_states) == 1
                else torch.cat(encoder_hidden_states, dim=0)
            )
            if packed_encoder_hidden_states.shape[0] != new_encoder_tokens:
                raise RuntimeError(
                    "Packed encoder hidden states do not match cross-attention "
                    "metadata: got "
                    f"{packed_encoder_hidden_states.shape[0]} rows for "
                    f"{new_encoder_tokens} new encoder KV tokens."
                )
            skip_cross_kv_projection = False
        else:
            if new_encoder_tokens != 0:
                raise RuntimeError(
                    "Cross-attention metadata asks to project encoder K/V, "
                    "but no encoder hidden states were supplied."
                )
            packed_encoder_hidden_states = None
            skip_cross_kv_projection = True

        def prepare_cross_metadata(cross_attn_metadata: AttentionMetadata) -> None:
            if encoder_kv_lens is None:
                cross_attn_metadata.prepare()
                return
            assert isinstance(cross_attn_metadata, TrtllmAttentionMetadata)
            cross_attn_metadata.prepare_encoder_decoder_from_precomputed_lengths(
                prompt_lens=attn_metadata.prompt_lens,
                kv_lens=encoder_kv_lens,
                context_kv_tokens=context_encoder_kv_tokens,
                generation_kv_tokens=generation_encoder_kv_tokens,
                max_kv_len=max_encoder_kv_len,
            )

        if attn_metadata.is_cuda_graph and attn_metadata.has_cross_sub_metadata:
            # Fast path for stable CUDA-graph generation steps: the encoder
            # KV lengths (kv_lens_cuda) and the frozen prompt lengths
            # (prompt_lens_cuda) are identical across all generation steps
            # for a fixed batch. Skip the expensive torch.tensor() allocations
            # and H2D copies inside prepare() when nothing has changed.
            is_stable_gen_step = (
                new_encoder_tokens == 0  # pure generation, no new cross-KV
                and self._ctx.state.cross_attn_stable_cached_tokens
                == encoder_num_cached_tokens_per_seq
                and self._ctx.state.cross_attn_stable_request_ids
                == attn_metadata.request_ids  # same batch and row order
            )
            if is_stable_gen_step:
                cross_attn_metadata = attn_metadata.cross
                # Only refresh the decoder-side Python references that the
                # kernel reads; these are pointer-level updates with no alloc.
                cross_attn_metadata._seq_lens = attn_metadata.seq_lens
                cross_attn_metadata._seq_lens_cuda = attn_metadata.seq_lens_cuda
                cross_attn_metadata.prompt_lens = attn_metadata.prompt_lens
                cross_attn_metadata.request_ids = attn_metadata.request_ids
                cross_attn_metadata.num_contexts = attn_metadata.num_contexts
            else:
                cross_attn_metadata = attn_metadata.update_cross_metadata(
                    encoder_seq_lens=encoder_seq_lens,
                    cross_kv_cache_manager=cross_kv_cache_manager,
                    encoder_num_cached_tokens_per_seq=encoder_num_cached_tokens_per_seq,
                )
                prepare_cross_metadata(cross_attn_metadata)
                if new_encoder_tokens == 0:
                    # Record this stable state for future fast-path use.
                    self._ctx.state.cross_attn_stable_cached_tokens = list(
                        encoder_num_cached_tokens_per_seq
                    )
                    self._ctx.state.cross_attn_stable_request_ids = list(attn_metadata.request_ids)
                else:
                    # Batch changed (new encoder request); reset cache.
                    self._ctx.state.cross_attn_stable_cached_tokens = None
                    self._ctx.state.cross_attn_stable_request_ids = None
        else:
            cross_attn_metadata = attn_metadata.create_cross_metadata(
                cross_kv_cache_manager=cross_kv_cache_manager,
                encoder_seq_lens=encoder_seq_lens,
                encoder_num_cached_tokens_per_seq=encoder_num_cached_tokens_per_seq,
            )
            if attn_metadata.is_cuda_graph:
                attn_metadata.cross = cross_attn_metadata
                if new_encoder_tokens == 0:
                    self._ctx.state.cross_attn_stable_cached_tokens = list(
                        encoder_num_cached_tokens_per_seq
                    )
                    self._ctx.state.cross_attn_stable_request_ids = list(attn_metadata.request_ids)
                else:
                    self._ctx.state.cross_attn_stable_cached_tokens = None
                    self._ctx.state.cross_attn_stable_request_ids = None
            else:
                self._ctx.state.cross_attn_stable_cached_tokens = None
                self._ctx.state.cross_attn_stable_request_ids = None
            prepare_cross_metadata(cross_attn_metadata)

        return {
            "encoder_hidden_states": packed_encoder_hidden_states,
            "cross_attn_metadata": cross_attn_metadata,
            "skip_cross_kv_projection": skip_cross_kv_projection,
        }


class EncoderDecoderWarmupDriver(WarmupDriver):
    """Warmup for encoder-decoder: cross-KV seeding and mixed-shape capture."""

    def _max_encoder_output_len(self, resource_manager: ResourceManager) -> int:
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER
        )
        max_encoder_output_len = int(self._ctx.config.max_seq_len)
        if cross_kv_cache_manager is not None:
            max_encoder_output_len = min(
                max_encoder_output_len,
                int(getattr(cross_kv_cache_manager, "max_seq_len", max_encoder_output_len)),
            )
        return max(1, max_encoder_output_len)

    def _enc_dec_hidden_size(self) -> int:
        config = self._ctx.deps.model.model_config.pretrained_config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            raise RuntimeError(
                "Encoder-decoder CUDA graph warmup could not infer encoder "
                "hidden size from the model config."
            )
        return int(hidden_size)

    def _add_cross_dummy_requests(
        self, requests: List[LlmRequest], resource_manager: ResourceManager
    ) -> bool:
        if not requests:
            return True
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER
        )
        if cross_kv_cache_manager is None:
            raise RuntimeError(
                "Encoder-decoder CUDA graph warmup requires "
                "ResourceManagerType.CROSS_KV_CACHE_MANAGER."
            )

        max_encoder_output_len = self._max_encoder_output_len(resource_manager)
        for request in requests:
            request.py_encoder_output = None
            request.py_skip_cross_kv_projection = True

        encoder_output_lens = [max_encoder_output_len] * len(requests)
        cross_dummy_requests = cross_kv_cache_manager.add_dummy_requests(
            request_ids=[request.py_request_id for request in requests],
            token_nums=encoder_output_lens,
            is_gen=True,
            max_beam_width=1,
            encoder_output_lens=encoder_output_lens,
        )
        if cross_dummy_requests is not None:
            return True

        kv_cache_manager = resource_manager.get_resource_manager(
            self._ctx.config.kv_cache_manager_key
        )
        draft_kv_cache_manager = self._ctx.get_draft_kv_cache_manager(resource_manager)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        for request in requests:
            kv_cache_manager.free_resources(request)
            if draft_kv_cache_manager is not None:
                draft_kv_cache_manager.free_resources(request)
            if spec_resource_manager is not None:
                spec_resource_manager.free_resources(request)
        return False

    def _populate_cross_kv_cache(self, inputs: Dict[str, Any]) -> None:
        encoder_hidden_states = inputs.get("encoder_hidden_states")
        cross_attn_metadata = inputs.get("cross_attn_metadata")
        if encoder_hidden_states is None or cross_attn_metadata is None:
            return

        decoder = getattr(get_top_level_model(self._ctx.deps.model), "decoder", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Encoder-decoder CUDA graph warmup requires a decoder with cross-attention layers."
            )

        attn_metadata = inputs["attn_metadata"]
        hidden_states = torch.ones(
            (attn_metadata.num_tokens, self._enc_dec_hidden_size()),
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        for layer in layers:
            cross_attn = getattr(layer, "cross_attn", None)
            if cross_attn is None:
                raise RuntimeError(
                    "Encoder-decoder CUDA graph warmup requires every decoder "
                    "layer to expose a cross_attn module."
                )
            cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attn_metadata=attn_metadata,
                cross_attn_metadata=cross_attn_metadata,
                skip_cross_kv_projection=False,
            )

    def _capture_mixed_cuda_graphs(
        self, resource_manager: ResourceManager, *, warmup, executor
    ) -> None:
        """Warm and capture reachable mixed encoder-decoder graph shapes.

        The first global CUDA-graph pass warms every shape so shared attention
        workspace reaches its final size. The second pass captures the same
        shapes. Runtime capture is deliberately disabled because graph capture
        executes KV-cache writes and must never run against live requests.
        """
        runner = self._ctx.state.cuda_graph_runner
        if not runner.enable_encoder_decoder_mixed_cuda_graph:
            return
        max_encoder_output_len = self._max_encoder_output_len(resource_manager)
        context_shapes = set(self._ctx.config.encoder_graph_shapes)
        if not context_shapes:
            logger.warning(
                "Skipping mixed encoder-decoder CUDA graph capture: "
                "no encoder CUDA graph shapes are configured."
            )
            return

        max_encoder_batch_size = max(batch_size for batch_size, _ in context_shapes)
        max_batch_token_counts = {
            total_tokens
            for batch_size, total_tokens in context_shapes
            if batch_size == max_encoder_batch_size
        }
        paired_context_count = 2 * max_encoder_batch_size
        if runner.max_supported_batch_size > paired_context_count:
            paired_token_counts = {
                first + second
                for first in max_batch_token_counts
                for second in max_batch_token_counts
            }
            context_shapes.update(
                (paired_context_count, token_count) for token_count in paired_token_counts
            )

        operation = "warmup" if runner.is_warmup_only else "capture"
        hidden_size = self._enc_dec_hidden_size()
        max_num_encoder_tokens = max(
            (
                total_encoder_tokens
                for num_contexts, total_encoder_tokens in context_shapes
                if total_encoder_tokens <= num_contexts * max_encoder_output_len
                and any(batch_size > num_contexts for batch_size in runner.supported_batch_sizes)
            ),
            default=0,
        )
        if max_num_encoder_tokens == 0:
            return
        model_config = self._ctx.deps.model.model_config.pretrained_config
        # The capture query length must equal the runtime decoder prefix or
        # every mixed batch misses its graph, silently and with no counter to
        # show it. Prefer the input processor's actual prefix (Whisper forces
        # [decoder_start, lang, task, no_timestamps] = 4); fall back to the
        # token-model heuristic: BART/mBART prepend a forced BOS token after
        # decoder_start, T5 uses decoder_start alone.
        prefix_fn = getattr(self._ctx.config.input_processor, "get_decoder_prefix_len", None)
        mixed_context_query_len = prefix_fn() if prefix_fn is not None else None
        if not mixed_context_query_len:
            mixed_context_query_len = (
                2 if getattr(model_config, "model_type", None) in ("bart", "mbart") else 1
            )
        logger.info(
            "Mixed encoder/decoder graph capture using decoder prefix "
            f"length {mixed_context_query_len}."
        )
        for num_contexts, total_encoder_tokens in sorted(
            context_shapes, key=lambda shape: shape[1], reverse=True
        ):
            if total_encoder_tokens > num_contexts * max_encoder_output_len:
                continue
            base_encoder_len, remainder = divmod(total_encoder_tokens, num_contexts)
            encoder_output_lens = [base_encoder_len + 1] * remainder + [base_encoder_len] * (
                num_contexts - remainder
            )
            if not encoder_output_lens or encoder_output_lens[-1] <= 0:
                continue

            for batch_size in runner.supported_batch_sizes:
                if batch_size <= num_contexts:
                    continue
                warmup_request = self._create_cuda_graph_warmup_request(
                    resource_manager,
                    batch_size,
                    draft_len=0,
                    mixed_context_encoder_output_lens=encoder_output_lens,
                    mixed_context_query_len=mixed_context_query_len,
                )
                with self._release_batch_context(warmup_request, resource_manager) as batch:
                    if batch is None:
                        logger.warning(
                            "Skipping mixed encoder-decoder CUDA graph "
                            f"{operation}: not enough KV cache space for "
                            f"batch size={batch_size}."
                        )
                        continue

                    context_requests = batch.context_requests
                    for request, encoder_output_len in zip(context_requests, encoder_output_lens):
                        request.state = LlmRequestState.CONTEXT_INIT
                        request.context_current_position = 0
                        request.context_chunk_size = mixed_context_query_len
                        request.cached_tokens = 0
                        request.py_batch_idx = None
                        request.py_encoder_output = torch.ones(
                            (encoder_output_len, hidden_size),
                            device="cuda",
                            dtype=self._ctx.config.dtype,
                        )
                        request.py_skip_cross_kv_projection = False

                    runner._get_static_encoder_hidden_states(
                        context_requests[0].py_encoder_output,
                        max_num_encoder_tokens,
                        allow_allocate=True,
                    )
                    logger.info(
                        "Run mixed encoder-decoder CUDA graph "
                        f"{operation} for batch size={batch_size}, "
                        f"context requests={num_contexts}, "
                        f"packed encoder tokens={total_encoder_tokens}"
                    )
                    with self._spec_decode_override(enable=False, draft_len=0):
                        self._executor._run_batch(batch, resource_manager)
                        torch.cuda.synchronize()


class EncoderDecoderRunner(DecoderMixin, EncoderMixin):
    """Run both halves of an encoder-decoder model.

    One mixin per half, composed rather than inherited from either runner:
    ``DecoderMixin`` brings the decoder half's phases and lifecycle,
    ``EncoderMixin`` the encoder phase. Where the decoder half differs for this
    model, the specialized phases below replace the default ones.
    """

    def __init__(
        self,
        model: nn.Module,
        deps: RunnerDeps,
        config: EncoderDecoderRunnerConfig,
        buffers: DecoderBuffers,
        *,
        warmup_flag: Callable[[], bool],
    ) -> None:
        if not config.is_encoder_decoder:
            raise ValueError("EncoderDecoderRunner requires an encoder-decoder model.")
        self._initialize_decoder(model, deps, config, buffers, warmup_flag=warmup_flag)
        self._initialize_encoder(model, deps, config)
        self._feature_staging: torch.Tensor | None = None
        self._feature_staging_event: torch.cuda.Event | None = None
        self._feature_copy_stream: torch.cuda.Stream | None = None

    def _make_preparer(self) -> InputPreparer:
        return EncoderDecoderInputPreparer(self._ctx)

    def _make_warmup(self, preparer: InputPreparer, executor: ForwardExecutor) -> WarmupDriver:
        return EncoderDecoderWarmupDriver(self._ctx, preparer, executor)

    def cleanup(self) -> None:
        """Release both halves; each side only knows its own."""
        EncoderMixin.cleanup(self)
        DecoderMixin.cleanup(self)

    def _build_attention_metadata(
        self,
        sequence_lengths: list[int],
        request_ids: list[int],
    ) -> VanillaAttentionMetadata | TrtllmAttentionMetadata:
        if len(sequence_lengths) != len(request_ids):
            raise ValueError("Encoder sequence lengths and request IDs must have the same length.")
        metadata = self._create_attention_metadata(
            enable_context_mla_with_cached_kv=False,
            num_heads_per_kv=1,
        )
        if not isinstance(metadata, (VanillaAttentionMetadata, TrtllmAttentionMetadata)):
            raise TypeError(
                "Only vanilla and TRT-LLM attention metadata are supported "
                "for encoder-decoder encoder execution."
            )
        metadata.seq_lens = torch.tensor(
            sequence_lengths, dtype=torch.int, pin_memory=prefer_pinned()
        )
        metadata.num_contexts = len(sequence_lengths)
        metadata.max_seq_len = self._config.max_seq_len
        metadata.request_ids = request_ids
        metadata.prepare_encoder_only()
        return metadata

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
    ) -> EncoderPreparedInputs:
        """Pack one scheduled encoder batch into the model's input contract."""
        del cuda_graph_lora_manager, runtime_draft_len
        encoder_requests = scheduled_requests.encoder_requests
        if not encoder_requests:
            raise ValueError("Encoder execution requires at least one request.")

        has_features = [
            request.py_encoder_input_features is not None for request in encoder_requests
        ]
        if any(has_features):
            if not all(has_features):
                raise ValueError(
                    "Feature- and token-driven encoder requests cannot share one batch."
                )
            return self._prepare_feature_inputs(
                encoder_requests,
                resource_manager=resource_manager,
            )
        return self._prepare_token_inputs(
            encoder_requests,
            resource_manager=resource_manager,
        )

    def _prepare_token_inputs(
        self,
        encoder_requests: list[LlmRequest],
        *,
        resource_manager: ResourceManager,
    ) -> EncoderPreparedInputs:
        input_ids: list[int] = []
        position_ids: list[int] = []
        sequence_lengths: list[int] = []
        request_ids: list[int] = []
        for request in encoder_requests:
            tokens = request.encoder_tokens
            if tokens is None:
                raise ValueError(f"Encoder request {request.py_request_id} has no encoder tokens.")
            sequence_length = len(tokens)
            input_ids.extend(tokens)
            position_ids.extend(
                apply_position_id_offset(
                    list(range(sequence_length)),
                    model=self._model,
                )
            )
            sequence_lengths.append(sequence_length)
            request_ids.append(request.py_request_id)

        num_tokens = len(input_ids)
        if num_tokens != len(position_ids):
            raise ValueError("Encoder input IDs and position IDs must have the same length.")
        if num_tokens > self._config.max_num_tokens:
            raise ValueError(
                f"Encoder packed length ({num_tokens}) exceeds max_num_tokens "
                f"({self._config.max_num_tokens})."
            )

        return self._prepare_packed_token_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_lengths=sequence_lengths,
            request_ids=request_ids,
            resource_manager=resource_manager,
        )

    def _prepare_feature_inputs(
        self,
        encoder_requests: list[LlmRequest],
        *,
        resource_manager: ResourceManager,
    ) -> EncoderPreparedInputs:
        features: list[torch.Tensor] = []
        sequence_lengths: list[int] = []
        request_ids: list[int] = []
        for request in encoder_requests:
            request_features = request.py_encoder_input_features
            if request_features is None:
                raise ValueError(f"Encoder request {request.py_request_id} has no input features.")
            features.append(request_features)
            sequence_lengths.append(int(request.encoder_output_len))
            request_ids.append(request.py_request_id)

        num_tokens = sum(sequence_lengths)
        if num_tokens > self._config.max_num_tokens:
            raise ValueError(
                f"Encoder packed length ({num_tokens}) exceeds max_num_tokens "
                f"({self._config.max_num_tokens})."
            )
        graph_inputs = self._prepare_encoder_feature_graph_inputs(
            features,
            sequence_lengths,
            request_ids,
            self._build_attention_metadata,
        )
        if graph_inputs is not None:
            return graph_inputs
        metadata = self._build_attention_metadata(sequence_lengths, request_ids)
        return EncoderPreparedInputs(
            {
                "input_features": self._pack_features(features),
                "encoder_attn_metadata": metadata,
                "encoder_seq_lens": sequence_lengths,
                "resource_manager": resource_manager,
            },
            sequence_lengths=sequence_lengths,
        )

    def _pack_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        first = features[0]
        uniform = first.device.type == "cpu" and all(
            feature.shape[1:] == first.shape[1:]
            and feature.dtype == first.dtype
            and feature.device.type == "cpu"
            for feature in features
        )
        if not uniform:
            return torch.cat(features, dim=0).to("cuda", non_blocking=True)

        rows = sum(feature.shape[0] for feature in features)
        staging = self._feature_staging
        if (
            staging is None
            or staging.dtype != first.dtype
            or staging.shape[1:] != first.shape[1:]
            or staging.shape[0] < rows
        ):
            if staging is not None:
                assert self._feature_staging_event is not None
                self._feature_staging_event.synchronize()
            staging = torch.empty(
                (rows, *first.shape[1:]),
                dtype=first.dtype,
                pin_memory=prefer_pinned(),
            )
            self._feature_staging = staging
            self._feature_staging_event = torch.cuda.Event()
            if self._feature_copy_stream is None:
                self._feature_copy_stream = torch.cuda.Stream()
        else:
            assert self._feature_staging_event is not None
            self._feature_staging_event.synchronize()

        offset = 0
        for feature in features:
            staging[offset : offset + feature.shape[0]].copy_(feature)
            offset += feature.shape[0]

        assert self._feature_copy_stream is not None
        assert self._feature_staging_event is not None
        consumer_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._feature_copy_stream):
            packed = staging[:rows].to("cuda", non_blocking=True)
            self._feature_staging_event.record()
        consumer_stream.wait_event(self._feature_staging_event)
        packed.record_stream(consumer_stream)
        return packed

    def warmup_encoder_graphs(self, resource_manager: ResourceManager) -> None:
        """Warm and capture the encoder half.

        Driven from the encoder launch thread through the coordinator, which
        keeps the name the probe looks up and the warmup flag bound to itself.
        """
        self.warmup_encoder(resource_manager)
        self.capture_encoder_graphs(resource_manager)

    def encoder_graph_batch_sizes(self, max_batch_size: int) -> tuple[int, ...]:
        """Startup graph sizing, while encoder scheduling remains in PyExecutor."""
        return resolve_cuda_graph_batch_sizes(
            self._encoder_graph_batch_sizes,
            max_batch_size,
            pad_to_limit=self._encoder_graph_pad_to_limit,
        )

    def forward_encoder(
        self,
        encoder_requests: list[LlmRequest],
        resource_manager: ResourceManager,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run the encoder phase for a batch of encoder requests."""
        scheduled_requests = ScheduledRequests()
        scheduled_requests.encoder_requests = list(encoder_requests)
        outputs = self._forward_encoder_phase(
            scheduled_requests,
            resource_manager=resource_manager,
            cuda_graph_lora_manager=None,
            runtime_draft_len=0,
            gather_context_logits=False,
        )
        return (outputs["encoder_hidden_states"], outputs["encoder_seq_lens"])

    def warmup_encoder(self, resource_manager: ResourceManager) -> None:
        """The encoder half warms up while capturing its graph shapes.

        Separate from :meth:`warmup`, which is the decoder half: the two are
        driven from different places. The coordinator drives the decoder half;
        the encoder half is reached by name, from the encoder launch thread.
        """

    def capture_encoder_graphs(self, resource_manager: ResourceManager) -> None:
        self._capture_encoder_cuda_graphs(
            lambda sequence_lengths: self._prepare_capture_inputs(
                sequence_lengths, resource_manager
            ),
            self._execute_prepared,
            build_metadata=self._build_attention_metadata,
        )

    def _prepare_capture_inputs(
        self, sequence_lengths: list[int], resource_manager: ResourceManager
    ) -> EncoderPreparedInputs:
        request_ids = list(range(len(sequence_lengths)))
        position_ids: list[int] = []
        for sequence_length in sequence_lengths:
            position_ids.extend(
                apply_position_id_offset(list(range(sequence_length)), model=self._model)
            )
        return self._prepare_packed_token_inputs(
            input_ids=[0] * sum(sequence_lengths),
            position_ids=position_ids,
            sequence_lengths=sequence_lengths,
            request_ids=request_ids,
            resource_manager=resource_manager,
        )

    def _prepare_packed_token_inputs(
        self,
        *,
        input_ids: list[int],
        position_ids: list[int],
        sequence_lengths: list[int],
        request_ids: list[int],
        resource_manager: ResourceManager,
    ) -> EncoderPreparedInputs:
        input_ids_cpu = torch.tensor(
            input_ids,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        position_ids_cpu = torch.tensor(
            position_ids,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        metadata = self._build_attention_metadata(sequence_lengths, request_ids)
        graph_inputs = {
            "input_ids": input_ids_cpu,
            "position_ids": position_ids_cpu,
            "seq_lens": sequence_lengths,
            "resource_manager": resource_manager,
        }
        prepared = self._prepare_encoder_graph_inputs(graph_inputs, metadata)
        if prepared is not None:
            return prepared
        return EncoderPreparedInputs(
            {
                "encoder_input_ids": input_ids_cpu.to("cuda", non_blocking=True),
                "encoder_position_ids": position_ids_cpu.to("cuda", non_blocking=True).unsqueeze(0),
                "encoder_attn_metadata": metadata,
                "encoder_seq_lens": sequence_lengths,
                "resource_manager": resource_manager,
            },
            sequence_lengths=sequence_lengths,
        )

    @torch.inference_mode()
    @nvtx_range("encoder_decoder_forward")
    def _forward_encoder_phase(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        gather_context_logits: bool,
    ) -> dict[str, Any]:
        del gather_context_logits
        prepared = self.prepare_inputs(
            scheduled_requests,
            resource_manager=resource_manager,
            cuda_graph_lora_manager=cuda_graph_lora_manager,
            runtime_draft_len=runtime_draft_len,
        )
        hidden_states = self._execute_prepared(prepared)
        return {
            "encoder_hidden_states": hidden_states,
            "encoder_seq_lens": prepared.sequence_lengths,
        }

    def _forward_encoder_stack(self, inputs: dict[str, Any]) -> torch.Tensor:
        encoder = getattr(self._model, "encoder", None)
        if encoder is None:
            inner = getattr(self._model, "model", None)
            encoder = getattr(inner, "encoder", None) if inner is not None else None
        if encoder is None:
            raise AttributeError("Encoder-decoder models must expose `encoder` or `model.encoder`.")

        input_features = inputs.get("input_features")
        if input_features is not None:
            return encoder(
                input_features=input_features,
                attn_metadata=inputs["encoder_attn_metadata"],
            )

        top_level_model = get_top_level_model(self._model)
        embedding = getattr(top_level_model, "shared_embedding", None) or getattr(
            top_level_model, "embed_tokens", None
        )
        input_ids = inputs["encoder_input_ids"]
        if embedding is None:
            hidden_states = input_ids
        else:
            hidden_states = embedding(input_ids)
            embedding_scale = getattr(top_level_model, "embed_scale", None)
            if embedding_scale is not None:
                hidden_states = hidden_states * embedding_scale

        position_ids = inputs.get("encoder_position_ids")
        if position_ids is not None and position_ids.dim() == 2:
            position_ids = position_ids.squeeze(0)
        return encoder(
            hidden_states=hidden_states,
            attn_metadata=inputs["encoder_attn_metadata"],
            position_ids=position_ids,
        )

    def _forward_graph_inputs(self, inputs: dict[str, Any]) -> torch.Tensor:
        return self._forward_encoder_stack(
            {
                "encoder_input_ids": inputs["input_ids"],
                "encoder_position_ids": inputs.get("position_ids"),
                "encoder_attn_metadata": inputs["attn_metadata"],
                "resource_manager": inputs.get("resource_manager"),
            }
        )

    def _execute_prepared(self, prepared: EncoderPreparedInputs) -> torch.Tensor:
        key = prepared.graph_key
        if key is None:
            return self._forward_encoder_stack(prepared.kwargs)

        graph_runner = self._encoder_cuda_graph_runner
        # Feature graphs must not install the token path's shared allocator pool.
        pool_context = (
            nullcontext()
            if graph_runner.feature_mode
            else with_shared_pool(graph_runner.get_graph_pool())
        )
        with pool_context:
            graph_outputs = self._execute_encoder_cuda_graph(
                prepared,
                self._forward_feature_graph_inputs
                if graph_runner.feature_mode
                else self._forward_graph_inputs,
            )
        if not isinstance(graph_outputs, torch.Tensor):
            raise TypeError("Encoder-decoder CUDA Graph replay must return hidden states.")
        if graph_runner.feature_mode:
            return graph_outputs[: sum(prepared.sequence_lengths)].clone()
        return graph_runner.restore_encoder_decoder_output(key, graph_outputs, prepared.kwargs)

    def _forward_feature_graph_inputs(self, inputs: dict[str, Any]) -> torch.Tensor:
        return self._forward_encoder_stack(
            {
                "input_features": inputs["input_features"],
                "encoder_attn_metadata": inputs["attn_metadata"],
                "encoder_seq_lens": inputs["seq_lens"],
            }
        )
