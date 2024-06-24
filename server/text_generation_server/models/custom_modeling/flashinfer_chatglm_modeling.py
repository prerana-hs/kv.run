# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import math

import torch
import flashinfer
from torch import nn
import torch.distributed
import torch.nn.functional as F

from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.utils.lora_utils import BatchedModelLoraWeight
from text_generation_server.utils.cache_manager_flashinfer import KvCachePool, KvCacheBatchPosition
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear
)

from typing import Optional, List, Tuple
from transformers.tokenization_utils import AddedToken
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.activations import ACT2FN

from punica_kernels import (
    add_lora_sgmv_custom_cutlass as add_lora,
    rms_norm,
)


class FlashinferBatch:
    def __init__(self, seq_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len


logger = logging.get_logger(__name__)


class ChatGLM3Config(PretrainedConfig):
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)


class ChatGLM3RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        return rms_norm(hidden_states, self.weight, self.variance_epsilon), residual


class ChatGLM3MLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        self.up_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense_h_to_4h",
            weights=weights,
            bias=False,
        )

        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense_4h_to_h",
            weights=weights,
            bias=False,
        )
        # self.intermediate_size = (
        #     config.intermediate_size // weights.process_group.size()
        # )

    def forward(self, hidden_states, lora: BatchedModelLoraWeight | None):
        # [s, b, 3hp]
        up = self.up_proj(hidden_states)
        if lora:
            add_lora(
                up,
                hidden_states,
                lora.gate.wa_ptr,
                lora.gate.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
        up = self.activation_func(up)
        # [s, b, h]
        down = self.down_proj(up)
        if lora:
            add_lora(
                down,
                hidden_states,
                lora.gate.wa_ptr,
                lora.gate.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        return down


def load_attention(config, prefix, weights, num_key_value_heads):
    if config.num_attention_heads != num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_weights_col(
        prefix=prefix,
        quantize=config.quantize,
    )

    # manually concatenate qkv project bias
    bias = weights.get_tensor(f"{prefix}.bias")

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.multi_query_group_num // weights.process_group.size()
        
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(
        get_linear(weight, bias=bias, quantize=config.quantize)
    )


class FlashChatGLM3Attention(nn.Module):
    def __init__(self, prefix: str, config: ChatGLM3Config, weights, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_qo_heads = self.num_heads // weights.process_group.size()

        self.num_key_value_heads = config.multi_query_group_num
        self.num_key_value_groups = self.num_qo_heads // self.num_key_value_heads
        
        # self.num_kv_heads = (
        #     config.num_key_value_heads // weights.process_group.size()
        # )
        self.num_kv_heads = self.num_key_value_heads // weights.process_group.size()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.qkv_proj = load_attention(config, f'{prefix}.query_key_value', weights, self.num_key_value_heads)

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.dense",
            weights=weights,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None
    ):
        qkv = self.qkv_proj(hidden_states)

        q_proj, k_proj, v_proj = qkv.split(
            [
                self.head_dim * self.num_qo_heads,
                self.head_dim * self.num_kv_heads,
                self.head_dim * self.num_kv_heads
            ],
            dim=1,
        )

        q_proj = q_proj.contiguous()
        k_proj = k_proj.contiguous()
        v_proj = v_proj.contiguous()

        if lora:
            add_lora(
                q_proj,
                hidden_states,
                lora.q.wa_ptr,
                lora.q.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
            add_lora(
                k_proj,
                hidden_states,
                lora.k.wa_ptr,
                lora.k.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
            add_lora(
                v_proj,
                hidden_states,
                lora.v.wa_ptr,
                lora.v.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        stack_attn_output = []
        workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=kvCachePool.device)
        prefillTotalSeqLen = prefillBatchPosition.total_seq_len
        if prefillTotalSeqLen > 0:
            # need to revisit if contiguous conversion is the best way
            q = q_proj[: prefillTotalSeqLen].view(prefillTotalSeqLen, self.num_qo_heads, self.head_dim).contiguous()
            k = k_proj[: prefillTotalSeqLen].view(prefillTotalSeqLen, self.num_kv_heads, self.head_dim).contiguous()
            v = v_proj[: prefillTotalSeqLen].view(prefillTotalSeqLen, self.num_kv_heads, self.head_dim).contiguous()
            
            seq_indptr = prefillBatchPosition.seq_indptr.clone()
            kv_page_indices = prefillBatchPosition.kv_page_indices.clone()
            kv_page_indptr = prefillBatchPosition.kv_page_indptr.clone()
            kv_last_page_len = prefillBatchPosition.kv_last_page_len.clone()
            
            flashinfer.append_paged_kv_cache(
                k,
                v,
                seq_indptr,
                kvCachePool.cache_data[self.layer_idx],
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_len)

            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            
            prefill_wrapper.begin_forward(
                seq_indptr,
                kv_page_indptr,
                kv_page_indices,
                kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                16
            )

            # group size: self.num_qo_heads / self.num_qo_heads

            attn_output_prefill = prefill_wrapper.forward(
                q, 
                kvCachePool.cache_data[self.layer_idx], 
                causal=True, 
                pos_encoding_mode="ROPE_LLAMA" # this may need change
            ).view(prefillTotalSeqLen, self.hidden_size)
            prefill_wrapper.end_forward()
            stack_attn_output.append(attn_output_prefill)

        decodeTotalSeqLen = decodeBatchPosition.total_seq_len
        if decodeTotalSeqLen > 0:
            q = q_proj[prefillTotalSeqLen :].view(decodeTotalSeqLen, self.num_qo_heads, self.head_dim).contiguous()
            k = k_proj[prefillTotalSeqLen :].view(decodeTotalSeqLen, self.num_kv_heads, self.head_dim).contiguous()
            v = v_proj[prefillTotalSeqLen :].view(decodeTotalSeqLen, self.num_kv_heads, self.head_dim).contiguous()

            flashinfer.append_paged_kv_cache(
                k,
                v,
                decodeBatchPosition.seq_indptr,
                kvCachePool.cache_data[self.layer_idx],
                decodeBatchPosition.kv_page_indices,
                decodeBatchPosition.kv_page_indptr,
                decodeBatchPosition.kv_last_page_len
            )

            decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            decode_wrapper.begin_forward(
                decodeBatchPosition.kv_page_indptr,
                decodeBatchPosition.kv_page_indices,
                decodeBatchPosition.kv_last_page_len,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                kvCachePool.page_len,
                pos_encoding_mode="ROPE_LLAMA"
            )
            
            attn_output_decode = decode_wrapper.forward(
                q, 
                kvCachePool.cache_data[self.layer_idx], 
                pos_encoding_mode="ROPE_LLAMA"
            ).view(decodeTotalSeqLen, self.hidden_size)

            decode_wrapper.end_forward()
            stack_attn_output.append(attn_output_decode)

        if len(stack_attn_output) == 1:
            attn_outputs = stack_attn_output[0]
        else:
            attn_outputs = torch.cat(stack_attn_output, dim=0)

        # output projection
        o = self.o_proj(attn_outputs)
        return o


class FlashChatGLM3Layer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.layer_id = layer_id
        prefix = f"transformer.encoder.layers.{layer_id}"
        self.self_attn = FlashChatGLM3Attention(
            prefix=f"{prefix}.self_attention", config=config, weights=weights, layer_idx=layer_id
        )
        self.mlp = ChatGLM3MLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id)

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        attn_output = self.self_attn(
            normed_hidden_states,
            kvCachePool,
            prefillBatchPosition,
            decodeBatchPosition,
            lora
        )

        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output, lora)

        return mlp_output, attn_res


class FlashChatGLM3Model(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        # embed_norm = config.hidden_size**0.5
        # self.embed_tokens = TensorParallelEmbedding(
        #     prefix="model.embed_tokens", weights=weights
        # )
        self.embed_tokens = TensorParallelEmbedding(
            prefix="transformer.embedding.word_embeddings", weights=weights
        )
        # self.embed_tokens.weight *= embed_norm

        self.layers = nn.ModuleList(
            [
                FlashChatGLM3Layer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix="transformer.encoder.final_layernorm", weights=weights, eps=config.layernorm_epsilon
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_dim
        self.num_heads = self.layers[0].self_attn.num_qo_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_kv_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool, 
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                kvCachePool,
                prefillBatchPosition,
                decodeBatchPosition,
                lora
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashChatGLMForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashChatGLM3Model(config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="transformer.embedding.word_embeddings" if config.tie_word_embeddings else "transformer.output_layer",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        prefillBatchPosition: KvCacheBatchPosition,
        decodeBatchPosition: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids,
            kvCachePool,
            prefillBatchPosition,
            decodeBatchPosition,
            lora
        )
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits