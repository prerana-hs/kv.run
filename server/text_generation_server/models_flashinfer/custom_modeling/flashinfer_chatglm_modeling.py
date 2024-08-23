# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import torch
from torch import nn
import torch.distributed
import torch.nn.functional as F

from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.utils.lora_utils import BatchedModelLoraWeight
from text_generation_server.utils.cache_manager_flashinfer import (
    KvCachePool,
    KvCacheBatchPosition,
)
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)

from typing import Optional, List, Tuple
from transformers.tokenization_utils import AddedToken
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.activations import ACT2FN

from text_generation_server.layers.flashinfer_attention import (
    POS_ENCODING_MODE,
    FlashinferAttentionWrapper,
    AttentionRotaryParams,
)
from punica_kernels import (
    rms_norm,
)
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
)

class FlashinferBatch:
    def __init__(self, seq_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len


logger = logging.get_logger(__name__)


class ChatGLMConfig(PretrainedConfig):
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
        rope_ratio=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs,
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
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.rope_ratio = rope_ratio
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)


class ChatGLMRMSNorm(nn.Module):
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


class ChatGLMMLP(nn.Module):
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

    def forward(self, hidden_states, loraWeight: BatchedModelLoraWeight):
        # [s, b, 3hp]
        up = self.up_proj(hidden_states)
        if loraWeight:
            loraWeight.apply_lora_weight_gate(up, hidden_states, self.layer_idx)
        up = self.activation_func(up)
        # [s, b, h]
        down = self.down_proj(up)
        if loraWeight:
            loraWeight.apply_lora_weight_down(down, up, self.layer_idx)
        return down


def load_attention(config, prefix, weights, bias):
    return TensorParallelColumnLinear.load_qkv(
        config,
        prefix=f"{prefix}.query_key_value",
        weights=weights,
        bias=False,
    )
    

def load_attention2(config, prefix, weights, num_key_value_heads):
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
        
def load_attention3(config, prefix, weights):
    weight = weights.get_weights_col(
        prefix=prefix,
        quantize=None,
    )
    bias = weights.get_tensor(f"{prefix}.bias")
    return TensorParallelColumnLinear(
        get_linear(weight, bias=bias,  quantize=None)
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
        num_key_value_heads = (
            config.multi_query_group_num // weights.process_group.size()
        )

        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(
        get_linear(weight, bias=bias, quantize=config.quantize)
    )


class FlashChatGLMAttention(nn.Module):
    def __init__(
        self,
        prefix: str,
        flashinferWrapper: FlashinferAttentionWrapper,
        config: ChatGLMConfig,
        weights,
        layer_idx: int,
    ):
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
        self.num_kv_heads = self.num_key_value_heads // weights.process_group.size()
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.flashinferWrapper = flashinferWrapper
        self.rotaryParams = AttentionRotaryParams(
            pos_encoding_mode=POS_ENCODING_MODE.NONE
        )

        self.layer_idx = layer_idx
        # self.qkv_proj = load_attention(
        #     config, prefix, weights, config.add_bias_linear or config.add_qkv_bias
        # )
        
        # self.qkv_proj = load_attention2(
        #     config, f"{prefix}.query_key_value", weights, self.num_key_value_heads
        # )
        
        self.qkv_proj = load_attention3(
            config, f"{prefix}.query_key_value", weights
        )
        
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_dim // 2,
            base=10000 * config.rope_ratio,
            device=weights.device,
        )

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
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        loraWeight: BatchedModelLoraWeight | None,
    ):
        q_dim = (
            self.flashinferWrapper.num_attention_heads * self.flashinferWrapper.head_dim
        )
        kv_dim = (
            self.flashinferWrapper.num_key_value_heads * self.flashinferWrapper.head_dim
        )
        qkv = self.qkv_proj(hidden_states)
        q_proj, k_proj, v_proj = qkv.split(
            [q_dim, kv_dim, kv_dim],
            dim=1,
        )
        q = q_proj.contiguous()
        k = k_proj.contiguous()
        v = v_proj.contiguous()

        if loraWeight:
            loraWeight.apply_lora_weight_kvq(q, k, v, hidden_states, self.layer_idx)

        q_multi_head = q.view(
                -1,
                self.flashinferWrapper.num_attention_heads,
                self.flashinferWrapper.head_dim,
            )
        
        k_multi_head = k.view(
                -1,
                self.flashinferWrapper.num_key_value_heads,
                self.flashinferWrapper.head_dim,
            )
        
        v_multi_head = v.view(
                -1,
                self.flashinferWrapper.num_key_value_heads,
                self.flashinferWrapper.head_dim,
            )
        
        # torch.save(q_multi_head, "query_layer_before_flashinfer")
        # torch.save(k_multi_head, "key_layer_before_flashinfer")
        # torch.save(cos, "cos")
        # torch.save(sin, "sin")
        # self.rotary_emb(
        #     q_multi_head,
        #     k_multi_head,
        #     cos,
        #     sin,
        # )
        # torch.save(q_multi_head, "query_layer_after_flashinfer")
        # torch.save(k_multi_head, "key_layer_after_flashinfer")
        
        q_after_rope, k_after_rope = self.wrap_causal_rotary(q_multi_head, k_multi_head, cos, sin)
        attn_outputs_raw = self.flashinferWrapper.computeAttention2(
            q_after_rope.contiguous(),
            k_after_rope.contiguous(),
            v_multi_head,
            kvCachePool.cache_data[self.layer_idx],
            is_prefill,
            batch_position,
            self.rotaryParams,
        )
        attn_outputs = self.o_proj(attn_outputs_raw)
        if loraWeight:
            loraWeight.apply_lora_weight_attn(
            attn_outputs, attn_outputs_raw, self.layer_idx
        )
        return attn_outputs
    
    def wrap_causal_rotary(self, q_multi_head: torch.Tensor, k_multi_head: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rotary_pos_emb_flashinfer = torch.cat((cos.transpose(1, 2), sin.transpose(1, 2)), dim=2).unsqueeze(0)
        q_temp = self.apply_rotary_pos_emb(q_multi_head.transpose(0, 1).unsqueeze(0), rotary_pos_emb_flashinfer)
        k_temp = self.apply_rotary_pos_emb(k_multi_head.transpose(0, 1).unsqueeze(0), rotary_pos_emb_flashinfer)
        return q_temp.squeeze(0).transpose(0, 1), k_temp.squeeze(0).transpose(0, 1)

    
    def apply_rotary_pos_emb(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        # x: [b, np, sq, hn]
        b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # truncate to support variable sizes
        rope_cache = rope_cache[:, :sq]
        xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
        rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)


class FlashChatGLM3Layer(nn.Module):
    def __init__(
        self, flashinferWrapper: FlashinferAttentionWrapper, layer_id, config, weights
    ):
        super().__init__()
        self.layer_id = layer_id
        prefix = f"transformer.encoder.layers.{layer_id}"
        self.self_attn = FlashChatGLMAttention(
            prefix=f"{prefix}.self_attention",
            flashinferWrapper=flashinferWrapper,
            config=config,
            weights=weights,
            layer_idx=layer_id,
        )
        self.mlp = ChatGLMMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id
        )

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.layernorm_epsilon,
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.layernorm_epsilon,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        cos: torch.Tensor,
        sin: torch.Tensor,
        loraWeight: BatchedModelLoraWeight | None,
    ):
        normed_hidden_states, _ = self.input_layernorm(hidden_states)

        attn_output = self.self_attn(
            normed_hidden_states,
            kvCachePool,
            is_prefill,
            batch_position,
            cos,
            sin,
            loraWeight,
        )

        residual = hidden_states
        layernorm_input = residual + attn_output
        normed_attn_res_output, _ = self.post_attention_layernorm(
            layernorm_input
        )
        residual = layernorm_input
        mlp_output = self.mlp(normed_attn_res_output, loraWeight) + residual

        return mlp_output


class FlashChatGLM3Model(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="transformer.embedding.word_embeddings", weights=weights
        )

        assert config.num_attention_heads % weights.process_group.size() == 0
        assert config.multi_query_group_num % weights.process_group.size() == 0
        num_attention_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = (
            config.multi_query_group_num // weights.process_group.size()
        )

        self.flashinferWrapper = FlashinferAttentionWrapper(
            num_attention_heads, num_key_value_heads, config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                FlashChatGLM3Layer(
                    self.flashinferWrapper,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix="transformer.encoder.final_layernorm",
            weights=weights,
            eps=config.layernorm_epsilon,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_dim
        self.num_heads = self.layers[0].self_attn.num_qo_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_kv_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight | None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_ids, max_seq_len = (
            self._getPositionIdsAndMaxSeqLenForPrefill(
                batch_position.seq_lens, hidden_states.device
            )
            if is_prefill
            else self._getPositionIdsAndMaxSeqLenForDecode(
                batch_position.seq_lens, hidden_states.device
            )
        )

        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_seq_len, hidden_states.dtype
        )

        self.flashinferWrapper.prepareAttention(
            is_prefill,
            batch_position,
            kvCachePool.page_len,
            POS_ENCODING_MODE.NONE,
            kvCachePool.cache_data[0].dtype,
        )
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                kvCachePool,
                is_prefill,
                batch_position,
                cos,
                sin,
                loraWeight,
            )

        hidden_states, _ = self.norm(hidden_states)
        self.flashinferWrapper.endBatchAttention(is_prefill)
        return hidden_states
    
    def _getPositionIdsAndMaxSeqLenForPrefill(
        self, seq_lens: torch.Tensor, device
    ) -> Tuple[torch.Tensor, int]:
        if seq_lens.numel() == 0:
            return torch.tensor([], dtype=torch.int32, device=device), 0
        position_ids = torch.cat(
            [
                torch.arange(seq_len, dtype=torch.int32, device=device)
                for seq_len in seq_lens
            ]
        )
        max_seq_len = torch.max(seq_lens).item()
        return position_ids, max_seq_len

    def _getPositionIdsAndMaxSeqLenForDecode(
        self, seq_lens: torch.Tensor, device
    ) -> Tuple[torch.Tensor, int]:
        if seq_lens.numel() == 0:
            return torch.tensor([], dtype=torch.int32, device=device), 0
        position_ids = torch.cat(
            [
                torch.tensor([seq_len - 1], dtype=torch.int32, device=device)
                for seq_len in seq_lens
            ]
        )
        max_seq_len = torch.max(seq_lens).item()
        return position_ids, max_seq_len

class FlashChatGLMForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashChatGLM3Model(config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="transformer.output_layer",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight | None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids,
            kvCachePool,
            is_prefill,
            batch_position,
            loraWeight,
        )
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
