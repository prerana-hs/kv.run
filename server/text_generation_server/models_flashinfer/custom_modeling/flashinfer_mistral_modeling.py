# the Flash infer version of mistral, adapted from HuggingFace

import torch
import torch.distributed

import math

import flashinfer

from torch import nn

from transformers.models.mistral.modeling_mistral import (
    ACT2FN,
    MistralConfig,
    PreTrainedModel,
)

from punica_kernels import (
    add_lora_sgmv_custom_cutlass as add_lora,
    rms_norm,
)

from text_generation_server.utils.lora_utils import (
    BatchedLoraWeight,
    LoraWeight,
    BatchedModelLoraWeight,
)
from text_generation_server.utils.cache_manager_flashinfer import (
    KvCacheBatchPosition,
    KvCachePool,
)

# from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

from tokenizers import processors
from transformers.utils import logging

from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.layernorm import FastRMSNorm

from text_generation_server.layers.flashinfer_attention import (
    POS_ENCODING_MODE,
    FlashinferAttentionWrapper,
    AttentionRotaryParams,
)


class FlashinferBatch:
    def __init__(self, seq_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len):
        self.seq_indptr = seq_indptr
        self.kv_page_indptr = kv_page_indptr
        self.kv_page_indices = kv_page_indices
        self.kv_last_page_len = kv_last_page_len


class MistralConfig(PretrainedConfig):
    model_type = "mistral"

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        sliding_window=None,
        speculator=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.speculator = speculator

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
    )


class MistralAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        flashinferWrapper: FlashinferAttentionWrapper,
        config,
        weights,
        layer_idx: int,
    ):
        super().__init__()

        self.flashinferWrapper = flashinferWrapper

        self.rotaryParams = AttentionRotaryParams(
            pos_encoding_mode=POS_ENCODING_MODE.ROPE_LLAMA,
            rope_scale=None,
            rope_theta=config.rope_theta,
        )

        self.layer_idx = layer_idx
        self.qkv_proj = load_attention(config, prefix, weights)
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

        self.max_past = (
            config.sliding_window if config.sliding_window is not None else -1
        )
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_size,
            base=config.rope_theta,
            device=weights.device,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )

        self.query_key_value = load_attention(config, prefix, weights)

        self.layer_idx = layer_idx
        self.config = config

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        loraWeight: BatchedModelLoraWeight | None,
    ) -> torch.Tensor:
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
        attn_outputs_raw = self.flashinferWrapper.computeAttention(
            q,
            k,
            v,
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


class MistralMLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx: int):
        super().__init__()
        act = config.hidden_act
        self.layer_idx = layer_idx
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
        # Fuse gate and up proj
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

    def forward(self, hidden_states, lora: BatchedModelLoraWeight | None):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        gate = gate_up_states[:, 0].contiguous()
        if lora:
            add_lora(
                gate,
                hidden_states,
                lora.gate.wa_ptr,
                lora.gate.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )

        gate = self.act(gate)
        up = gate_up_states[:, 1].contiguous()
        if lora:
            add_lora(
                up,
                hidden_states,
                lora.up.wa_ptr,
                lora.up.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
        t = gate * up
        down = self.down_proj(t)
        if lora:
            add_lora(
                down,
                hidden_states,
                lora.down.wa_ptr,
                lora.down.wb_ptr,
                lora.segment,
                self.layer_idx,
                lora.rank,
            )
        return down
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class MistralLayer(nn.Module):
    def __init__(
        self, flashinferWrapper: FlashinferAttentionWrapper, layer_id, config, weights
    ):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = MistralAttention(
            prefix=f"{prefix}.self_attn",
            flashinferWrapper=flashinferWrapper,
            config=config,
            weights=weights,
            layer_idx=layer_id,
        )
        self.mlp = MistralMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id
        )

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            kvCachePool,
            is_prefill,
            batch_position,
            lora,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output, lora)

        return mlp_output, attn_res


class FlashMistralModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        embed_norm = config.hidden_size**0.5
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        # self.embed_tokens.weight *= embed_norm

        assert config.num_attention_heads % weights.process_group.size() == 0
        assert config.num_key_value_heads % weights.process_group.size() == 0
        num_attention_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()

        self.flashinferWrapper = FlashinferAttentionWrapper(
            num_attention_heads, num_key_value_heads, config.hidden_size
        )

        self.layers = nn.ModuleList(
            [
                MistralLayer(
                    self.flashinferWrapper,
                    layer_id,
                    config=config,
                    weights=weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix=f"model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None,
    ):
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        self.flashinferWrapper.prepareAttention(
            is_prefill,
            batch_position,
            kvCachePool.page_len,
            POS_ENCODING_MODE.ROPE_LLAMA,
            kvCachePool.cache_data[0].dtype,
        )
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                kvCachePool,
                is_prefill,
                batch_position,
                lora,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        self.flashinferWrapper.endBatchAttention(is_prefill)
        return hidden_states


class FlashMistralForCausalLM(torch.nn.Module):
    def __init__(self, prefix, config, weights, name=None):
        if name is None:
            name = "model"
        super().__init__()
        # self.embed_tokens = TensorParallelEmbedding(
        #     prefix=(
        #         f"{name}.embed_tokens"
        #         if not prefix
        #         else f"{prefix}.{name}.embed_tokens"
        #     ),
        #     weights=weights,
        # )
        self.model = FlashMistralModel(
            config=config,
            weights=weights,
        )
        self.lm_head = SpeculativeHead.load(
            config,
            # TODO dirty hack for idefics2.
            prefix=(
                "lm_head" if not prefix or name != "model" else f"{prefix}.lm_head"
            ),
            weights=weights,
        )
        self.max_past = config.sliding_window
        self.max_past_tensor = (
            torch.tensor(config.sliding_window, device=weights.device)
            if self.max_past is not None
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kvCachePool: KvCachePool,
        is_prefill: bool,
        batch_position: KvCacheBatchPosition,
        lora: BatchedModelLoraWeight | None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, kvCachePool, is_prefill, batch_position, lora
        )
        logits = self.lm_head(hidden_states)
        return logits
