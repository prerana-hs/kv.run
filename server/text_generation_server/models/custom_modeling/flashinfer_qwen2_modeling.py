# Adapted from HuggingFace Transformers Library
# https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/models/llama/modeling_llama.py

import math

import torch
import flashinfer
from torch import nn
import torch.distributed

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
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import (
    ACT2FN,
    Qwen2Config,
    PreTrainedModel,
)

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

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

Qwen2Tokenizer = None

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

class Qwen2Config(PretrainedConfig):
    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        speculator=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.speculator = speculator

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Qwen2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        # We need to at least pass vocab_file and merges_file to base class
        # in case a slow tokenizer needs to be initialized; other can be
        # configured through files.
        # following GPT2TokenizerFast, also adding unk_token, bos_token, and eos_token

        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        return rms_norm(hidden_states, self.weight, self.variance_epsilon), residual


class Qwen2MLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        act = config.hidden_act
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


def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
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

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    # manually concatenate qkv project bias
    bias = torch.cat((weights.get_tensor(f"{prefix}.q_proj.bias"),
                      weights.get_tensor(f"{prefix}.k_proj.bias"),
                      weights.get_tensor(f"{prefix}.v_proj.bias")), dim=0)

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
        get_linear(weight, bias=bias, quantize=config.quantize)
    )


class FlashQwen2Attention(nn.Module):
    def __init__(self, prefix: str, config: Qwen2Config, weights, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_qo_heads = self.num_heads // weights.process_group.size()
        self.num_kv_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )
        self.config = config
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.qkv_proj = load_attention(config, prefix, weights)
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
        
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

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
            )
            
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


class FlashQwen2Layer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.layer_id = layer_id
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashQwen2Attention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights, layer_idx=layer_id
        )
        self.mlp = Qwen2MLP(prefix=f"{prefix}.mlp", config=config, weights=weights, layer_idx=layer_id)

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


class FlashQwen2Model(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        # embed_norm = config.hidden_size**0.5
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        # self.embed_tokens.weight *= embed_norm

        self.layers = nn.ModuleList(
            [
                FlashQwen2Layer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
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


class FlashQwen2ForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashQwen2Model(config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="model.embed_tokens" if config.tie_word_embeddings else "lm_head",
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