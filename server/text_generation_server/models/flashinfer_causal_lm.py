# Modified from https://github.com/punica-ai/punica/blob/master/src/punica/models/llama_lora.py
# Editor: Junyi Shen

import torch
import torch.distributed
from typing import Any, TypedDict, Optional
from transformers.models.llama.modeling_llama import LlamaConfig
from text_generation_server.utils.lora_utils import ModelLoraManager, ModelConfigForLora
from text_generation_server.utils.cache_manager_flashinfer import ModelKvCache, KvCachePool
from .custom_modeling.flashinfer_llama_modeling import LlamaForCausalLM
from .custom_modeling.flashinfer_gemma_modeling import (
    GemmaTokenizerFast,
    FlashGemmaForCausalLM,
    GemmaConfig,
)
from transformers import PreTrainedTokenizerBase
import transformers
from text_generation_server.pb import generate_pb2
import json

import time
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict
from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils import (
    NextTokenChooser,
    StoppingCriteria,
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from text_generation_server.utils.dist import MEMORY_FRACTION
from dataclasses import dataclass
from transformers import AutoTokenizer

from loguru import logger
tracer = trace.get_tracer(__name__)

from .causal_lm import CausalLMBatch

class TextGenerationChunk(TypedDict):
    index: int
    token_id: int
    text: str
    is_stop: bool

@dataclass
class FlashinferBatch(CausalLMBatch):
    @classmethod
    def Empty(cls, batch_id):
        return cls(
            batch_id=batch_id,
            requests=None,
            prefix_offsets=None,
            read_offsets=None,
            next_token_choosers=None,
            stopping_criterias=None,
            top_n_tokens=None,
            top_n_tokens_tensor=None,
            input_ids=None,
            requests_idx_mapping=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            all_input_ids=None,
            input_lengths=None,
            max_input_length=None,
            padding_right_offset=None,
            max_tokens=None,
        )

    @classmethod
    def from_pb(
            cls,
            pb: generate_pb2.Batch,
            tokenizer: PreTrainedTokenizerBase = None,
            dtype: torch.dtype = None,
            device: torch.device = 'cuda',
    ) -> "CausalLMBatch":
        input_ids = []
        next_token_choosers = []
        stopping_criterias = []
        top_n_tokens = []
        prefix_offsets = []
        read_offsets = []

        # Parse batch
        for i, r in enumerate(pb.requests):
            prompt = r.inputs

            next_token_choosers.append(
                NextTokenChooser.from_pb(r.parameters, device, tokenizer)
            )
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)
            tokenized_inputs = tokenizer.encode(prompt)
            input_len = len(tokenized_inputs)
            prefix_offsets.append(input_len - 5)
            read_offsets.append(input_len)
            input_ids.append(tokenized_inputs)

        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=None,
            input_ids=input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            all_input_ids=None,
            input_lengths=None,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=None,
            padding_right_offset=None,
            max_tokens=None,
        )

class RequestContext:
    def __init__(
        self,
        input_ids: list[int],
        lora_id: str,
        tokenizer,
        *,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        maxlen: int,
        stop_token_id: int,
    ):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.maxlen = maxlen
        self.stop_token_id = stop_token_id

        # Logits processing adapted from: https://github.com/lm-sys/FastChat/blob/bb7ca37c2bfad629ba4751dec188bdcdc2cf0c81/fastchat/serve/inference.py
        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        if 0 < top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
        if top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

        self.output_ids = [int(x) for x in input_ids]
        self.prompt_len = len(self.output_ids)
        self.lora_id = lora_id
        self.tokenizer = tokenizer
        self.prefix_offset = 0
        self.read_offset = 0

    def get_next_token_id(self, logits: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits.device)
            else:
                t = None
            last_token_logits = self.logits_processor(t, logits[-1].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

        if self.temperature <= 0 or self.top_p <= 0:
            _, indices = torch.topk(last_token_logits, 2)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
        token = int(indices.tolist()[0])
        return token

    def append_token(self, token_id: int):
        self.output_ids.append(token_id)

    def is_stop(self) -> int:
        if len(self.output_ids) >= self.maxlen:
            return True
        if self.output_ids[-1] == self.stop_token_id:
            return True
        return False

    def is_prefill(self) -> bool:
        return len(self.output_ids) == self.prompt_len

    def decode_tokens(self) -> str:
        # Adapted from: https://github.com/huggingface/text-generation-inference/blob/a5def7c222174e03d815f890093584f3e815c5ce/server/text_generation_server/models/model.py#L68
        prefix_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset : self.read_offset],
            skip_special_tokens=True,
        )
        new_text = self.tokenizer.decode(
            self.output_ids[self.prefix_offset :], skip_special_tokens=True
        )
        if len(new_text) > len(prefix_text) and not new_text.endswith("\uFFFD"):
            new_text = new_text[len(prefix_text) :]
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.output_ids)
            return new_text
        else:
            return ""


class FlashinferLM(Model):
    def __init__(
        self,
        model_type: str,
        model_id: str = None,
        lora_id_path_dict: Dict[str, str] = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False
    ):
        if use_medusa:
            raise RuntimeError("Medusa decoding is not enabled for AutoModel")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        self.device = device
        self.dtype = dtype

        if model_type == "llama":
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = LlamaForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device,
                load_in_8bit=quantize == "bitsandbytes",
                trust_remote_code=trust_remote_code,
            )
        elif model_type == "gemma":
            process_group, rank, world_size = initialize_torch_distributed()
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{rank}")
                dtype = torch.bfloat16 if dtype is None else dtype
            else:
                raise NotImplementedError("FlashGemma is only available on GPU")

            gemmaConfig = GemmaConfig.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            
            gemmaConfig.quantize = quantize
            gemmaConfig.use_medusa = False

            torch.distributed.barrier(group=process_group)

            filenames = weight_files(model_id, revision=revision, extension=".safetensors")
            weights = Weights(filenames, device, dtype, process_group=process_group)
            if gemmaConfig.quantize in ["gptq", "awq"]:
                weights._set_gptq_params(model_id, revision)

            model = FlashGemmaForCausalLM(gemmaConfig, weights)
            tokenizer = GemmaTokenizerFast.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
                use_fast=True,
                from_slow=False,
            )
            model.config = gemmaConfig
        else:
            raise NotImplementedError(f"Flashinfer is not implemented for: {model_type}")     
        
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and quantize != "bitsandbytes"
        ):
            model = model.cuda()

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model_config = model.config
        
        # consider moving it into cache manager
        PAGE_LEN = 16
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        cache_page_size = (
            2 * 
            PAGE_LEN * 
            self.model_config.num_hidden_layers * 
            self.model_config.num_attention_heads * 
            (self.model_config.hidden_size // self.model_config.num_attention_heads) *
            dtype_size
        )
        
        currentDevice = torch.cuda.current_device()
        total_free_memory, _ = torch.cuda.mem_get_info(currentDevice)
        total_gpu_memory = torch.cuda.get_device_properties(currentDevice).total_memory
        free_memory = max(
            0, total_free_memory - (1 - MEMORY_FRACTION) * total_gpu_memory
        )
        num_pages_to_allocate = int(free_memory * 0.80 / cache_page_size)
        print(f"Cache allocation:\n"
            f"  Cache Page Size: {cache_page_size / 1024 / 1024} MB\n"
            f"  Dtype Size: {dtype_size}\n"
            f"  Free Memory: {free_memory / 1024 / 1024 / 1024} GB\n"
            f"  Total GPU Memory: {total_gpu_memory / 1024 / 1024 / 1024} GB\n"
            f"  Number of Pages to Allocate: {num_pages_to_allocate}")

        kvCachePool = KvCachePool(
            max_pages=num_pages_to_allocate,
            num_layers=self.model_config.num_hidden_layers,
            num_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.hidden_size // self.model_config.num_attention_heads,
            page_len=PAGE_LEN,
            dtype=dtype,
            device=device
        )
                
        self.modelKvCache = ModelKvCache(kvCachePool)
        self.model_config_for_lora = ModelConfigForLora(
            num_hidden_layers=self.model_config.num_hidden_layers,
            hidden_size=self.model_config.hidden_size,
            intermediate_size=self.model_config.intermediate_size,
            name_or_path=self.model_config.name_or_path,
        )
        
        self.loraManager = ModelLoraManager(self.model_config_for_lora, dtype, device)
        self.loraManager.set_lora_weights(lora_id_path_dict, self.model_config_for_lora or {}, device, dtype)
        self.reqctx: dict[int, RequestContext] = {}

        super(FlashinferLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    def load_lora_adapters(self, lora_id_path_dict: Dict[str, str]):
        self.loraManager.set_lora_weights(
            lora_id_path_dict,
            self.model_config_for_lora,
            device=self.device,
            dtype = self.dtype,
        )

    def remove_lora_adapters(self, lora_ids: list[str] = None):
        self.loraManager.remove_lora_weights(lora_ids)

    def get_lora_adapters(self):
        return list(self.loraManager.lora_weights)

    def has_request(self):
        return len(self.reqctx)>0

    @property
    def batch_type(self) -> Type[FlashinferBatch]:
        return FlashinferBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def add_request(self, batch: FlashinferBatch):
        ids = []
        for r in range(len(batch.requests)):
            id = batch.requests[r].id
            # Router sends initial request in each iteration
            if id not in self.reqctx:
                lora_id = batch.requests[r].lora_id or 'empty'
                input = batch.input_ids[r]
                parameters = batch.requests[r].parameters
                stop = batch.requests[r].stopping_parameters

                if lora_id not in self.loraManager.lora_weights:
                    raise ValueError("Cannot find lora weights", lora_id)

                self.reqctx[id] = RequestContext(
                    input,
                    lora_id,
                    self.tokenizer,
                    temperature=parameters.temperature,
                    repetition_penalty=parameters.repetition_penalty,
                    top_p=parameters.top_p,
                    top_k=parameters.top_k,
                    maxlen=min(stop.max_new_tokens, 4096),
                    stop_token_id=self.tokenizer.eos_token_id,
                )
                ids.append(id)
        return ids

    @tracer.start_as_current_span("generate_token")
    @torch.no_grad()
    def generate_token(
        self, batch: FlashinferBatch
    )-> Tuple[List[Generation], Optional[FlashinferBatch], Tuple[int, int]]:
        start = time.time_ns()

        if hasattr(batch, 'requests') and batch.requests:
            ids = self.add_request(batch)
            if ids:
                print("====Request " + str(ids) + " added.")

        if not self.reqctx:
            return None, batch, (0, 0)

        reqs = sorted(
            self.reqctx.items(),
            key=lambda req: (not req[1].is_prefill(), req[1].lora_id),
        )

        input_ids = []
        lora_ids, lora_lens = [], []
        batchKvCache = self.modelKvCache.getOrCreate(batch.batch_id)
        prefill_reqIds = []
        decode_reqIds = []

        for requestId, req in reqs:
            if req.is_prefill():
                input_ids.extend(req.output_ids)
                prefill_reqIds.append(requestId)
                batchKvCache.create(requestId, req.prompt_len)
            else:
                input_ids.append(req.output_ids[-1])
                decode_reqIds.append(requestId)
                batchKvCache.get(requestId).increment()
            if lora_ids and lora_ids[-1] == req.lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(req.lora_id)
                lora_lens.append(1)

        input_ids = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=self.device,
            )
        
        prefillBatchPosition = batchKvCache.getKvCacheBatchPosition(prefill_reqIds, isPrefill=True)
        decodeBatchPosition = batchKvCache.getKvCacheBatchPosition(decode_reqIds, isPrefill=False)

        # Forward pass
        raw_logits, _ = self.model(input_ids, self.modelKvCache.kvCachePool, prefillBatchPosition, decodeBatchPosition, \
                                   self.loraManager.get_lora_batched_weights(lora_ids, lora_lens))

        start_decode = time.time_ns()

        prefill_logits = raw_logits[prefillBatchPosition.seq_indptr[1:] - 1] if prefillBatchPosition.total_seq_len > 0 else torch.tensor([], device=self.device)
        decode_logits = raw_logits[prefillBatchPosition.total_seq_len:]
        logits = torch.cat([prefill_logits, decode_logits])

        all_stop = True
        generations: List[Generation] = []
        for i, (reqid, reqctx) in enumerate(reqs):
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            text = reqctx.decode_tokens()

            is_stop = reqctx.is_stop()
            if is_stop:
                output_text, _, _  = self.decode_token(reqctx.output_ids[:reqctx.read_offset], skip_special_tokens=True)
                generated_text = GeneratedText(output_text, reqctx.read_offset, 0, None)
                self.reqctx.pop(reqid)
                batchKvCache.release(reqid)
            else:
                generated_text = None
                all_stop = False

            generation = Generation(
                reqid, None,
                Tokens(
                    [next_token_id],
                    reqctx.output_ids[reqctx.prefix_offset : reqctx.read_offset],
                    [text],
                    [next_token_id in self.all_special_ids]
                ),
                generated_text, None,
            )
            generations.append(generation)

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        # The router stops generation only when batch=None
        if all_stop:
            batch = None
        return generations, batch, (forward_ns, decode_ns)