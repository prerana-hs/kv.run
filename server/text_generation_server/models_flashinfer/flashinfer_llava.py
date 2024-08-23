# Modified from transformers/models/llava_next/modeling_llava_next.py
# Editor: Junyi Shen

import torch, time
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Iterable
from text_generation_server.models import Model
from text_generation_server.models.types import (
    Tokens,
    Generation,
    GeneratedText,
)
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, AutoConfig, AutoModel, AutoProcessor
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from loguru import logger
from text_generation_server.pb import generate_pb2
tracer = trace.get_tracer(__name__)

from text_generation_server.models_flashinfer.flashinfer_causal_lm import (
    FlashinferBatch,
    RequestContext, 
    RequestKvCache,
    getKvCacheBatchPosition,
    KvCacheBatchPosition
)
from text_generation_server.models_flashinfer.flashinfer_llama import FlashinferLlama
from text_generation_server.models.vlm_causal_lm import (
    image_text_replacement,
    load_data_uri,
    split,
)

@dataclass(frozen=True)
class LlavaBatch(FlashinferBatch):
    pixel_values: Optional[List[torch.Tensor]]
    pixel_attention_mask: Optional[List[torch.Tensor]]
    image_sizes: Optional[List[Tuple[int, int]]]
    batch_ids: List[torch.Tensor]

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches):
        batch = super(LlavaBatch, cls).concatenate(batches)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        batch = super().filter(request_ids)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @classmethod
    def batch_tokenized_inputs(cls, requests, tokenizer, processor, config):
        batch_inputs = []
        image_inputs = []
        max_truncation = 0
        for r in requests:
            chunks = split(r.inputs)
            full_text = ""
            image_id = 0
            for chunk in chunks:
                if chunk["type"] == "text":
                    full_text += chunk["content"]
                elif chunk["type"] == "image":
                    image = chunk["content"]
                    # Should never receive URLs anymore, processing should be done
                    # On the rust layer.
                    # This avoid making n queries per TP
                    # if image.startswith("https://") or image.startswith("http://"):
                    #     image = processor.image_processor.fetch_images(image)
                    if image.startswith("data:"):
                        image = load_data_uri(image)
                    else:
                        raise RuntimeError(
                            "Cannot process input image not starting with data:"
                        )
                    image_input = processor.image_processor(image, return_tensors="pt")
                    full_text += image_text_replacement(image_input, config, image_id)
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk['type']}")

            batch_inputs.append(full_text)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs,
            truncation=True,
            max_length=max_truncation,
            add_special_tokens=not config.model_type == "paligemma",
        )["input_ids"]
        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat(
                    [img["pixel_values"] for img in image_inputs], dim=0
                ),
            }
            if "pixel_attention_mask" in image_input:
                new_image_inputs["pixel_attention_mask"] = torch.cat(
                    [img["pixel_attention_mask"] for img in image_inputs], dim=0
                )
            if "image_sizes" in image_input:
                new_image_inputs["image_sizes"] = torch.cat(
                    [img["image_sizes"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
        else:
            image_inputs = None
        return batch_tokenized_inputs, image_inputs

    @classmethod
    def from_pb_processor(
        self,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
        service
    ) -> "LlavaBatch":
        batch_tokenized_inputs, image_inputs = self.batch_tokenized_inputs(
            pb.requests, tokenizer, processor, config
        )
        self.batch_ids = batch_tokenized_inputs
        batch = self.from_tokenized(pb, batch_tokenized_inputs, service)
        if image_inputs is not None:
            batch.pixel_values = image_inputs["pixel_values"].to(device=device)
            if "pixel_attention_mask" in image_inputs:
                batch.pixel_attention_mask = image_inputs["pixel_attention_mask"].to(
                    device=device
                )
            else:
                batch.pixel_attention_mask = None
            if "image_sizes" in image_inputs:
                batch.image_sizes = image_inputs["image_sizes"].to(device=device)
            else:
                batch.image_sizes = None
        else:
            batch.pixel_values = None
            batch.pixel_attention_mask = None
            batch.image_sizes = None
        return batch
    
    @classmethod
    def from_tokenized(
        self, 
        batchPb: generate_pb2.Batch,
        batch_tokenized_inputs: List[torch.Tensor],
        service
        ) -> FlashinferBatch:
        request_contexts = []

        for i,request in enumerate(batchPb.requests):
            input_ids = batch_tokenized_inputs[i]
            parameters = request.parameters
            request_context = RequestContext(
                request.id,
                input_ids,
                next_token_chooser_parameter=parameters,
                maxlen=min(request.stopping_parameters.max_new_tokens, 4096),
                stop_token_id=service.tokenizer.eos_token_id,
                is_stopped=False,
                request_kv_cache=RequestKvCache(
                    service.kvCachePool,
                    service.kvCachePool.page_len,
                    len(input_ids+576),
                ),
                prefill_logprobs=request.prefill_logprobs,
                lora_id=request.lora_id,
            )

            request_contexts.append(request_context)

        return FlashinferBatch(
            batch_id=batchPb.id, is_prefill=True, request_contexts=request_contexts
        )
        
class LlavaLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False
    ):  
        # Initialize LlavaLM
        self.config = AutoConfig.from_pretrained(model_id)
        self.vision_tower = AutoModel.from_config(self.config.vision_config)
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self.multi_modal_projector = LlavaMultiModalProjector(self.config)
        self.vocab_size = self.config.text_config.vocab_size
        
        llama_config = AutoConfig.from_pretrained('lmsys/vicuna-7b-v1.5')
        setattr(self.config, 'num_attention_heads', llama_config.num_attention_heads)
        
        self.language_model = FlashinferLlama(
            model_id= model_id,
            lora_ids= None, 
            revision= revision,
            quantize= quantize,
            dtype= dtype,
            trust_remote_code= trust_remote_code, 
        )
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.id_embedder = self.language_model.model.embed_tokens
        self.vision_feature_select_strategy = self.config.vision_feature_select_strategy
        logger.info(f"Initialized LlavaLM with model_id: {model_id}")
        
    @property
    def batch_type(self) -> Type[LlavaBatch]:
        return LlavaBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.language_model.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
    @tracer.start_as_current_span("generate_token")
    @torch.no_grad()
    def generate_token(
        self, 
        batch: LlavaBatch,
        embeddings: torch.Tensor = None,
    ) -> Tuple[List[Generation], Optional[LlavaBatch], Tuple[int, int]]:
        start = time.time_ns()
        input_ids, lora_ids, lora_lens = [], [], []
        request_kv_caches = []
        all_input_ids_stacked: List[List[int]] = []
        for request_context in batch.request_contexts:
            if not request_context.is_stopped:
                all_input_ids_stacked.append(request_context.output_ids)
                if batch.is_prefill:
                    input_ids.extend(request_context.output_ids)
                else:
                    input_ids.append(request_context.output_ids[-1])
                request_kv_caches.append(request_context.request_kv_cache)
                if not batch.is_prefill:
                    request_context.request_kv_cache.increment()

                if lora_ids and lora_ids[-1] == request_context.lora_id:
                    lora_lens[-1] += 1
                elif request_context.lora_id:
                    lora_ids.append(request_context.lora_id)
                    lora_lens.append(1)

        all_input_ids_tensor = self.language_model._get_all_input_ids_tensor(
            all_input_ids_stacked, batch.request_contexts
        )
        input_ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.language_model.device,
        )

        batch_position: KvCacheBatchPosition = getKvCacheBatchPosition(
            request_kv_caches, isPrefill=batch.is_prefill, device=self.language_model.device
        )

        loraWeights = (
            self.language_model.loraManager.get_lora_batched_weights(lora_ids, lora_lens)
            if lora_ids
            else None
        )
        
        raw_logits, _ = self.language_model.model(
            input_ids = input_ids_tensor if embeddings is None else None,
            kvCachePool = self.language_model.kvCachePool,
            is_prefill = batch.is_prefill,
            batch_position = batch_position,
            loraWeights = loraWeights,
            input_embeddings = embeddings,
        )

        start_decode = time.time_ns()
        logits = (
            raw_logits[batch_position.seq_indptr[1:] - 1]
            if batch.is_prefill
            else raw_logits
        )

        all_stop = True
        generations: List[Generation] = []
        num_stopped_requests = 0
        start_next_token_id = time.time_ns()

        next_token_ids, next_token_logprobs, logprobs, _, _ = (
            self.language_model._get_next_batch_token_id_heterogeneous(
                batch.request_contexts, all_input_ids_tensor, logits
            )
        )
        next_token_id_ns = time.time_ns() - start_next_token_id

        for i, request_context in enumerate(batch.request_contexts):
            if request_context.is_stopped:
                num_stopped_requests += 1
                continue
            next_token_id = next_token_ids[i - num_stopped_requests]
            request_context.append_token(next_token_id)
            text = self.language_model.tokenizer.decode(
                next_token_id,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )

            stop_reason = request_context.get_stop_reason()
            if stop_reason != None:
                output_text = self.language_model.tokenizer.decode(
                    request_context.output_ids[request_context.prompt_len :],
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
                generated_text = GeneratedText(
                    output_text,
                    len(request_context.output_ids) - request_context.prompt_len + 1,
                    stop_reason,
                    None,
                )
                request_context.is_stopped = True
                request_context.request_kv_cache.release()
            else:
                generated_text = None
                all_stop = False

            request_context.prefill_tokens = None

            generation = Generation(
                request_context.request_id,
                request_context.prefill_tokens,
                Tokens(
                    [next_token_id],
                    [0],  # prob
                    [text],
                    [next_token_id in self.language_model.all_special_ids],
                ),
                generated_text,
                # top_tokens
                None,
            )
            generations.append(generation)

        forward_ns = start_decode - start
        decode_ns = next_token_id_ns
        # The router stops generation only when batch=None
        if all_stop:
            return generations, None, (forward_ns, decode_ns)
        else:
            return generations, batch, (forward_ns, decode_ns)
        
    def decode_batch(
        self, cachedBatchesPb: Iterable[generate_pb2.CachedBatch]
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], Tuple[int, int], int]:
        start_concat = time.time_ns()
        batch = self.language_model._convertCachedBatch(cachedBatchesPb)
        concat_ns = time.time_ns() - start_concat
        generations, next_batch, timings = self.generate_token(batch)
        if next_batch:
            self.language_model.batch_cache.set(next_batch)
        return generations, next_batch, timings, concat_ns

    def prefill_batch(
        self, batch
    ) -> Tuple[List[Generation], Optional[FlashinferBatch], Tuple[int, int]]:
        
        embeds_ids = torch.tensor(batch.batch_ids, device=self.language_model.device)
        embeds_ids[(batch.input_ids == self.config.image_token_index)] = 0
        inputs_embeds = self.id_embedder(embeds_ids)
        
        image_outputs = self.vision_tower(batch.pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        image_features = self.multi_modal_projector(selected_image_feature)
        inputs_embeds = inputs_embeds.to(image_features.dtype)
        
        input_embeddings = torch.cat([image_features,inputs_embeds], dim=1).half()
        generations, next_batch, timings = self.generate_token(batch, input_embeddings)
        if next_batch:
            self.language_model.batch_cache.set(next_batch)
        return generations, batch, timings
    
if __name__ == '__main__':
    model = LlavaLM(model_id='llava-hf/llava-v1.6-vicuna-7b-hf')
    print(model)