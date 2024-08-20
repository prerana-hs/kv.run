# Modified from Llava Official Repo
# Editor: Junyi Shen

import torch, time, re
from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict
from text_generation_server.models import Model
from text_generation_server.models.types import (
    Tokens,
    Generation,
    GeneratedText,
)
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, AutoConfig, AutoModel
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from loguru import logger
from io import BytesIO
from PIL import Image
import base64
from text_generation_server.pb import generate_pb2
tracer = trace.get_tracer(__name__)

from text_generation_server.models_flashinfer.flashinfer_causal_lm import FlashinferBatch
from text_generation_server.models_flashinfer.flashinfer_llama import FlashinferLlama

from text_generation_server.models.causal_lm import (
    NextTokenChooser,
    StoppingCriteria,
    CausalLMBatch,
)

IMAGES = re.compile(r"!\[[^\]]*\]\((.*?)\s*(\"(?:.*[^\"])\")?\s*\)")
def split(string) -> List[Dict[str, str]]:
    parts = []
    cursor = 0
    for pattern in IMAGES.finditer(string):
        start = pattern.start()
        if start != cursor:
            parts.append({"type": "text", "content": string[cursor:start]})

        parts.append({"type": "image", "content": pattern.group(1)})
        cursor = pattern.end()

    if cursor != len(string):
        parts.append({"type": "text", "content": string[cursor:]})

    return parts
def load_data_uri(image_uri: str) -> Image.Image:
    image_uri = image_uri.split(",")[-1]
    content = base64.b64decode(image_uri)
    image = Image.open(BytesIO(content))
    return image

@dataclass(frozen=True)
class LlavaBatch(FlashinferBatch):
    pixel_values: Optional[List[torch.Tensor]] = None
    pixel_attention_mask: Optional[List[torch.Tensor]] = None
    image_sizes: Optional[List[Tuple[int, int]]] = None

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
                    image_input = processor(image, return_tensors="pt")
                    #full_text += image_text_replacement(image_input, config, image_id)
                    full_text += "<image>"
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk['type']}")

            batch_inputs.append(full_text)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs, padding=True, truncation=True, max_length=max_truncation
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
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "LlavaBatch":
        batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(
            pb.requests, tokenizer, processor, config
        )
        batch = cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)
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
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase = None,
        batch_tokenized_inputs = None,
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
        for i, (r, tokenized_inputs) in enumerate(zip(pb.requests, batch_tokenized_inputs)):

            next_token_choosers.append(
                NextTokenChooser.from_pb(r.parameters, device, tokenizer)
            )
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)
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
        self.id_embedder = self.language_model.model.model.embed_tokens
        self.vision_feature_select_strategy = self.config.vision_feature_select_strategy

        logger.info(f"Initialized LlavaLM with model_id: {model_id}")
        
    @property
    def batch_type(self) -> Type[LlavaBatch]:
        return LlavaBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.language_model.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
        
    @torch.no_grad()
    def prefill_token(
        self, 
        batch: LlavaBatch, 
    ):
        logger.info('Prefilling token')
        ids = self.language_model.add_request(batch)
        image_outputs = self.vision_tower(batch.pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.config.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        image_features = self.multi_modal_projector(selected_image_feature)
        input_ids = torch.tensor(batch.input_ids, device=self.device)
        inputs_embeds = self.id_embedder(input_ids)
        inputs_embeds = inputs_embeds.to(image_features.dtype)

        input_embeddings = torch.cat([image_features,inputs_embeds], dim=1).half()
        lens = [input_embeddings.size(1) for _ in batch.requests]

        batchKvCache = self.language_model.modelKvCache.getOrCreate(batch.batch_id)
        prefill_reqIds = []
        for r,l in zip(batch.requests,lens):
            prefill_reqIds.append(r.id)
            batchKvCache.create(r.id, l)
        prefillBatchPosition = batchKvCache.getKvCacheBatchPosition(prefill_reqIds, isPrefill=True)
        decodeBatchPosition = batchKvCache.getKvCacheBatchPosition([], isPrefill=False)

        raw_logits, _ = self.language_model.model(
            None, 
            self.language_model.modelKvCache.kvCachePool, 
            prefillBatchPosition, 
            decodeBatchPosition, 
            None,
            input_embeddings,
            )

        logits = raw_logits[prefillBatchPosition.seq_indptr[1:] - 1] if prefillBatchPosition.total_seq_len > 0 else torch.tensor([], device=self.device)
        return logits
    
    @torch.no_grad()
    def generate_token(
        self, batch: LlavaBatch
    ):
        logger.info('Generating token')
        input_ids = []
        batchKvCache = self.language_model.modelKvCache.getOrCreate(batch.batch_id)
        decode_reqIds = []

        for request in batch.requests:
            id = request.id
            reqctx = self.language_model.reqctx.get(id)
            input_ids.append(reqctx.output_ids[-1])
            decode_reqIds.append(id)
            batchKvCache.get(id).increment()

        input_ids = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=self.device,
            )
        
        prefillBatchPosition = batchKvCache.getKvCacheBatchPosition([], isPrefill=True)
        decodeBatchPosition = batchKvCache.getKvCacheBatchPosition(decode_reqIds, isPrefill=False)

        logits, _ = self.language_model.model(
            input_ids, 
            self.language_model.modelKvCache.kvCachePool, 
            prefillBatchPosition, 
            decodeBatchPosition, 
            None)
        return logits
        
    def generate(
        self, batch: LlavaBatch
    )-> Tuple[List[Generation], Optional[LlavaBatch], Tuple[int, int]]:
        start = time.time_ns()
        if batch.pixel_values is not None:
            logits = self.prefill_token(batch)
            batch.pixel_values = None
        else:
            logits = self.generate_token(batch)
        
        start_decode = time.time_ns()
        generations: List[Generation] = []
        for i, (request) in enumerate(batch.requests):
            id = request.id
            reqctx = self.language_model.reqctx[id]
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            text = reqctx.decode_tokens()

            is_stop = reqctx.is_stop()
            if is_stop:
                output_text, _, _  = self.language_model.decode_token(reqctx.output_ids[:reqctx.read_offset], skip_special_tokens=True)
                generated_text = GeneratedText(output_text, reqctx.read_offset, 0, None)
                self.language_model.reqctx.pop(id)
                batchKvCache = self.language_model.modelKvCache.getOrCreate(batch.batch_id)
                batchKvCache.release(id)
                batch.requests.remove(request)
            else:
                generated_text = None

            generation = Generation(
                id, None,
                Tokens(
                    [next_token_id],
                    reqctx.output_ids[reqctx.prefix_offset : reqctx.read_offset],
                    [text],
                    [next_token_id in self.language_model.all_special_ids]
                ),
                generated_text, None,
            )
            generations.append(generation)

        return generations, batch, (start_decode - start, time.time_ns() - start_decode)
    
if __name__ == '__main__':
    model = LlavaLM(model_id='liuhaotian/llava-v1.5-7b')
    print(model)