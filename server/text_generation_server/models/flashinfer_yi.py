import torch
import torch.distributed
from typing import Optional, List
from text_generation_server.models.flashinfer_causal_lm import FlashinferLM

from text_generation_server.models.custom_modeling.flashinfer_yi_modeling import (
    YiTokenizerFast,
    YiConfig,
    FlashYiForCausalLM,
)
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from text_generation_server.models.custom_modeling.flashinfer_llama_modeling import (
    FlashLlamaForCausalLM
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)


class FlashinferYi(FlashinferLM):
    def __init__(
        self,
        model_id: str,
        lora_ids: List[str] = None,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        trust_remote_code: bool = False,
    ):
        process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        else:
            raise NotImplementedError("Flashinfer Yi is only available on GPU")

        # yiConfig = YiConfig.from_pretrained(
        #     model_id, revision=revision, trust_remote_code=trust_remote_code
        # )
        yiConfig = AutoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        yiConfig.quantize = quantize
        yiConfig.speculator = speculator
        torch.distributed.barrier(group=process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=process_group)
        if quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        # model = FlashYiForCausalLM(yiConfig, weights)
        prefix = ""
        model = FlashLlamaForCausalLM(prefix, yiConfig, weights)
        # tokenizer = YiTokenizerFast.from_pretrained(
        #     model_id,
        #     revision=revision,
        #     padding_side="left",
        #     truncation_side="left",
        #     trust_remote_code=trust_remote_code,
        #     use_fast=True,
        #     from_slow=False,
        # )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        super(FlashinferYi, self).__init__(
            model=model,
            tokenizer=tokenizer,
            config=yiConfig,
            dtype=dtype,
            device=device,
            lora_ids=lora_ids,
        )
