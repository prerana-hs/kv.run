from text_generation_server.pb import generate_pb2
import torch
from text_generation_server.models.flash_causal_lm import (
    FlashCausalLMBatch
)
from text_generation_server.models.flash_llama import (
    FlashLlama
)
from text_generation_server.models.causal_lm import CausalLM, CausalLMBatch
import random, json
from test_cases import DEMO, LoraSpec
from collections import defaultdict
# Load demo inputs
lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)
from text_generation_server.utils.speculate import get_speculate, set_speculate


# Create input requests
def make_input(lora_id, lora_or_base, id=0, promptOverride=None):
    if lora_or_base == "lora":
        prompts = lora_specs[lora_id].lora_prompts
    elif lora_or_base == "base" or lora_or_base == "empty":
        prompts = lora_specs[lora_id].base_prompts
        lora_id = "empty"
    else:
        raise ValueError(f"Unknown lora_or_base={lora_or_base}")
    prompt = random.choice(prompts) if not promptOverride else promptOverride
    inputs = prompt

    request = generate_pb2.Request(
        id=id,
        inputs=inputs,
        truncate=256,
        prefill_logprobs=True,
        top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            repetition_penalty=1.1,
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=256, stop_sequences=[], ignore_eos_token=True
        ),
        lora_id=lora_id,
    )
    return request


flash = False
set_speculate(0)

if flash:
    service = FlashLlama(model_id="baichuan-inc/Baichuan2-7B-Chat", trust_remote_code=True, dtype=torch.bfloat16)
else:
    service = CausalLM(model_id="baichuan-inc/Baichuan2-7B-Chat", trust_remote_code=True, dtype=torch.bfloat16)
requests = [
    make_input(
        "abcdabcd987/gsm8k-llama2-7b-lora-16",
        "base",
        id=0,
        promptOverride="why is deep learning so popular these days?",
    )
]

tokenizer = service.tokenizer
batch = generate_pb2.Batch(id=0, requests=requests, size=len(requests))
if flash:
    pb_batch = FlashCausalLMBatch.from_pb(
        batch, tokenizer, torch.bfloat16, torch.device("cuda")
    )
else:
    pb_batch = CausalLMBatch.from_pb(
        batch, tokenizer, torch.bfloat16, torch.device("cuda")
    )

display_results = defaultdict(lambda: [])

service.warmup(pb_batch)

while True:
    generations, next_batch, _ = service.generate_token(pb_batch)

    for gen in generations:
        if gen.prefill_tokens:
            display_results[gen.request_id] = [
                "Prompt:\n"
                + tokenizer.decode(gen.prefill_tokens.token_ids)
                + "\nAnswer:\n"
            ]
        if gen.generated_text:
            display_results[gen.request_id] += [gen.generated_text.text]
    # Stop if all input generations are done
    if all([g.generated_text for g in generations]):
        break

for id in display_results:
    print(str(id) + "=" * 30)
    print("".join(display_results[id]))
