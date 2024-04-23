import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models import Model, get_model
from transformers import AutoTokenizer
import torch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch
import random
from test_cases import DEMO, LoraSpec

model = PunicaLM(model_id="meta-llama/Llama-2-7b-hf",
               lora_ids={'gsm8k':'abcdabcd987/gsm8k-llama2-7b-lora-16'})
print(model.get_lora_adapters())

model.remove_lora_adapters(['gsm8k'])
print(model.get_lora_adapters())
model.remove_lora_adapters()
print(model.get_lora_adapters())

model.load_lora_adapters({'gsm8k':'abcdabcd987/gsm8k-llama2-7b-lora-16',
                         'sqlctx':'abcdabcd987/sqlctx-llama2-7b-lora-16',
                         'viggo':'abcdabcd987/viggo-llama2-7b-lora-16'})
print(model.get_lora_adapters())

tokenizer = model.tokenizer

lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)

def make_input(model_name, lora_or_base, id = 0):
    if lora_or_base == "lora":
        prompts = lora_specs[model_name].lora_prompts
        lora_id = model_name
    elif lora_or_base == "base":
        prompts = lora_specs[model_name].base_prompts
        lora_id = "empty"
    else:
        raise ValueError(f"Unknown lora_or_base={lora_or_base}")
    prompt = prompts[1] #random.choice(prompts)

    # Try out prefill / decode from the client side
    request = generate_pb2.Request(
        inputs=prompt,
        lora_id=lora_id,
        id=id,
        truncate=256,
        prefill_logprobs=True,
        top_n_tokens=20,
        parameters=generate_pb2.NextTokenChooserParameters(
            temperature=0.9,
            top_k=10,
            top_p=0.9,
            typical_p=0.9,
            repetition_penalty=1.1
        ),
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=2048,
            stop_sequences=[],
            ignore_eos_token=True))
    return request

requests = [make_input('gsm8k', 'lora', 2), make_input('gsm8k', 'base', 1)]

batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))
pb_batch = PunicaBatch.from_pb(batch, tokenizer, torch.float16, torch.device("cuda"))

model.add_request(pb_batch)

results = {}
for r in requests:
    results[r.id] = []

empty_pb_batch = PunicaBatch.from_pb(generate_pb2.Batch())

while True:
    generations, _, _ = model.generate_token(empty_pb_batch)
    if not generations:
        break
    for gen in generations:
        results[gen.request_id].append(gen.tokens.texts)

for id in results:
    print(str(id) + '='*30)
    print(''.join([r[0] for r in results[id]]))
