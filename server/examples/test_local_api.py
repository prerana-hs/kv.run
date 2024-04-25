from text_generation_server.pb import generate_pb2
import torch
from text_generation_server.models.punica_causal_lm import PunicaLM, PunicaBatch, EmptyPunicaBatch
import random, json
from test_cases import DEMO, LoraSpec

# Load model
service = PunicaLM(model_id="meta-llama/Llama-2-7b-hf",
               lora_ids={'gsm8k':'abcdabcd987/gsm8k-llama2-7b-lora-16'})
tokenizer = service.tokenizer

# Test print lora adapters
print(service.get_lora_adapters())

# Test remove lora adapters
service.remove_lora_adapters(['gsm8k'])
print(service.get_lora_adapters())
service.remove_lora_adapters()
print(service.get_lora_adapters())

# Test load lora adapters
service.load_lora_adapters({'gsm8k':'abcdabcd987/gsm8k-llama2-7b-lora-16',
                         'sqlctx':'abcdabcd987/sqlctx-llama2-7b-lora-16',
                         'viggo':'abcdabcd987/viggo-llama2-7b-lora-16'})
print(service.get_lora_adapters())

# Load demo inputs
lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)

# Create input requests
def make_input(model_name, lora_or_base):
    if lora_or_base == "lora":
        prompts = lora_specs[model_name].lora_prompts
        lora_id = model_name
    elif lora_or_base == "base":
        prompts = lora_specs[model_name].base_prompts
        lora_id = "empty"
    else:
        raise ValueError(f"Unknown lora_or_base={lora_or_base}")
    prompt = random.choice(prompts)

    inputs = json.dumps({"inputs": prompt, "lora_id": lora_id})
    request = generate_pb2.Request(
        inputs=inputs,
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

# Create an input batch of two queries
requests = [make_input('gsm8k', 'lora'), make_input('gsm8k', 'base')]
batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))
pb_batch = PunicaBatch.from_pb(batch, tokenizer, torch.float16, torch.device("cuda"))

# Add input batch to model service
service.add_request(pb_batch)

results = {}
for r in pb_batch.requests:
    results[r.id] = []

# Iterative generation: each step generates a token for each input in the batch
while True:
    # When calling iterative text generation, we may add new inputs (use pb_batch like above)
    # or use an empty batch (use EmptyPunicaBatch)
    generations, _, _ = service.generate_token(EmptyPunicaBatch)
    # Stop if all input generations are done
    if not generations:
        break
    for gen in generations:
        results[gen.request_id].append(gen.tokens.texts[0])

for id in results:
    print(str(id) + '='*30)
    print(''.join(results[id]))
