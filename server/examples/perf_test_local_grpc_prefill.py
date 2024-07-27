import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
import random, json
from test_cases import DEMO, LoraSpec

lora_specs = {}
for name, spec in DEMO.items():
    lora_prompts, base_prompts = spec.generate_prompts()
    lora_specs[name] = LoraSpec(lora_prompts, base_prompts)


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
            max_new_tokens=2048, stop_sequences=[], ignore_eos_token=True
        ),
        lora_id=lora_id,
    )
    return request


promptOverride = "What is deep learning?"
global_request_id = 0
global_batch_id = 0


def generateBatch(batch_size: int):
    requests = []
    for i in range(batch_size):
        requests.append(
            make_input(
                "tjluyao/gemma-2b-it-math",
                "base",
                id=global_request_id,
                promptOverride=promptOverride,
            )
        )
        global_request_id = global_request_id + 1
    batch_pb2 = generate_pb2.Batch(
        id=global_batch_id, requests=requests, size=len(requests)
    )
    global_batch_id = global_batch_id + 1
    return batch_pb2


num_tests = 10
batch_size = 32

forward_ms_all = []
decode_ms_all = []
total_ms_all = []

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
    print(stub.Info(generate_pb2.InfoRequest()))
    wr = generate_pb2.WarmupRequest(
        batch=generateBatch(2),
        max_total_tokens=2048,
        max_prefill_tokens=1024,
        max_input_length=1024,
    )
    stub.Warmup(wr)
    for i in range(num_tests):
        batch = generateBatch(batch_size)
        pr = generate_pb2.PrefillRequest(batch=batch)
        resp = stub.Prefill(pr)
        forward_ms_all.append(resp.forward_ns / 1e6)
        decode_ms_all.append(resp.decode_ns / 1e6)
        total_ms_all.append(resp.total_ns / 1e6)

print(forward_ms_all)
print(decode_ms_all)
print(total_ms_all)
