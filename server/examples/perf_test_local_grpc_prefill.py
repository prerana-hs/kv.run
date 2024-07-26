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
requests = []
batch_size = 32
for i in range(batch_size):
    requests.append(
        make_input(
            "tjluyao/gemma-2b-it-math", "base", id=i, promptOverride=promptOverride
        )
    )

# Assemble input batch
pb_batch_with_inputs = generate_pb2.Batch(id=0, requests=requests, size=len(requests))
pb_batch_empty = generate_pb2.Batch()

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)

    # Info
    print(stub.Info(generate_pb2.InfoRequest()))
    # Warm up
    wr = generate_pb2.WarmupRequest(
        batch=pb_batch_with_inputs,
        max_total_tokens=2048,
        max_prefill_tokens=1024,
        max_input_length=1024,
    )
    stub.Warmup(wr)
    # Prefill
    pr = generate_pb2.PrefillRequest(batch=pb_batch_with_inputs)
    resp = stub.Prefill(pr)
    generations, cbatch, forward_ns, decode_ns, total_ns = (
        resp.generations,
        resp.batch,
        resp.forward_ns,
        resp.decode_ns,
        resp.total_ns,
    )
    print(
        f"Finished prefilling :\n"
        f"forward_ns: {forward_ns/1e6}ms \n"
        f"decode_ns: {decode_ns/1e6}ms \n"
        f"total_ns: {total_ns/1e6}ms \n"
    )
