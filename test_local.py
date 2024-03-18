import grpc
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.models import Model, get_model
from transformers import AutoTokenizer
import torch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.bloom import BloomCausalLMBatch, BLOOMSharded

# put this file in ROOT\server, so you don't need to compile TGI

# Start the local server:
# SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=1 text_generation_server/cli.py serve bigscience/bloom-560m --sharded

model_id = "bigscience/bloom-560m"
revision = "main"
filenames = weight_hub_files(model_id, revision, ".safetensors")
download_weights(filenames, model_id, revision)
default_bloom = BLOOMSharded(model_id)
bloom_560m_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", padding_side="left")

# Assemble input batch
default_pb_batch = generate_pb2.Batch(id = 0, requests = requests, size = len(requests))

default_bloom_batch = BloomCausalLMBatch.from_pb(default_pb_batch, bloom_560m_tokenizer, torch.float32, torch.device("cuda"))
generations, next_batch, _ = default_bloom.generate_token(default_bloom_batch)


# Try out prefill / decode from the client side
request = generate_pb2.Request(
    inputs="What is deep learning?",
    id=0,
    truncate=1024,
    prefill_logprobs=True,
    top_n_tokens=20,
    parameters=generate_pb2.NextTokenChooserParameters(
        temperature=0.9,
        top_k=10,
        top_p=0.9,
        typical_p=0.9,
        do_sample=False,
        seed=0,
        repetition_penalty=1.2,
        frequency_penalty=0.1,
        watermark=True,
        grammar='',
        grammar_type=0),
    stopping_parameters=generate_pb2.StoppingCriteriaParameters(
        max_new_tokens=1024,
        stop_sequences=[],
        ignore_eos_token=True))
requests = [req] * 1

with grpc.insecure_channel("unix:///tmp/text-generation-server-0") as channel:
    # Info
    stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
    print(stub.Info(generate_pb2.InfoRequest()))

    # Warm up
    wr = generate_pb2.WarmupRequest(batch = default_pb_batch, max_total_tokens = 2048, max_prefill_tokens = 1024*10, max_input_length = 1024)
    stub.Warmup(wr)

    # Prefill
    pr = generate_pb2.PrefillRequest(batch = batch)
    resp = stub.Prefill(pr)
    gen, cbatch = resp.generations, resp.batch

    # Decode
    dr = generate_pb2.DecodeRequest(batches = [cbatch])
    resp = stub.Decode(dr)
    gen, cbatch = resp.generations, resp.batch

    print('done')
