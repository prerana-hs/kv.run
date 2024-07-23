import asyncio
import os
import torch
import time
import signal

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from text_generation_server.cache import Cache
from text_generation_server.interceptor import ExceptionInterceptor
from text_generation_server.models_flashinfer import get_model
from text_generation_server.models_flashinfer.flashinfer_causal_lm import FlashinferLM

from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.tracing import UDSOpenTelemetryAioServerInterceptor


class SignalHandler:
    KEEP_PROCESSING = True

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print(f"Exiting gracefully: Signal {signum}")
        self.KEEP_PROCESSING = False


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(
        self,
        model: FlashinferLM,
        cache: Cache,
        quantize: Optional[str],
        server_urls: List[str],
    ):
        self.cache = cache
        self.model = model
        self.quantize = quantize
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of TextGenerationService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return generate_pb2.HealthResponse()

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        self.model.clear_cache()
        return generate_pb2.ClearCacheResponse()

    async def FilterBatch(self, request, context):
        flashinferBatch = self.model.filter_batch(request.batch_id)
        return generate_pb2.FilterBatchResponse(batch=flashinferBatch.to_pb())

    async def Warmup(self, request, context):
        return generate_pb2.WarmupResponse(
            max_supported_total_tokens=request.max_total_tokens
        )

    async def Prefill(self, request, context):
        start = time.time_ns()
        generations, next_batch, timings = self.model.prefill_batch(request.batch)
        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )

    async def Decode(self, request, context):
        start = time.time_ns()
        generations, next_batch, timings, concat_ns = self.model.decode_batch(
            request.batches
        )
        return generate_pb2.DecodeResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            concat_ns=concat_ns,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )


def serve(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[str],
    trust_remote_code: bool,
    uds_path: Path,
    lora_ids: Optional[str],
):
    async def serve_inner(
        model_id: str,
        revision: Optional[str],
        sharded: bool = False,
        quantize: Optional[str] = None,
        speculate: Optional[int] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        unix_socket_template = "unix://{}-{}"
        if sharded:
            server_urls = [
                unix_socket_template.format(uds_path, rank)
                for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        try:
            model = get_model(
                model_id,
                revision,
                sharded,
                quantize,
                dtype,
                trust_remote_code,
                lora_ids,
            )
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
            TextGenerationService(model, Cache(), quantize, server_urls), server
        )
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["TextGenerationService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)

        await server.start()

        logger.info("Server started at {}".format(local_url))
        signal_handler = SignalHandler()
        while signal_handler.KEEP_PROCESSING:
            await asyncio.sleep(0.5)

    asyncio.run(
        serve_inner(
            model_id, revision, sharded, quantize, speculate, dtype, trust_remote_code
        )
    )
