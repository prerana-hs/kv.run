<<<<<<< HEAD
# kv.run
(Limited) comparison of popular model serving solutions

| Solution        | Inference backend | Serving backend      | Advanced kernel support                                                                          | Model support              |
|-----------------|-------------------|----------------------|--------------------------------------------------------------------------------------------------|----------------------------|
| Huggingface TGI | Pytorch           | HF TGI (Rust)        | Paged + Flash attention                                                                          | Language                   |
| Deepspeed MII   | PyTorch           | Deepspeed (Python)   | [DeepSpeed-Kernels](https://github.com/microsoft/DeepSpeed-Kernels)                              | Language                   |
| TensorRT-LLM    | TensorRT-LLM      | TensorRT-LLM (C++)   | [TensorRT XQA](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md) | Language                   |
| vLLM            | vLLM              | vLLM (Python)        | Paged + Flash attention                                                                          | Language                   |
| kv.run          | PyTorch           | HF TGI + more (Rust) | Paged + Flash attention, [FlashInfer](https://github.com/flashinfer-ai/flashinfer)               | Language, diffusion (exp.) |



## Installation
#### Sync submodules
```shell
git submodule sync
git submodule update --init
```

#### Install Rust
[Script](https://rustup.rs/):
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
#### Install Protobuf
```shell
sudo apt-get install libssl-dev gcc -y
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

#### Build Code Base
```shell
make install
```

#### Install Kernel Libraries
```shell
# Install FlashInfer
# For CUDA 12.1 & torch 2.3
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
# For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html

# Install Flash and Paged Attention
cd server
make install-flash-attention && make install-vllm-cuda
```
You can debug/edit code in the build folder. When done, use python copy_back.py to copy changes back to the original src folder.

## Usages
#### Local API tests
```shell
cd server/examples && python test_local_api.py
```
#### Local UI demo
(Inherited from [Punica](https://github.com/punica-ai/punica))

[demo.mp4](https://github.com/mlsys-io/kv.run/assets/12567967/977b09fb-bd90-4757-85ab-e5fc2a58cd93)

#### Deploy services
```shell
text-generation-launcher --model-id tjluyao/llama-3-8b --lora-ids tjluyao/llama-3-8b-math;tjluyao/llama-3-8b-zh
```
#### Using quantized models
Add --quantize [Method] to the command above, for example:
```shell
text-generation-launcher --model-id TechxGenus/gemma-2b-GPTQ --lora-ids tjluyao/gemma-2b-it-math --quantize gptq
```
The supported quantization methods include:
- AWQ: 4-bit. Need specific quantized model.
- EETQ: 8-bit. Can work for any model.
- GPTQ: 4-bit. Need specific quantized model.
- Bitandbytes: 8-bit. Can work for any model.

For AWQ and EETQ quantization, you need to build their specific kernels:
```shell
# AWQ
cd server && make install-awq
git clone https://github.com/casper-hansen/AutoAWQ && cd AutoAWQ
pip install -e .
# EETQ
cd server && make install-eetq
# GTPQ
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install -vvv --no-build-isolation -e .
```

## Model support matrix
Note: L = Language, I = Image

| Model                                                                        | MOE  | Size  | Modality | Quantization | Tensor Parallelism | FlashInfer | Multi-LoRA |
|------------------------------------------------------------------------------|------|-------|----------|--------------|--------------------|------------|------------|
| [Idefics](https://huggingface.co/HuggingFaceM4/idefics-9b)                   |     | 9B    | L, I ⇒ L |              |                    |            |            |
| [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b)                |     | 8B    | L, I ⇒ L |              |                    |            |            |
| [Llava Next (1.6)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |     | 13B   | L, I ⇒ L |              |                    |            |            |
| [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf)                   |     | 7B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)        |     | 8B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Phi 1.5](https://huggingface.co/microsoft/phi-1_5)                          |     | 1.3B  | L ⇒ L    |              |                    |            |            |
| [Phi 3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)             |     | 3.8B  | L ⇒ L    |              |                    |            |            |
| [Gemma](https://huggingface.co/google/gemma-2b)                              |     | 2B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)             |     | 104B  | L ⇒ L    |              |                    |            |            |
| [Dbrx](https://huggingface.co/databricks/dbrx-instruct)                      | ✔    | 132B  | L ⇒ L   |              |                    |            |            |
| [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)               |     | 2.8B  | L ⇒ L    |              |                    |            |            |
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)         |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)      | ✔    | 8x22B | L ⇒ L   |              |                    |            |            |
| [Gpt Bigcode](https://huggingface.co/bigcode/gpt_bigcode-santacoder)         |     | 1.1B  | L ⇒ L    |              |                    |            |            |
| [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)            |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                   |     | 7B    | L ⇒ L    |              | ✔                  |            |            |
| [StarCoder 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)   |     | 15B   | L ⇒ L    |              |                    |            |            |
| [Qwen 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)        |     | 15B   | L ⇒ L    |              |                    |            |            |
| [Opt](https://huggingface.co/facebook/opt-6.7b)                              |     | 6.7B  | L ⇒ L    |              |                    |            |            |
| [T5](https://huggingface.co/google-t5/t5-11b)                                |     | 11B   | L ⇒ L    |              |                    |            |            |
| [Galactica](https://huggingface.co/facebook/galactica-120b)                  |     | 120B  | L ⇒ L    |              |                    |            |            |
| [SantaCoder](https://huggingface.co/bigcode/santacoder)                      |     | 1.1B  | L ⇒ L    |              |                    |            |            |
| [Bloom](https://huggingface.co/bigscience/bloom-560m)                        |     | 560M  | L ⇒ L    |              |                    |            |            |
| [Mpt](https://huggingface.co/mosaicml/mpt-7b-instruct)                       |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Gpt2](https://huggingface.co/openai-community/gpt2)                         |     | 124M  | L ⇒ L    |              |                    |            |            |
| [Gpt Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)                   |     | 20B   | L ⇒ L    |              | ✔                  |            |            |
=======
# kv.run
(Limited) comparison of popular model serving solutions

| Solution        | Inference backend | Serving backend      | Advanced kernel support                                                                          | Model support              |  
|-----------------|-------------------|----------------------|--------------------------------------------------------------------------------------------------|----------------------------|
| Huggingface TGI | Pytorch           | HF TGI (Rust)        | Paged + Flash attention                                                                          | Language                   | 
| Deepspeed MII   | PyTorch           | Deepspeed (Python)   | [DeepSpeed-Kernels](https://github.com/microsoft/DeepSpeed-Kernels)                              | Language                   |
| TensorRT-LLM    | TensorRT-LLM      | TensorRT-LLM (C++)   | [TensorRT XQA](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md) | Language                   |
| vLLM            | vLLM              | vLLM (Python)        | Paged + Flash attention                                                                          | Language                   |
| kv.run          | PyTorch           | HF TGI + more (Rust) | Paged + Flash attention, [FlashInfer](https://github.com/flashinfer-ai/flashinfer)               | Language, diffusion (exp.) |

 

## Installation
#### Sync submodules
```shell
git submodule sync
git submodule update --init
```

#### Install Rust
[Script](https://rustup.rs/):
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
#### Install Protobuf
```shell
sudo apt-get install libssl-dev gcc -y
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

#### Build Code Base
```shell
make install
```

#### Install Kernel Libraries
```shell
# Install FlashInfer
# For CUDA 12.1 & torch 2.3
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
# For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html

# Install Flash and Paged Attention
cd build/server 
make install-flash-attention & make install-vllm-cuda
```
You can debug/edit code in the build folder. When done, use python copy_back.py to copy changes back to the original src folder.

## Usages
#### Local API tests
```shell
cd build/server/examples & python test_local_api.py
```
#### Deploy services
```shell
text-generation-launcher --model-id tjluyao/llama-3-8b --lora-ids tjluyao/llama-3-8b-math;tjluyao/llama-3-8b-zh
```
#### Using quantized models
Add --quantize [Method] to the command above, for example:
```shell
text-generation-launcher --model-id TechxGenus/gemma-2b-GPTQ --lora-ids tjluyao/gemma-2b-it-math --quantize gptq
```
The supported quantization methods include: 
- AWQ: 4-bit. Need specific quantized model. 
- EETQ: 8-bit. Can work for any model. 
- GPTQ: 4-bit. Need specific quantized model.
- Bitandbytes: 8-bit. Can work for any model.

For AWQ and EETQ quantization, you need to build their specific kernels: 
```shell
# AWQ
cd build/server & make install-awq
# EETQ
cd build/server & make install-eetq
```

## Model support matrix
Note: L = Language, I = Image

<<<<<<< HEAD
| Model                                                                        | MOE  | Size  | Modality | Quantization | Tensor Parallelism | FlashInfer | Multi-LoRA |   
|------------------------------------------------------------------------------|------|-------|----------|--------------|--------------------|------------|------------|
| [Idefics](https://huggingface.co/HuggingFaceM4/idefics-9b)                   |     | 9B    | L, I ⇒ L |              |                    |            |            |
| [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b)                |     | 8B    | L, I ⇒ L |              |                    |            |            |
| [Llava Next (1.6)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |     | 13B   | L, I ⇒ L |              |                    |            |            |
| [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf)                   |     | 7B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)        |     | 8B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Phi 1.5](https://huggingface.co/microsoft/phi-1_5)                          |     | 1.3B  | L ⇒ L    |              |                    |            |            |
| [Phi 3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)             |     | 3.8B  | L ⇒ L    |              |                    |            |            |
| [Gemma](https://huggingface.co/google/gemma-2b)                              |     | 2B    | L ⇒ L    |              |                    | ✔          | ✔          |
| [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)             |     | 104B  | L ⇒ L    |              |                    |            |            |
| [Dbrx](https://huggingface.co/databricks/dbrx-instruct)                      | ✔    | 132B  | L ⇒ L   |              |                    |            |            |
| [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)               |     | 2.8B  | L ⇒ L    |              |                    |            |            |
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)         |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)      | ✔    | 8x22B | L ⇒ L   |              |                    |            |            |
| [Gpt Bigcode](https://huggingface.co/bigcode/gpt_bigcode-santacoder)         |     | 1.1B  | L ⇒ L    |              |                    |            |            |
| [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)            |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                   |     | 7B    | L ⇒ L    |              | ✔                  |            |            |
| [StarCoder 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)   |     | 15B   | L ⇒ L    |              |                    |            |            |
| [Qwen 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)        |     | 15B   | L ⇒ L    |              |                    |            |            |
| [Opt](https://huggingface.co/facebook/opt-6.7b)                              |     | 6.7B  | L ⇒ L    |              |                    |            |            |
| [T5](https://huggingface.co/google-t5/t5-11b)                                |     | 11B   | L ⇒ L    |              |                    |            |            |
| [Galactica](https://huggingface.co/facebook/galactica-120b)                  |     | 120B  | L ⇒ L    |              |                    |            |            |
| [SantaCoder](https://huggingface.co/bigcode/santacoder)                      |     | 1.1B  | L ⇒ L    |              |                    |            |            |
| [Bloom](https://huggingface.co/bigscience/bloom-560m)                        |     | 560M  | L ⇒ L    |              |                    |            |            |
| [Mpt](https://huggingface.co/mosaicml/mpt-7b-instruct)                       |     | 7B    | L ⇒ L    |              |                    |            |            |
| [Gpt2](https://huggingface.co/openai-community/gpt2)                         |     | 124M  | L ⇒ L    |              |                    |            |            |
| [Gpt Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)                   |     | 20B   | L ⇒ L    |              | ✔                  |            |            |
=======
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv --disable-pytest-warnings -m "punica_test" build/server/tests
```

# Single Device Multi-GPU Support

## tgi_server.server
server is called by cli

<img width="618" alt="截屏2024-03-31 03 51 00" src="https://github.com/nativ-ai/torch-MIL/assets/104136162/f565447c-000f-4b29-b504-8f4294c4bdd9">
<img width="666" alt="截屏2024-03-31 03 51 29" src="https://github.com/nativ-ai/torch-MIL/assets/104136162/91463205-bf77-48df-9d12-71460a3986f1">

<<<<<<< HEAD
<<<<<<< HEAD
pb folder files are automated generaeted by gRPC, in the case of TGI, use
```bash
<<<<<<< HEAD
cd /torch-MIL/third_party/text-generation-inference/server
make install
```

## Model Deployment and Reasoning
In this setion, FlashLlama is used as an example. Root path for the folllowing relative path is: text-generation-inference/server/text_generation_server

### Tensor Parallel
References from Zhihu: 
[Megatron详解](https://zhuanlan.zhihu.com/p/366906920)
[Tensor Parallel](https://zhuanlan.zhihu.com/p/626008269)

Column-wise Parallel: `TensorParallelColumnLinear()` @utils/layers.py
```python
class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load_multi(cls, config, prefixes: List[str], weights, bias: bool, dim: int):
        weight = weights.get_multi_weights_col(
            prefixes, quantize=config.quantize, dim=dim
        )

        if bias:
            b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=dim)
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)
```

Row-wise Parallel: `TensorParallelRowLinear()` @utils/layers.py
```python
class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(weight, bias, config.quantize),
            process_group=weights.process_group,
        )

    def forward(self, input: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1 and reduce:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out
```

All-Reduce: `torch.distributed.all_reduce()`


#### Attention
![image](https://github.com/kvrun/Model-Serving/assets/104136162/6325b2d1-d011-4443-b960-1edfa25ee370)

<<<<<<< HEAD
``` python
# FlashLlamaAttention() @models/custom_modeling/flash_llama_modeling.py
class FlashLlamaAttention(torch.nn.Module): 
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_size,
            base=config.rope_theta,
            device=weights.device,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )

        # Tensor Parallel for QKV is dealt here
        # -> load_attention()@models/custom_modeling/flash_llama_modeling.py: 105 `return TensorParallelColumnLinear.load_multi()`
        self.query_key_value = load_attention(config, prefix, weights)

        # Tensor Parallel for B is delat here
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
=======


**QKV**: FlashLlamaAttention()@models/custom_modeling/flash_llama_modeling.py: 171  `self.query_key_value = load_attention(config, prefix, weights)`
-> 
load_attention()@models/custom_modeling/flash_llama_modeling.py: 105 `return TensorParallelColumnLinear.load_multi()`

**B**: FlashLlamaAttention()@models/custom_modeling/flash_llama_modeling.py: 173 `self.o_proj = TensorParallelRowLinear.load()`
>>>>>>> ddffeed (Update README.md)

        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

<<<<<<< HEAD
```


#### FeedForward
![image](https://github.com/kvrun/Model-Serving/assets/104136162/79b01fb7-e664-41f1-8bc1-186d48e7e9fc)

``` python
# LlamaMLP() @models/custom_modeling/flash_llama_modeling.py
class LlamaMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
        # Fuse gate and up proj, A is dealt here
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        # B is dealt here
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

```
=======
#### FeedForward
![image](https://github.com/kvrun/Model-Serving/assets/104136162/79b01fb7-e664-41f1-8bc1-186d48e7e9fc)

``` python
# LlamaMLP() @models/custom_modeling/flash_llama_modeling.py
class LlamaMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
        # Fuse gate and up proj, A is dealt here
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        # B is dealt here
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

```
>>>>>>> ddffeed (Update README.md)


### Model load
The following code offers a calling logic in FlashLlama model of TGI, in this case, the methods uses distributed calculation listed.

```python
# server()  @server.py:
    model = get_model()

#---------------->
# get_model()  @models/__init__py:
    return Flashllama()

#---------------->
# FlashLlama()  @models/flash_llma.py:
  def __init__():
    self.process_group, rank, world_size = initialize_torch_distributed() # 33
  
    filenames = weights_files()  #65
    weights = Weights(filenames)  #66
      
    model = FlashLlamaForCausalLM(prefix, config, weights)  #71

#---------------->
# FlashLlamaForCausalLM()  @models/custom_modeling/flash_llama_modeling.py:
  def __init__(self, prefix, config, weights):  #412
    self.embed_tokens = TensorParallelEmbedding()
    self.model = FlashLlamaModel(prefix, config, weights)
    self.lm_head = SpeculativeHead.load() # if !use_medusa: lm_head = TensorParallelHead()

#---------------->
#FlashLlamaModel()  @models/custom_modeling/flash_llama_modeling.py:
  process_group = weights.process_group
  self.tp_rank = process_group.rank()
  self.tp_world_size = process_group.size()
  self.layers = nn.ModuleList([FlashLlamaLayer() for i in range(n_layers)])

#---------------->
# FlashLlamaLayer   @models/custom_modeling/flash_llama_modeling.py:
  self.self_attn = FlashLlamaAttention()
  self.mlp = LlamaMLP()
  self.input_layernorm = FastRMSNorm.load()
  self.post_attention_layernorm = FastRMSNorm.load()

```
<<<<<<< HEAD

Some of the important class and functions used:

#### Weight() @utils/weight.py
``` python
class Weight():
    def __init__():
        #...
        self.process_group = process_group
        #...

    def get_partial_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        size = slice_.get_shape()[dim]
        block_size = (size + world_size - 1) // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype != torch.int32:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        assert (
            size % world_size == 0
        ), f"The choosen size {size} is not compatible with sharding on {world_size} shards"
        return self.get_partial_sharded(tensor_name, dim)

```


=======
HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv --disable-pytest-warnings -m "punica_test" build/server/tests
```

# Single Device Multi-GPU Support

## 

## tgi_server.server
server is called by cli

<img width="618" alt="截屏2024-03-31 03 51 00" src="https://github.com/nativ-ai/torch-MIL/assets/104136162/f565447c-000f-4b29-b504-8f4294c4bdd9">
<img width="666" alt="截屏2024-03-31 03 51 29" src="https://github.com/nativ-ai/torch-MIL/assets/104136162/91463205-bf77-48df-9d12-71460a3986f1">
=======
>>>>>>> 8633d0c (Update README.md)
But "from text_generation_server.pb import generate_pb2_grpc, generate_pb2", pb folder has .gitignore file in it, and source file not uploaded.
=======
pb folder files are automated generaeted by gRPC, in the case of TGI, use
```bash
cd /torch-MIL/third_party/text-generation-inference/server
make install
'''


=======

Some of the important class and functions used:

#### Weight() @utils/weight.py
``` python
class Weight():
    def __init__():
        #...
        self.process_group = process_group
        #...

    def get_partial_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        size = slice_.get_shape()[dim]
        block_size = (size + world_size - 1) // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype != torch.int32:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        assert (
            size % world_size == 0
        ), f"The choosen size {size} is not compatible with sharding on {world_size} shards"
        return self.get_partial_sharded(tensor_name, dim)

```
>>>>>>> ddffeed (Update README.md)

