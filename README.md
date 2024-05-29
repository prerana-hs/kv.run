# kv.run
(Limited) comparison of popular model serving solutions

| Solution        | Inference backend | Server backend       | Advanced kernel support                                                                          | Model support              |  
|-----------------|-------------------|----------------------|--------------------------------------------------------------------------------------------------|----------------------------|
| Huggingface TGI | Pytorch           | HF TGI (Rust)        | Paged + Flash attention                                                                          | Language                   | 
| Deepspeed MII   | PyTorch           | Deepspeed (Python)   | [DeepSpeed-Kernels](https://github.com/microsoft/DeepSpeed-Kernels)                              | Language                   |
| TensorRT-LLM    | TensorRT-LLM      | TensorRT-LLM (C++)   | [TensorRT XQA](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md) | Language                   |
| vLLM            | vLLM              | vLLM (Python)        | Paged + Flash attention                                                                          | Language                   |
| kv.run          | PyTorch           | HF TGI + more (Rust) | Paged + Flash attention, [FlashInfer](https://github.com/flashinfer-ai/flashinfer)               | Language, diffusion (exp.) |

 

## Installation
### Sync submodules
```shell
git submodule sync
git submodule update --init
```

### Install Rust
[Script](https://rustup.rs/): e.g.,
```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
### Install proto
```shell
sudo apt-get install libssl-dev gcc -y
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```
### Install FlashInfer
```shell
# For CUDA 12.1 & torch 2.3
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
# For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html
```

### Build Code Base
```shell
# Build server only
make codebase
make install-server
# Build all 
make install
```

You can debug/edit code in the build folder. When done, use python copy_back.py to copy changes back to the original src folder.

## Model support matrix

| Model                                                                        | MOE  | Size  | Modality     | Quantization    | Tensor Parallelism    | Kernels             | Multi-LoRA |   
|------------------------------------------------------------------------------|------|-------|--------------|-----------------|-----------------------|---------------------|------------|
| [Idefics](https://huggingface.co/HuggingFaceM4/idefics-9b)                   |     | 9B    | Lang, Img ⇒ Lang     |                 |                       | TGI ✔ FlashInfer   |           |
| [Idefics 2](https://huggingface.co/HuggingFaceM4/idefics2-8b)                |     | 8B    | Lang, Img ⇒ Lang     |                 |                       | TGI ✔ FlashInfer   |           |
| [Llava Next (1.6)](https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf) |     | 13B   | Lang, Img ⇒ Lang     |                 |                       | TGI ✔ FlashInfer   |           |
| [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-hf)                   |     | 7B    | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer ✔  | ✔          |
| [Llama 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)        |     | 8B    | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer ✔  | ✔          |
| [Phi 1.5](https://huggingface.co/microsoft/phi-1_5)                          |     | 1.3B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Phi 3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)             |     | 3.8B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Gemma](https://huggingface.co/google/gemma-2b)                              |     | 2B    | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer ✔  | ✔          |
| [Cohere](https://huggingface.co/CohereForAI/c4ai-command-r-plus)             |     | 104B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Dbrx](https://huggingface.co/databricks/dbrx-instruct)                      | ✔    | 132B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Mamba](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)               |     | 2.8B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)         |     | 7B    | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Mixtral](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)      | ✔    | 8x22B | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Gpt Bigcode](https://huggingface.co/bigcode/gpt_bigcode-santacoder)         |     | 1.1B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)            |     | 7B    | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Falcon](https://huggingface.co/tiiuae/falcon-7b-instruct)                   |     | 7B    | Lang ⇒ Lang        |                 |    ✔                   | TGI ✔ FlashInfer   |           |
| [StarCoder 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)   |     | 15B   | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Qwen 2](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1)        |     | 15B   | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Opt](https://huggingface.co/facebook/opt-6.7b)                              |     | 6.7B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [T5](https://huggingface.co/google-t5/t5-11b)                                |     | 11B   | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Galactica](https://huggingface.co/facebook/galactica-120b)                  |     | 120B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [SantaCoder](https://huggingface.co/bigcode/santacoder)                      |     | 1.1B  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Bloom](https://huggingface.co/bigscience/bloom-560m)                        |     | 560M  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Mpt](https://huggingface.co/mosaicml/mpt-7b-instruct)                       |     | 7B    | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Gpt2](https://huggingface.co/openai-community/gpt2)                         |     | 124M  | Lang ⇒ Lang        |                 |                       | TGI ✔ FlashInfer   |           |
| [Gpt Neox](https://huggingface.co/EleutherAI/gpt-neox-20b)                   |     | 20B   | Lang ⇒ Lang        |                 |      ✔                 | TGI ✔ FlashInfer   |           |
