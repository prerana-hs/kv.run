# Pytorch-MIL
```bash
git submodule sync
git submodule update --init
```

## To compile server code with kernels

```bash
make codebase
make install-server
```

You can debug/edit code in the build folder. When done, use python copy_back.py to copy changes back to the original src folder.


## To compile all

```bash
make install
```

## To test Punica code

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 pytest -s -vv --disable-pytest-warnings -m "punica_test" build/server/tests
```

# Single Device Multi-GPU Support

## tgi_server.server
server is called by cli

<img width="618" alt="截屏2024-03-31 03 51 00" src="https://github.com/nativ-ai/torch-MIL/assets/104136162/f565447c-000f-4b29-b504-8f4294c4bdd9">
<img width="666" alt="截屏2024-03-31 03 51 29" src="https://github.com/nativ-ai/torch-MIL/assets/104136162/91463205-bf77-48df-9d12-71460a3986f1">

pb folder files are automated generaeted by gRPC, in the case of TGI, use
```bash
cd /torch-MIL/third_party/text-generation-inference/server
make install
```

## Model Deployment and Reasoning
In this setion, FlashLlama is used as an example. Root path for the folllowing relative path is: text-generation-inference/server/text_generation_server

### Tensor Parallel
A Reference from Zhihu: [Link](https://zhuanlan.zhihu.com/p/626008269)

Column-wise Parallel: `TensorParallelColumnLinear()` @utils/layers.py

Row-wise Parallel: `TensorParallelRowLinear()` @utils/layers.py

All-Reduce: 

### Attention
![image](https://github.com/kvrun/Model-Serving/assets/104136162/6325b2d1-d011-4443-b960-1edfa25ee370)

**QKV**: FlashLlamaAttention()@models/custom_modeling/flash_llama_modeling.py: 171  `self.query_key_value = load_attention(config, prefix, weights)`
-> 
load_attention()@models/custom_modeling/flash_llama_modeling.py: 105 `return TensorParallelColumnLinear.load_multi()`

**B**: FlashLlamaAttention()@models/custom_modeling/flash_llama_modeling.py: 173 


### FeedForward
![image](https://github.com/kvrun/Model-Serving/assets/104136162/79b01fb7-e664-41f1-8bc1-186d48e7e9fc)

**A**: LlamaMLP()@models/custom_modeling/flash_llama_modeling.py: 260

**B**: LlamaMLP()@models/custom_modeling/flash_llama_modeling.py: 267

