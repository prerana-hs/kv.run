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

But "from text_generation_server.pb import generate_pb2_grpc, generate_pb2", pb folder has .gitignore file in it, and source file not uploaded.


