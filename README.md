# Pytorch-MIL
```bash
git submodule sync
git submodule update --init
```

## To compile server code with kernels

Make sure you compile/install FlashInfer first.

```bash
make codebase
make install-server
```

You can debug/edit code in the build folder. When done, use python copy_back.py to copy changes back to the original src folder.


## To compile all

```bash
make install
```

## To test Punica Llama with APIs

```bash
cd build/server
python examples/test_local_api.py 
```