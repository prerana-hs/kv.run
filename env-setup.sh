curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

sudo apt-get install unzip
sudo apt install pkg-config
sudo apt-get install libssl-dev gcc -y
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

export BUILD_EXTENSIONS=True
make install
text-generation-launcher --model-id tjluyao/llama-3-8b --disable-flashinfer