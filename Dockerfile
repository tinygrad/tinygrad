FROM ubuntu:latest

RUN \
  apt update -y && \
  apt install -y --no-install-recommends git g++ cmake ninja-build llvm-15-dev zlib1g-dev libglew-dev flex bison libfl-dev libboost-thread-dev libboost-filesystem-dev nvidia-cuda-toolkit-gcc ca-certificates \
  software-properties-common gpg-agent clang

RUN \
  export DEBIAN_FRONTEND=noninteractive && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt update -y && \
  apt install -y python3.8 python3-pip && \
  ln -s /usr/bin/python3 /usr/bin/python && \
  rm -rf /var/lib/apt/lists/*

RUN \
  git clone --recurse-submodules https://github.com/gpuocelot/gpuocelot.git && \
  cd gpuocelot/ocelot && \
  git checkout 19626fc00b6ee321638c3111074269c69050e091 && \
  mkdir build && \
  cd build && \
  cmake .. -Wno-dev -G Ninja -DOCELOT_BUILD_TOOLS=OFF && \
  ninja && \
  ninja install && \
  rm -rf /gpuocelot