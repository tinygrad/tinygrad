FROM ubuntu:latest

RUN \
  export DEBIAN_FRONTEND=noninteractive && \
  apt-get update -y && \
  apt-get install -y --no-install-recommends git g++ cmake ninja-build llvm-15-dev libz-dev libglew-dev flex bison libfl-dev libboost-thread-dev libboost-filesystem-dev nvidia-cuda-toolkit-gcc ca-certificates

RUN \
  git clone --recurse-submodules https://github.com/gpuocelot/gpuocelot.git && \
  cd gpuocelot/ocelot && \
  git checkout 19626fc00b6ee321638c3111074269c69050e091 && \
  mkdir build && \
  cd build && \
  cmake .. -Wno-dev -G Ninja -DOCELOT_BUILD_TOOLS=OFF && \
  ninja && \
  ninja install