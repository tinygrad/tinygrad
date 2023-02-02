FROM ubuntu:20.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip git wget

WORKDIR /tinygrad

COPY accel /tinygrad/accel
COPY tinygrad /tinygrad/tinygrad
COPY README.md /tinygrad/
COPY setup.py /tinygrad/

RUN pip3 install --no-cache-dir /tinygrad

WORKDIR /src
ENV PYTHONPATH /src

CMD ["/usr/bin/env", "python3"]

FROM base AS testing

RUN pip3 install --no-cache-dir "tinygrad[testing]"
RUN wget https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py -O /usr/local/lib/python3.8/dist-packages/google/protobuf/internal/builder.py

COPY test ./test/
COPY extra ./extra/
COPY models ./models/
COPY datasets ./datasets/

FROM base AS llvm

RUN pip3 install --no-cache-dir "tinygrad[llvm]"

FROM base AS opencl

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-runtime-compilers intel-oneapi-runtime-opencl

RUN pip3 install --no-cache-dir "tinygrad[gpu]"

FROM opencl AS cuda

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -O /tmp/cuda-keyring_1.0-1_all.deb && dpkg -i /tmp/cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get install -y nvidia-cuda-toolkit cuda

RUN pip3 install --no-cache-dir "tinygrad[cuda]"