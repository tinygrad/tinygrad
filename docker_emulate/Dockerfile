FROM --platform=linux/amd64 catthehacker/ubuntu:act-latest
RUN apt update && apt install -y git curl python3-pip && rm -rf /var/lib/apt/lists/*
# amd
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'Acquire::http::Pipeline-Depth "5";' > /etc/apt/apt.conf.d/99parallel \
    && apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor > /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian jammy main" > /etc/apt/sources.list.d/rocm.list \
    && echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update \
    && apt-get install --no-install-recommends --allow-unauthenticated -y hsa-rocr comgr hsa-rocr-dev liburing-dev libc6-dev rocm-llvm clang \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /root/code/local_tinygrad
# tinygrad
RUN git clone --depth 1 https://github.com/Qazalin/tinygrad.git /root/code/tinygrad/tinygrad
WORKDIR /root/code/tinygrad/tinygrad
RUN git remote remove origin && git remote add origin https://github.com/Qazalin/tinygrad.git
RUN git fetch --depth 1 origin remu-server && git checkout remu-server
RUN pip install -e .

RUN apt-get install libc6

EXPOSE 80
CMD ["/bin/bash"]