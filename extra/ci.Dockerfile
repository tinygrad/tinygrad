FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg sudo git xz-utils && \
    echo "deb [ allow-insecure=yes ] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1 noble main" > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600 && \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key -o /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-20 main" > /etc/apt/sources.list.d/llvm.list && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
      opencl-headers ocl-icd-libopencl1 \
      intel-oneapi-runtime-openmp=2023.2.1-16 intel-oneapi-runtime-compilers-common=2023.2.1-16 intel-oneapi-runtime-compilers=2023.2.1-16 \
      intel-oneapi-runtime-dpcpp-sycl-opencl-cpu=2023.2.1-16 intel-oneapi-runtime-tbb-common=2021.10.0-49541 \
      intel-oneapi-runtime-tbb=2021.10.0-49541 intel-oneapi-runtime-opencl=2023.2.1-16 \
      comgr \
      mesa-vulkan-drivers \
      libllvm20 clang-20 lld-20 \
      qemu-user-static && \
    rm -rf /var/lib/apt/lists/*

# **** AMD ****
RUN printf '/opt/rocm/lib\n/opt/rocm/lib64\n' >> /etc/ld.so.conf.d/rocm.conf

# **** CUDA (nvrtc only, runs on gpuocelot) ****
RUN mkdir -p /usr/local/cuda/targets/x86_64-linux && \
    curl -fL https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-11.5.119-archive.tar.xz \
      | tar -xJ -C /usr/local/cuda/targets/x86_64-linux --strip-components=1 && \
    echo /usr/local/cuda/targets/x86_64-linux/lib > /etc/ld.so.conf.d/cuda-nvrtc.conf

# **** gpuocelot ****
RUN curl --output-dir /usr/local/lib -fLO https://github.com/tinygrad/gpuocelot/releases/download/v0.1.0/libgpuocelot.so

# **** WebGPU dawn ****
RUN curl -fL https://github.com/wpmed92/pydawn/releases/download/v0.1.6/libwebgpu_dawn.so -o /usr/local/lib/libwebgpu_dawn.so

# **** mesa (both regular and cpu variants) ****
RUN curl -fL https://github.com/sirhcm/tinymesa/releases/download/v1/libtinymesa-mesa-25.2.7-linux-amd64.so -o /usr/lib/libtinymesa.so && \
    curl -fL https://github.com/sirhcm/tinymesa/releases/download/v1/libtinymesa_cpu-mesa-25.2.7-linux-amd64.so -o /usr/lib/libtinymesa_cpu.so

# **** tinydreno ****
RUN curl -fL https://github.com/sirhcm/tinydreno/raw/refs/heads/master/libllvm-qcom.so -o /usr/lib/libllvm-qcom.so

RUN ldconfig

# **** python ****
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
