FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg xz-utils && \
    echo "deb [ allow-insecure=yes ] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.1 noble main" > /etc/apt/sources.list.d/rocm.list && \
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600 && \
    apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
      intel-oneapi-runtime-openmp=2023.2.1-16 intel-oneapi-runtime-compilers-common=2023.2.1-16 intel-oneapi-runtime-compilers=2023.2.1-16 \
      intel-oneapi-runtime-dpcpp-sycl-opencl-cpu=2023.2.1-16 intel-oneapi-runtime-tbb-common=2021.10.0-49541 \
      intel-oneapi-runtime-tbb=2021.10.0-49541 intel-oneapi-runtime-opencl=2023.2.1-16 \
      comgr && \
    mkdir -p /artifacts && \
    cp -a --parents \
      /opt/rocm/lib/libamd_comgr.so* \
      /opt/intel/oneapi/lib/clbltfnshared.rtl \
      /opt/intel/oneapi/lib/intel64/ \
      /artifacts/ && \
    mkdir -p /artifacts/etc/OpenCL/vendors && \
    echo /opt/intel/oneapi/lib/intel64/libintelocl.so.2023.16.7.0 > /artifacts/etc/OpenCL/vendors/intel64.icd && \
    mkdir -p /artifacts/usr/local/cuda/targets/x86_64-linux /artifacts/usr/local/lib /artifacts/usr/lib && \
    curl -fL https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-11.5.119-archive.tar.xz \
      | tar -xJ -C /artifacts/usr/local/cuda/targets/x86_64-linux --strip-components=1 && \
    curl --output-dir /artifacts/usr/local/lib -fLO https://github.com/tinygrad/gpuocelot/releases/download/v0.1.0/libgpuocelot.so && \
    curl -fL https://github.com/wpmed92/pydawn/releases/download/v0.1.6/libwebgpu_dawn.so -o /artifacts/usr/local/lib/libwebgpu_dawn.so && \
    curl -fL https://github.com/sirhcm/tinymesa/releases/download/v1/libtinymesa-mesa-25.2.7-linux-amd64.so -o /artifacts/usr/lib/libtinymesa.so && \
    curl -fL https://github.com/sirhcm/tinymesa/releases/download/v1/libtinymesa_cpu-mesa-25.2.7-linux-amd64.so -o /artifacts/usr/lib/libtinymesa_cpu.so

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git sudo \
      opencl-headers ocl-icd-libopencl1 \
      mesa-vulkan-drivers \
      libllvm20 clang-20 lld-20 \
      qemu-user-static && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /artifacts/ /

RUN useradd -m -u 1001 runner && \
    echo 'runner ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/runner && \
    chmod 0440 /etc/sudoers.d/runner && \
    echo /opt/intel/oneapi/lib/intel64 > /etc/ld.so.conf.d/intel-oneapi.conf && \
    echo /usr/local/cuda/targets/x86_64-linux/lib > /etc/ld.so.conf.d/cuda-nvrtc.conf && \
    ln -s /usr/bin/clang-20 /usr/local/bin/clang && \
    ldconfig

USER runner
