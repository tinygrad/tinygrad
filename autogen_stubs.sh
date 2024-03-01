#!/bin/bash -e

# setup instructions for clang2py
if [[ ! $(clang2py -V) ]]; then
  pushd .
  cd /tmp
  sudo apt-get install -y --no-install-recommends clang
  pip install clang==14.0.6
  git clone https://github.com/geohot/ctypeslib.git
  cd ctypeslib
  pip install --user .
  clang2py -V
  popd
fi

BASE=tinygrad/runtime/autogen/

fixup() {
  sed -i '1s/^/# mypy: ignore-errors\n/' $1
  sed -i 's/ *$//' $1
  grep FIXME_STUB $1 || true
}

generate_opencl() {
  clang2py /usr/include/CL/cl.h -o $BASE/opencl.py -l /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -k cdefstum
  fixup $BASE/opencl.py
  # hot patches
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/opencl.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libOpenCL.so.1')\ctypes.CDLL(ctypes.util.find_library('OpenCL'))\g" $BASE/opencl.py
  python3 -c "import tinygrad.runtime.autogen.opencl"
}

generate_hip() {
  clang2py /opt/rocm/include/hip/hip_ext.h /opt/rocm/include/hip/hiprtc.h \
  /opt/rocm/include/hip/hip_runtime_api.h /opt/rocm/include/hip/driver_types.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o $BASE/hip.py -l /opt/rocm/lib/libamdhip64.so
  echo "hipDeviceProp_t = hipDeviceProp_tR0600" >> $BASE/hip.py
  echo "hipGetDeviceProperties = hipGetDevicePropertiesR0600" >> $BASE/hip.py
  fixup $BASE/hip.py
  # we can trust HIP is always at /opt/rocm/lib
  #sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libhiprtc.so')\ctypes.CDLL(ctypes.util.find_library('hiprtc'))\g" $BASE/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')\ctypes.CDLL(ctypes.util.find_library('amdhip64'))\g" $BASE/hip.py
  python3 -c "import tinygrad.runtime.autogen.hip"

  clang2py /opt/rocm/include/amd_comgr/amd_comgr.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o $BASE/comgr.py -l /opt/rocm/lib/libamd_comgr.so
  fixup $BASE/comgr.py
  python3 -c "import tinygrad.runtime.autogen.comgr"
}

generate_cuda() {
  clang2py /usr/include/cuda.h /usr/include/nvrtc.h -o $BASE/cuda.py -l /usr/lib/x86_64-linux-gnu/libcuda.so -l /usr/lib/x86_64-linux-gnu/libnvrtc.so
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" $BASE/cuda.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so')\ctypes.CDLL(ctypes.util.find_library('cuda'))\g" $BASE/cuda.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libnvrtc.so')\ctypes.CDLL(ctypes.util.find_library('nvrtc'))\g" $BASE/cuda.py
  fixup $BASE/cuda.py
  python3 -c "import tinygrad.runtime.autogen.cuda"
}

generate_hsa() {
  clang2py \
    /opt/rocm/include/hsa/hsa.h \
    /opt/rocm/include/hsa/hsa_ext_amd.h \
    /opt/rocm/include/hsa/hsa_ext_finalize.h /opt/rocm/include/hsa/hsa_ext_image.h \
    --clang-args="-I/opt/rocm/include" \
    -o $BASE/hsa.py -l /opt/rocm/lib/libhsa-runtime64.so
  fixup $BASE/hsa.py
  python3 -c "import tinygrad.runtime.autogen.hsa"
}

if [ "$1" == "opencl" ]; then generate_opencl
elif [ "$1" == "hip" ]; then generate_hip
elif [ "$1" == "cuda" ]; then generate_cuda
elif [ "$1" == "hsa" ]; then generate_hsa
elif [ "$1" == "all" ]; then generate_opencl; generate_hip; generate_cuda; generate_hsa
else echo "usage: $0 <type>"
fi
