#!/bin/bash

# setup instructions for clang2py

#sudo apt-get install -y --no-install-recommends clang
#pip install clang==14.0.6
#git clone https://github.com/geohot/ctypeslib.git
#cd ctypeslib
#pip install .
#clang2py -V

generate_opencl() {
  clang2py /usr/include/CL/cl.h -o autogen/opencl.py -l /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -k cdefstum
  grep FIXME_STUB autogen/opencl.py || true
  # hot patches
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" autogen/opencl.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libOpenCL.so.1')\ctypes.CDLL(ctypes.util.find_library('OpenCL'))\g" autogen/opencl.py
  python3 -c "import autogen.opencl"
}

generate_hip() {
  clang2py /opt/rocm/include/hip/hip_ext.h /opt/rocm/include/hip/hiprtc.h \
  /opt/rocm/include/hip/hip_runtime_api.h /opt/rocm/include/hip/driver_types.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o autogen/hip.py -l /opt/rocm/lib/libamdhip64.so
  echo "hipDeviceProp_t = hipDeviceProp_tR0600" >> autogen/hip.py
  echo "hipGetDeviceProperties = hipGetDevicePropertiesR0600" >> autogen/hip.py
  grep FIXME_STUB autogen/hip.py || true
  # we can trust HIP is always at /opt/rocm/lib
  #sed -i "s\import ctypes\import ctypes, ctypes.util\g" autogen/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libhiprtc.so')\ctypes.CDLL(ctypes.util.find_library('hiprtc'))\g" autogen/hip.py
  #sed -i "s\ctypes.CDLL('/opt/rocm/lib/libamdhip64.so')\ctypes.CDLL(ctypes.util.find_library('amdhip64'))\g" autogen/hip.py
  python3 -c "import autogen.hip"

  clang2py /opt/rocm/include/amd_comgr/amd_comgr.h \
  --clang-args="-D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -x c++" -o autogen/comgr.py -l /opt/rocm/lib/libamd_comgr.so
  grep FIXME_STUB autogen/comgr.py || true
  python3 -c "import autogen.comgr"
}

generate_cuda() {
  clang2py /usr/include/cuda.h /usr/include/nvrtc.h -o autogen/cuda.py -l /usr/lib/x86_64-linux-gnu/libcuda.so -l /usr/lib/x86_64-linux-gnu/libnvrtc.so
  sed -i "s\import ctypes\import ctypes, ctypes.util\g" autogen/cuda.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libcuda.so')\ctypes.CDLL(ctypes.util.find_library('cuda'))\g" autogen/cuda.py
  sed -i "s\ctypes.CDLL('/usr/lib/x86_64-linux-gnu/libnvrtc.so')\ctypes.CDLL(ctypes.util.find_library('nvrtc'))\g" autogen/cuda.py
  grep FIXME_STUB autogen/cuda.py || true
}

if [ "$1" == "opencl" ]; then generate_opencl
elif [ "$1" == "hip" ]; then generate_hip
elif [ "$1" == "cuda" ]; then generate_cuda
else echo "usage: $0 <type>"
fi
