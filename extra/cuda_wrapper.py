import subprocess
from cuda import nvrtc, cuda, cudart

def cuda_get_include_paths(compiler):
  try:
    # Run the compiler command to get the include paths
    result = subprocess.check_output([compiler, "-E", "-x", "c", "-", "-v"], input="", stderr=subprocess.STDOUT, universal_newlines=True)
    lines = result.splitlines()

    includes = []
    for line in lines:
      if line.startswith("#$ INCLUDES="):
        line = line.strip().rstrip()[len("#$ INCLUDES=\""):-1]
        includes = line.split()
    return includes

  except Exception as e:
    print(f"An error occurred, CUDA might be unavailable: {e}")
    return []

cuda_includes = cuda_get_include_paths(compiler="nvcc")

def cuda_unwrap(x):
  assert x[0] == nvrtc.nvrtcResult.NVRTC_SUCCESS, str(x[0])
  return x[1] if len(x[1:]) == 1 else tuple(x[1:])

def cuda_arch(device):
  x = cuda_unwrap(cuda.cuDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device))
  y = cuda_unwrap(cuda.cuDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device))
  return (x,y)

def cuda_compile(prg, compute_capability=(3,5)):
  err, prog = nvrtc.nvrtcCreateProgram(str.encode(prg), b"<null>", 0, [], [])

  sm_cc = "".join([str(x) for x in compute_capability])
  options = [x.encode("ascii") for x in cuda_includes]
  options += [b"-split-compile=0", f"--gpu-architecture=sm_{sm_cc}".encode("ascii")]

  err, = nvrtc.nvrtcCompileProgram(prog, len(options), options)

  if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
    logSize = cuda_unwrap(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * logSize
    err, = nvrtc.nvrtcGetProgramLog(prog, log)
    print(log.decode('utf-8'))
    raise RuntimeError

  # Get PTX from compilation
  ptx_size = cuda_unwrap(nvrtc.nvrtcGetPTXSize(prog))
  ptx = b" " * ptx_size
  err, = nvrtc.nvrtcGetPTX(prog, ptx)

  return ptx[:-1]