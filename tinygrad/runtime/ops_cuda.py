from __future__ import annotations
import subprocess, hashlib, tempfile, ctypes, ctypes.util, functools
from pathlib import Path
from typing import Tuple, Optional
import gpuctypes.cuda as cuda
from tinygrad.helpers import DEBUG, getenv, diskcache, from_mv, init_c_var, pretty_ptx, cpu_time_execution, compile_cuda_style, encode_args_cuda_style, time_execution_cuda_style
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import CUDARenderer

CUDACPU = getenv("CUDACPU") == 1
if CUDACPU:
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  cuda.cuLaunchKernel = lambda src, gx, gy, gz, lx, ly, lz, shared, stream, unused_extra, args: gpuocelot_lib.ptx_run(src, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), lx, ly, lz, gx, gy, gz, shared)

def check(status):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))).decode()}")

def cu_time_execution(cb, enable=False) -> Optional[float]: return time_execution_cuda_style(cb, cuda.CUevent, cuda.cuEventCreate, cuda.cuEventRecord, cuda.cuEventSynchronize, cuda.cuEventDestroy_v2, cuda.cuEventElapsedTime, enable=enable) if not CUDACPU else cpu_time_execution(cb, enable=enable)

@diskcache
def compile_cuda(prg) -> bytes: return compile_cuda_style(prg, [f'--gpu-architecture={CUDADevice.default_arch_name}', "-I/usr/local/cuda/include", "-I/usr/include"], cuda.nvrtcProgram, cuda.nvrtcCreateProgram, cuda.nvrtcCompileProgram, cuda.nvrtcGetPTX, cuda.nvrtcGetPTXSize, cuda.nvrtcGetProgramLog, cuda.nvrtcGetProgramLogSize, check)

class CUDAProgram:
  def __init__(self, device:CUDADevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 5: print(pretty_ptx(lib.decode('utf-8')))
    if DEBUG >= 6:
      try:
        fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
        with open(fn + ".ptx", "wb") as f: f.write(lib)
        subprocess.run(["ptxas", f"-arch={CUDADevice.default_arch_name}", "-o", fn, fn+".ptx"], check=True)
        print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
      except Exception as e: print("failed to generate SASS", str(e))

    if not CUDACPU:
      check(cuda.cuCtxSetCurrent(self.device.context))
      self.module = init_c_var(cuda.CUmodule(), lambda x: check(cuda.cuModuleLoadData(ctypes.byref(x), lib)))
      check(cuda.cuModuleGetFunction(ctypes.byref(prg := cuda.CUfunction()), self.module, name.encode("utf-8")))
    self.prg = prg if not CUDACPU else lib

  def __del__(self):
    if not CUDACPU: check(cuda.cuModuleUnload(self.module))

  def __call__(self, *bufs, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], vals:Tuple[int, ...]=(), wait=False):
    if not CUDACPU: check(cuda.cuCtxSetCurrent(self.device.context))
    c_kernel_input_config = encode_args_cuda_style(bufs, vals, cuda.CUdeviceptr_v2, (1,2,0))[0] if not CUDACPU else (bufs+tuple(vals))
    return cu_time_execution(lambda: check(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, c_kernel_input_config)), enable=wait)

class CUDAAllocator(LRUAllocator):
  def __init__(self, device:CUDADevice):
    self.device = device
    super().__init__()
  def _alloc(self, size):
    check(cuda.cuCtxSetCurrent(self.device.context))
    return init_c_var(cuda.CUdeviceptr(), lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), size)))
  def _free(self, opaque): check(cuda.cuMemFree_v2(opaque))
  def copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.device.context))
    check(cuda.cuMemcpyHtoD_v2(dest, from_mv(src), len(src), None))
  def copyout(self, dest:memoryview, src):
    check(cuda.cuCtxSetCurrent(self.device.context))
    check(cuda.cuMemcpyDtoH_v2(from_mv(dest), src, len(dest)))

class CUDADevice(Compiled):
  default_arch_name = "sm_35"
  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    if not CUDACPU:
      check(cuda.cuInit(0))
      check(cuda.cuDeviceGet(ctypes.byref(device := cuda.CUdevice()), device_id))
      check(cuda.cuCtxCreate_v2(ctypes.byref(context := cuda.CUcontext()), 0, device))
      self.context = context
      check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), device_id))
      if device_id == 0: CUDADevice.default_arch_name = f"sm_{major.value}{minor.value}"

    from tinygrad.features.graph.cuda import CUDAGraph
    super().__init__(CUDAAllocator(self) if not CUDACPU else MallocAllocator,
                     LinearizerOptions(supports_float4_alu=False, global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024]),
                     CUDARenderer, compile_cuda, functools.partial(CUDAProgram, self), graph=CUDAGraph if not CUDACPU else None)
  def synchronize(self):
    if not CUDACPU:
      check(cuda.cuCtxSetCurrent(self.context))
      check(cuda.cuCtxSynchronize())
