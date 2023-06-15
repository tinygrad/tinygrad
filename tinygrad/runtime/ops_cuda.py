import subprocess
from typing import Optional
import numpy as np
from tinygrad.helpers import DEBUG, getenv
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut, RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage
from tinygrad.codegen.assembly_ptx import PTXCodegen
from pycuda.compiler import compile as cuda_compile # type: ignore

EMULATING = (getenv("CUDACPU", 0) == 1)
if EMULATING:
  import ctypes, ctypes.util
  lib = ctypes.CDLL(ctypes.util.find_library("cudacpu"))
  lib.ptx_kernel_create.argtypes = [ctypes.c_char_p]
  lib.ptx_kernel_create.restype = ctypes.c_void_p
  lib.ptx_kernel_destroy.argtypes = [ctypes.c_void_p]
  lib.ptx_call.argtypes = [ctypes.c_void_p,  ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  class PTXKernel:
    def __init__(self, source: bytes): self.kernel = lib.ptx_kernel_create(ctypes.c_char_p(source))
    def __call__(self, *args, block, grid): lib.ptx_call(self.kernel, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), *block, *grid)
    def __del__(self): lib.ptx_kernel_destroy(self.kernel)
else:
  import pycuda.autoprimaryctx # type: ignore # pylint: disable=unused-import # noqa: F401
  import pycuda.driver as cuda # type: ignore
  class RawCUDABuffer(RawBufferCopyInOut):
    def __init__(self, size, dtype): super().__init__(size, dtype, cuda.mem_alloc(size * dtype.itemsize))
    def _copyin(self, x:np.ndarray, stream:Optional[cuda.Stream]=None): cuda.memcpy_htod_async(self._buf, x, stream)
    def _copyout(self, x:np.ndarray): cuda.memcpy_dtoh(x, self._buf)

class CUDAProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if DEBUG >= 6:
        with open("/tmp/cubin", "wb") as f:
          f.write(cuda_compile(prg, target="cubin", no_extern_c=True))
        sass = subprocess.check_output(['nvdisasm', '/tmp/cubin']).decode('utf-8')
        print(sass)
      if not binary or EMULATING:
        prg = cuda_compile(prg, target="ptx", no_extern_c=True, arch=("sm_35" if EMULATING else None), options=["-Wno-deprecated-gpu-targets"]).decode('utf-8')
    except Exception as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5: print(prg)
    # TODO: name is wrong, so we get it from the ptx using hacks
    self.prg = cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0]) if not EMULATING else PTXKernel(prg.encode('utf-8'))

  def __call__(self, global_size, local_size, *args, wait=False):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]

    if wait and not EMULATING:
      start, end = cuda.Event(), cuda.Event()
      start.record()
    self.prg(*[x._buf for x in args], block=tuple(local_size), grid=tuple(global_size))
    if wait and not EMULATING:
      end.record()
      end.synchronize()
      return start.time_till(end)*1e-3

class CUDACodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "__global__", smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
    gid = [f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)],
    half_prekernel = """
      #include <cuda_fp16.h>
      struct __align__(8) half4 {
        half2 x, y;
        __device__ __forceinline__ explicit operator float4() const {return make_float4(__half2float(x.x), __half2float(x.y), __half2float(y.x), __half2float(y.y)); }
      };
      typedef unsigned char uchar;
      typedef long long int64;
    """)
  supports_float4_alu = False

CUDABuffer = Compiled(RawMallocBuffer, CUDACodegen, CUDAProgram) if EMULATING else Compiled(RawCUDABuffer, PTXCodegen if getenv("PTX") else CUDACodegen, CUDAProgram, cuda.Context.synchronize)
