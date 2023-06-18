import subprocess
from typing import Optional
from time import perf_counter
import numpy as np
from pycuda.compiler import compile as cuda_compile # type: ignore
from tinygrad.helpers import DEBUG, getenv, fromimport
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut, RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

if getenv("CUDACPU", 0) == 1:
  import ctypes, ctypes.util
  lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  class cuda:
    class module:
      def __init__(self, src): self.src = src
      def get_function(self, _): return self
      def __call__(self, *args, block, grid): lib.ptx_run(self.src, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), *block, *grid)
    module_from_buffer = lambda src: cuda.module(src) # pylint: disable=unnecessary-lambda # noqa: E731
    class Event:
      def __init__(self): pass
      def record(self): self.start = perf_counter()
      def time_till(self, other): return self.start - other.start
      def synchronize(self): pass
    class Context:
      synchronize = lambda:0 # noqa: E731
    CompileError = Exception
  __cuda_compile = cuda_compile
  def _cuda_compile(prg, **kwargs): return __cuda_compile(prg, **{**kwargs, 'arch': 'sm_35', 'options': kwargs.get('options', []) + ['-Wno-deprecated-gpu-targets']}) # noqa: E731
  cuda_compile = _cuda_compile # noqa: F811
  RawCUDABuffer = RawMallocBuffer
else:
  import pycuda.autoprimaryctx # type: ignore # pylint: disable=unused-import # noqa: F401
  import pycuda.driver as cuda # type: ignore
  class RawCUDABuffer(RawBufferCopyInOut): # type: ignore
    def __init__(self, size, dtype): super().__init__(size, dtype, cuda.mem_alloc(size * dtype.itemsize)) # type: ignore
    def _copyin(self, x:np.ndarray, stream:Optional[cuda.Stream]=None): cuda.memcpy_htod_async(self._buf, x, stream) # type: ignore
    def _copyout(self, x:np.ndarray): cuda.memcpy_dtoh(x, self._buf) # type: ignore

class CUDAProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if DEBUG >= 6:
        with open("/tmp/cubin", "wb") as f:
          f.write(cuda_compile(prg, target="cubin", no_extern_c=True))
        sass = subprocess.check_output(['nvdisasm', '/tmp/cubin']).decode('utf-8')
        print(sass)
      if not binary: prg = cuda_compile(prg, target="ptx", no_extern_c=True).decode('utf-8')
    except cuda.CompileError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5: print(prg)
    # TODO: name is wrong, so we get it from the ptx using hacks
    self.prg = cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0])

  def __call__(self, global_size, local_size, *args, wait=False):
    local_size = (local_size + [1] * (3 - len(local_size))) if local_size is not None else (1,1,1)
    global_size = global_size + [1] * (3 - len(global_size))
    assert all(x%y == 0 for x,y in zip(global_size, local_size)), f"local:{local_size} must divide global:{global_size}"
    global_size = [x//y for x,y in zip(global_size, local_size)]
    if wait:
      start, end = cuda.Event(), cuda.Event()
      start.record()
    self.prg(*[x._buf for x in args], block=tuple(local_size), grid=tuple(global_size))
    if wait:
      end.record()
      end.synchronize()
      return start.time_till(end)*1e-3

class CUDACodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "typedef unsigned char uchar;\ntypedef unsigned int uint;\ntypedef unsigned long ulong;\n__global__", smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
    gid = [f'blockDim.{chr(120+i)}*blockIdx.{chr(120+i)}+threadIdx.{chr(120+i)}' for i in range(3)],
    lid = [f'threadIdx.{chr(120+i)}' for i in range(3)],
    half_prekernel = """
      #include <cuda_fp16.h>
      struct __align__(8) half4 {
        half2 x, y;
        __device__ __forceinline__ explicit operator float4() const {return make_float4(__half2float(x.x), __half2float(x.y), __half2float(y.x), __half2float(y.y)); }
      };
    """)
  supports_float4_alu = False

CUDABuffer = Compiled(RawCUDABuffer, fromimport("tinygrad.codegen.assembly_ptx", "PTXCodegen") if getenv("PTX") else CUDACodegen, CUDAProgram, cuda.Context.synchronize)
