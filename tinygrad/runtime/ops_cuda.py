import subprocess, time, re, hashlib, tempfile
from typing import Optional
import numpy as np
from pycuda.compiler import compile as cuda_compile # type: ignore
from tinygrad.helpers import DEBUG, getenv, fromimport, colored
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut, RawMallocBuffer
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])', lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

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
      def record(self): self.start = time.perf_counter()
      def time_till(self, other): return self.start - other.start
      def synchronize(self): pass
    class Context:
      synchronize = lambda:0 # noqa: E731
    CompileError = Exception
  class context:
    class device:
      compute_capability = lambda: (3,5) # pylint: disable=unnecessary-lambda # noqa: E731
    get_device = lambda: context.device # pylint: disable=unnecessary-lambda # noqa: E731
  import pycuda.driver # type: ignore
  pycuda.driver.Context = context
  RawCUDABuffer = RawMallocBuffer
else:
  import pycuda.autoprimaryctx # type: ignore # pylint: disable=unused-import # noqa: F401
  import pycuda.driver as cuda # type: ignore
  class RawCUDABuffer(RawBufferCopyInOut): # type: ignore
    def __init__(self, size, dtype): super().__init__(size, dtype, cuda.mem_alloc(size * dtype.itemsize)) # type: ignore
    def _copyin(self, x:np.ndarray, stream:Optional[cuda.Stream]=None): cuda.memcpy_htod_async(self._buf, x.ravel(), stream) # type: ignore
    def _copyout(self, x:np.ndarray): cuda.memcpy_dtoh(x, self._buf) # type: ignore

class CUDAProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if DEBUG >= 6:
        fn = f"{tempfile.gettempdir()}/tinycuda_{hashlib.md5(prg.encode('utf-8')).hexdigest()}"
        with open(fn, "wb") as f:
          f.write(cuda_compile(prg, target="cubin", no_extern_c=True))
        sass = subprocess.check_output(['nvdisasm', fn]).decode('utf-8')
        print(sass)
      if not binary: prg = cuda_compile(prg, target="ptx", no_extern_c=True, options=['-Wno-deprecated-gpu-targets']).decode('utf-8')
    except cuda.CompileError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 5: print(pretty_ptx(prg))
    # TODO: name is wrong, so we get it from the ptx using hacks
    self.prg = cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0])

  def __call__(self, global_size, local_size, *args, wait=False):
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
    gid = [f'blockIdx.{chr(120+i)}' for i in range(3)],
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
