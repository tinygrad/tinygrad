import subprocess, time, hashlib, tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
from pycuda.compiler import compile as cuda_compile
from tinygrad.helpers import DEBUG, getenv, pretty_ptx, diskcache
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cuda import CUDARenderer

def arch(): return "sm_" + "".join([str(x) for x in pycuda.driver.Context.get_device().compute_capability()])
CUDACPU = getenv("CUDACPU") == 1

if CUDACPU:
  import ctypes, ctypes.util
  lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  class cuda:
    class module:
      def __init__(self, src): self.src = src
      def get_function(self, _): return self
      def __call__(self, *args, block, grid, shared): lib.ptx_run(self.src, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), *block, *grid, shared)
    module_from_buffer = lambda src: cuda.module(src) # pylint: disable=unnecessary-lambda # noqa: E731
    class Event:
      def __init__(self): pass
      def record(self): self.start = time.perf_counter()
      def time_till(self, other): return other.start - self.start
      def synchronize(self): pass
    class Context:
      synchronize = lambda:0 # noqa: E731
    CompileError = Exception
  class context:
    class device:
      compute_capability = lambda: (3,5) # pylint: disable=unnecessary-lambda # noqa: E731
    get_device = lambda: context.device # pylint: disable=unnecessary-lambda # noqa: E731
  import pycuda.driver
  pycuda.driver.Context = context
else:
  import pycuda.autoprimaryctx # pylint: disable=unused-import # noqa: F401
  import pycuda.driver as cuda # type: ignore
  class CUDAAllocator(LRUAllocator):
    def _alloc(self, size, dtype):
      if size == 0: return None
      return cuda.mem_alloc(size * dtype.itemsize) # type: ignore
    def copyin(self, dest, src:memoryview): cuda.memcpy_htod_async(dest, src) # type: ignore
    def copyout(self, dest:memoryview, src): cuda.memcpy_dtoh(dest, src) # type: ignore

@diskcache
def compile_cuda(prg) -> bytes: return cuda_compile(prg, target="ptx", no_extern_c=True, options=['-Wno-deprecated-gpu-targets'])

class CUDAProgram:
  def __init__(self, name:str, _prg:bytes, bufs:int, vars:int=0):
    prg = _prg.decode('utf-8')
    if DEBUG >= 5: print(pretty_ptx(prg))
    if DEBUG >= 6:
      try:
        fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(prg.encode('utf-8')).hexdigest()}").as_posix()
        with open(fn + ".ptx", "wb") as f: f.write(prg.encode('utf-8'))
        subprocess.run(["ptxas", f"-arch={arch()}", "-o", fn, fn+".ptx"], check=True)
        print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
      except Exception as e: print("failed to generate SASS", str(e))
    # TODO: name is wrong, so we get it from the ptx using hacks
    self.prg = cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0])

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], shared:int=0, wait=False):
    if wait:
      start, end = cuda.Event(), cuda.Event()
      start.record()
    self.prg(*[np.int32(x) if (isinstance(x, int) and not CUDACPU) else x for x in args], block=tuple(local_size), grid=tuple(global_size), shared=shared)
    if wait:
      end.record()
      end.synchronize()
      return start.time_till(end)*1e-3

class CUDADevice(Compiled):
  def __init__(self, device:str):
    super().__init__(MallocAllocator if CUDACPU else CUDAAllocator(),
                     LinearizerOptions(supports_float4_alu=False, global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024]),
                     CUDARenderer, compile_cuda, CUDAProgram)
  def synchronize(self): return cuda.Context.synchronize()
