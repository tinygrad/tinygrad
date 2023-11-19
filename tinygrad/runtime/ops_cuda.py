import subprocess, re, hashlib, tempfile, ctypes, ctypes.util
from pathlib import Path
from typing import Tuple
import numpy as np
from extra.cuda_wrapper import cuda_compile, cuda_unwrap, cuda_arch
from pycuda.compiler import compile as pycuda_compile
from tinygrad.helpers import DEBUG, getenv, colored, diskcache
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut, RawMallocBuffer, LRUAllocator
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cuda import CUDARenderer

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
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  RawCUDABuffer = RawMallocBuffer
  compute_capability = (3,5)
  class cuda:
    cuCtxSynchronize = lambda: None
else:
  from cuda import cuda
  cuda_unwrap(cuda.cuInit(0))
  device = cuda_unwrap(cuda.cuDeviceGet(0))
  context = cuda_unwrap(cuda.cuCtxCreate(0, device))
  compute_capability = cuda_arch(device)

  class CUDAAllocator(LRUAllocator):
    def __init__(self): super().__init__(self._get_cur_free_space(None))
    def _do_alloc(self, size, dtype, device, **kwargs): return cuda_unwrap(cuda.cuMemAlloc(size * dtype.itemsize))
    def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.
    def _get_cur_free_space(self, device): return cuda_unwrap(cuda.cuMemGetInfo())[0]
  CUDAAlloc = CUDAAllocator()
  class RawCUDABuffer(RawBufferCopyInOut): # type: ignore
    def __init__(self, size, dtype): super().__init__(size, dtype, allocator=CUDAAlloc)
    def _copyin(self, x:np.ndarray):
      cont_x = np.require(x, requirements='C')
      cuda.cuMemcpyHtoDAsync(self._buf, cont_x.ctypes.data, self.size * self.dtype.itemsize, 0) # type: ignore
    def _copyout(self, x:np.ndarray): cuda.cuMemcpyDtoH(x.ctypes.data, self._buf, self.size * self.dtype.itemsize) # type: ignore

# @diskcache
def compile_cuda(prg) -> bytes: return cuda_compile(prg, compute_capability=compute_capability)

class CUDAProgram:
  def __init__(self, name:str, _prg:bytes):
    prg = _prg.decode('utf-8')
    if DEBUG >= 5: print(pretty_ptx(prg))
    if DEBUG >= 6:
      try:
        fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(prg.encode('utf-8')).hexdigest()}").as_posix()
        with open(fn + ".ptx", "wb") as f: f.write(prg.encode('utf-8'))
        sm_cc = "".join([str(x) for x in compute_capability])
        subprocess.run(["ptxas", f"-arch=sm_{sm_cc}", "-o", fn, fn+".ptx"], check=True)
        print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
      except Exception as e: print("failed to generate SASS", str(e))
    # TODO: name is wrong, so we get it from the ptx using hacks

    if getenv("CUDACPU", 0) == 1:
      self.prg = _prg
    else:
      mod_data = np.char.array(bytes(prg.encode('utf-8')))
      module = cuda_unwrap(cuda.cuModuleLoadData(mod_data.ctypes.data))
      self.prg = cuda_unwrap(cuda.cuModuleGetFunction(module, bytes(prg.split(".visible .entry ")[1].split("(")[0].encode("utf-8"))))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], shared:int=0, wait=False):
    if getenv("CUDACPU", 0) == 1:
      pass_args = [x._buf if isinstance(x, RawCUDABuffer) else np.int32(x) if (isinstance(x, int) and not getenv("CUDACPU")) else x for x in args]
      gpuocelot_lib.ptx_run(self.prg, len(pass_args), (ctypes.c_void_p * len(pass_args))(*[ctypes.cast(x, ctypes.c_void_p) for x in pass_args]), *local_size, *global_size, shared)
      return 0.01

    if wait:
      start, end = cuda_unwrap(cuda.cuEventCreate(0)), cuda_unwrap(cuda.cuEventCreate(0))
      cuda.cuEventRecord(start, 0)
    pass_args = [np.array([int(x._buf)], dtype=np.uint64) if not isinstance(x, int) else np.array([x], dtype=np.int32) for x in args]
    pass_args_ptr = np.array([arg.ctypes.data for arg in pass_args], dtype=np.uint64)
    cuda_unwrap(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, shared, 0, pass_args_ptr.ctypes.data, 0))
    if wait:
      cuda.cuEventRecord(end, 0)
      cuda.cuEventSynchronize(end)
      res = cuda_unwrap(cuda.cuEventElapsedTime(start, end)) * 1e-3
      cuda.cuEventDestroy(start)
      cuda.cuEventDestroy(end)
      return res

CUDABuffer = Compiled(RawCUDABuffer, LinearizerOptions(supports_float4=False if getenv("PTX") else True, supports_float4_alu=False, global_max = [65535, 65535, 2147483647], local_max = [64, 1024, 1024]),
                      CUDARenderer, compile_cuda, CUDAProgram, cuda.cuCtxSynchronize)
