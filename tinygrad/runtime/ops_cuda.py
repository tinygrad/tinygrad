import subprocess, time, re, hashlib, tempfile, ctypes, ctypes.util
from pathlib import Path
from typing import Tuple, cast, Callable, Any
import numpy as np
import extra.cuda_wrapper as cuda
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
  
  lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  class cuda_class:
    cuDeviceComputeCapability = _ = lambda x: (3,5)
    cuEventRecord = cuEventDestroy = cuModuleLoadData = cuModuleUnload = lambda x: x
    cuModuleGetFunction = _ = lambda x, y: x
    cuLaunchKernel = _ = lambda src, gx, gy, gz, lx, ly, lz, shared, stream, args: lib.ptx_run(src, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), lx, ly, lz, gx, gy, gz, shared)
    cuEventCreate = _ = lambda: time.perf_counter()
    cuEventElapsedTime = _ = lambda x, y: y - x

    nvrtcCreateProgram = cuda.nvrtcCreateProgram
    nvrtcCompileProgram = cuda.nvrtcCompileProgram
    nvrtcGetPTX = cuda.nvrtcGetPTX

    cuCtxSynchronize = _ = lambda: None

  cuda: Any = cuda_class # type: ignore
  RawCUDABuffer = RawMallocBuffer
else:
  cuda.cuInit(0)
  device = cuda.cuDeviceGet(0)
  cuda.cuCtxCreate(0, device)

  class CUDAAllocator(LRUAllocator):
    def __init__(self): super().__init__(self._get_cur_free_space(None))
    def _do_alloc(self, size, dtype, device, **kwargs): return cuda.cuMemAlloc(size * dtype.itemsize)
    def _do_free(self, buf): cuda.cuMemFree(buf)
    def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.
    def _get_cur_free_space(self, device): return cuda.cuMemGetInfo()[0]
  CUDAAlloc = CUDAAllocator()
  class RawCUDABuffer(RawBufferCopyInOut): # type: ignore
    def __init__(self, size, dtype): super().__init__(size, dtype, allocator=CUDAAlloc)
    def _copyin(self, x:np.ndarray): cuda.cuMemcpyHtoDAsync(self._buf, np.require(x, requirements='C').ctypes.data_as(ctypes.c_void_p), self.size * self.dtype.itemsize, 0)
    def _copyout(self, x:np.ndarray): cuda.cuMemcpyDtoH(np.require(x, requirements='C').ctypes.data_as(ctypes.c_void_p), self._buf, self.size * self.dtype.itemsize)

@diskcache
def compile_cuda(prg) -> bytes:
  prog = cuda.nvrtcCreateProgram(prg, "<null>", [], [])
  cuda.nvrtcCompileProgram(prog, ['--gpu-architecture=sm_' + "".join([str(x) for x in cuda.cuDeviceComputeCapability(0)])])
  return cuda.nvrtcGetPTX(prog)

def time_execution(cb, enable=False):
  if enable:
    start, end = cuda.cuEventCreate(), cuda.cuEventCreate()
    cuda.cuEventRecord(start)
  cb()
  if enable:
    cuda.cuEventRecord(end)
    cuda.cuEventSynchronize(end)
    ret = cuda.cuEventElapsedTime(start, end) * 1e-3
    cuda.cuEventDestroy(start)
    cuda.cuEventDestroy(end)
    return ret

class CUDAProgram:
  def __init__(self, name:str, _prg:bytes):
    prg = _prg.decode('utf-8')
    if DEBUG >= 5: print(pretty_ptx(prg))
    if DEBUG >= 6:
      try:
        fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(prg.encode('utf-8')).hexdigest()}").as_posix()
        with open(fn + ".ptx", "wb") as f: f.write(prg.encode('utf-8'))
        subprocess.run(["ptxas", "-arch=sm_"+"".join([str(x) for x in cuda.cuDeviceComputeCapability(0)]), "-o", fn, fn+".ptx"], check=True)
        print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
      except Exception as e: print("failed to generate SASS", str(e))

    self.c_struct_t = None
    self.module = cuda.cuModuleLoadData(prg.encode('utf-8'))
    self.prg = cuda.cuModuleGetFunction(self.module, name)

  def __del__(self):
    cuda.cuModuleUnload(self.module)

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], shared:int=0, wait=False):
    if getenv("CUDACPU", 0) == 1:
      c_params = [x._buf if not isinstance(x, int) else x for x in args]
    else:
      if self.c_struct_t is None: self.c_struct_t = cuda.getCStructForType([(ctypes.c_void_p if not isinstance(x, int) else ctypes.c_int) for x in args])
      c_params = cast(Callable, self.c_struct_t)(*[x._buf if not isinstance(x, int) else x for x in args])

    return time_execution(lambda: cuda.cuLaunchKernel(self.prg, *global_size, *local_size, 0, 0, c_params), enable=wait)

CUDABuffer = Compiled(RawCUDABuffer, LinearizerOptions(supports_float4=False if getenv("PTX") else True, supports_float4_alu=False, global_max = [65535, 65535, 2147483647], local_max = [64, 1024, 1024]),
                      CUDARenderer, compile_cuda, CUDAProgram, cuda.cuCtxSynchronize)
