import subprocess, time, re, hashlib, tempfile
from typing import Optional, List
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
def arch(): return "sm_" + "".join([str(x) for x in pycuda.driver.Context.get_device().compute_capability()])

if getenv("CUDACPU", 0) == 1:
  import ctypes, ctypes.util
  lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  class cuda:
    class device_attribute:
      MAX_GRID_DIM_X = 100
      MAX_GRID_DIM_Y = 100
      MAX_GRID_DIM_Z = 100
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
      get_device = lambda: context.device # pylint: disable=unnecessary-lambda # noqa: E731
    CompileError = Exception
  class context:
    class device:
      compute_capability = lambda: (3,5) # pylint: disable=unnecessary-lambda # noqa: E731
      get_attribute = lambda x: 300*x # pylint: disable=unnecessary-lambda # noqa: E731
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
  def __init__(self, name:str, prg:str, global_size:List[int], local_size:List[int], binary=False):
    dev = cuda.Context.get_device()
    self.max_grid = [dev.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X), dev.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y), dev.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Z)]
    self.global_size = global_size
    self.prg = prg
    self.check_device_limit(prg, global_size)
    if not binary:
      try: prg = cuda_compile(prg, target="ptx", no_extern_c=True, options=['-Wno-deprecated-gpu-targets']).decode('utf-8')
      except cuda.CompileError as e:
        if DEBUG >= 3: print("FAILED TO BUILD", prg)
        raise e
    if DEBUG >= 5: print(pretty_ptx(prg))
    # TODO: name is wrong, so we get it from the ptx using hacks
    if DEBUG >= 6:
      try:
        fn = f"{tempfile.gettempdir()}\\tinycuda_{hashlib.md5(prg.encode('utf-8')).hexdigest()}"
        with open(fn + ".ptx", "wb") as f: f.write(prg.encode('utf-8'))
        subprocess.run(["ptxas", f"-arch={arch()}", "-o", fn, fn+".ptx"], check=True)
        print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
      except Exception as e: print("failed to generate SASS", str(e))
    
  def check_device_limit(self, prg, global_size):
    self.sub_prg = []
    self.sub_global_size = []    

    for j in range(self.global_size[1]//self.max_grid[1]+1):
      prg = self.prg.replace("gidx1", "(gidx1+%d)"%(j*self.max_grid[1])).replace("(gidx1+%d)"%(j*self.max_grid[1]), "gidx1", 1) if j > 0 else self.prg
      if self.global_size[1] > self.max_grid[1]: global_size = [self.global_size[0], self.max_grid[1], self.global_size[2]] if j < self.global_size[1]//self.max_grid[1] else [self.global_size[0], self.global_size[1]%self.max_grid[1], self.global_size[2]]

      for k in range(self.global_size[2]//self.max_grid[2]+1):
        prg = self.prg.replace("gidx0", "(gidx0+%d)"%(k*self.max_grid[2])).replace("(gidx0+%d)"%(k*self.max_grid[2]), "gidx0", 1) if k > 0 else self.prg
        if self.global_size[2] > self.max_grid[2]: global_size = [self.global_size[0], self.global_size[1], self.max_grid[2]] if k < self.global_size[2]//self.max_grid[2] else [self.global_size[0], self.global_size[1], self.global_size[2]%self.max_grid[2]]

        self.sub_prg.append(prg)
        self.sub_global_size.append(global_size)

    print("==========")
    if self.global_size[2] > 65535 or self.global_size[1] > 65535:
      print(*self.sub_global_size, sep='\n')
      print(*self.sub_prg, sep='\n')

    # for gd in range(3):
    #   if global_size[gd] > self.max_grid[gd]:
    #     self.subprg = [self.prg.replace("gidx%d"%(2-gd), "(gidx%d+%d)"%(2-gd, self.max_grid[gd]*i)).replace("(gidx%d+%d)"%(2-gd, self.max_grid[gd]*i), "gidx%d"%(2-gd), 1) if i>0 else self.prg for i in range(global_size[gd]//self.max_grid[gd])]
    #     self.global_size = [tuple([global_size[gid] if gid != gd else self.max_grid[gd] for gid in range(3)]) for i in range(global_size[gd]//self.max_grid[gd])]

    #     self.subprg.append(self.prg.replace("gidx%d"%(2-gd), "(gidx%d+%d)"%(2-gd, self.max_grid[gd]*(global_size[gd]//self.max_grid[gd]))).replace("(gidx%d+%d)"%(2-gd, self.max_grid[gd]*(global_size[gd]//self.max_grid[gd])), "gidx%d"%(2-gd), 1))
    #     self.global_size.append(tuple([global_size[gid] if gid != gd else global_size[gd]%self.max_grid[gd] for gid in range(3)]))

    self.sub_prg = [cuda_compile(prg, target="ptx", no_extern_c=True, options=['-Wno-deprecated-gpu-targets']).decode('utf-8') for prg in self.sub_prg]
    self.sub_prg = [cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0]) for prg in self.sub_prg]

  def __call__(self, global_size, local_size, *args, wait=False):
    print(global_size)
    if wait:
      start, end = cuda.Event(), cuda.Event()
      start.record()
    for prg, gs in zip(self.sub_prg, self.sub_global_size): 
      # if self.global_size[2] > 65535:
      #   print(prg)
      #   print(gs)
      prg(*[x._buf for x in args], block=tuple(local_size), grid=tuple(gs)) # type: ignore
    if wait:
      end.record()
      end.synchronize()
      return start.time_till(end)*1e-3

class CUDACodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "__global__", smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4",
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
