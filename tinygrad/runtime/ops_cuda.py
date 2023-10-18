import subprocess, time, re, hashlib, tempfile, functools
from pathlib import Path
from typing import Optional
import numpy as np
from tinygrad.helpers import DEBUG, getenv, colored, fromimport
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut, RawMallocBuffer, LRUAllocator
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
from functools import lru_cache

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
  class CUDAAllocator(LRUAllocator):
    def _do_alloc(self, size, dtype, device, **kwargs): return cuda.mem_alloc(size * dtype.itemsize) # type: ignore
    def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.
  CUDAAlloc = CUDAAllocator(pycuda.driver.Context.get_device().total_memory())
  class RawCUDABuffer(RawBufferCopyInOut): # type: ignore
    def __init__(self, size, dtype): super().__init__(size, dtype, allocator=CUDAAlloc)
    def _copyin(self, x:np.ndarray, stream:Optional[cuda.Stream]=None): cuda.memcpy_htod_async(self._buf, x.ravel(), stream) # type: ignore
    def _copyout(self, x:np.ndarray): cuda.memcpy_dtoh(x, self._buf) # type: ignore

@lru_cache
def find_cicc_path():
  nvcc_path = subprocess.check_output(f"{'which' if os.name != 'nt' else 'where'} nvcc", shell=True).decode().split()[0]
  
  if os.path.getsize(nvcc_path) < 200: # get path from bin alias
    nvcc_dir = open(nvcc_path, "r").read().split('\n')[2].split(' ')[1][:-4]
  else:
    nvcc_dir = nvcc_path[:-4]

  tmp = nvcc_dir + "cicc"
  if os.path.exists(tmp): return tmp
  tmp = nvcc_dir + "../nvvm/bin/cicc"
  if os.path.exists(tmp): return tmp

  raise EnvironmentError("cicc not found")

CUDA_PROGRAM_HEADER = 'const float INFINITY = __builtin_inff(); const float NAN = __builtin_nanf ("");struct __attribute__((device_builtin)) uint3{unsigned int x, y, z;};struct __attribute__((device_builtin)) __attribute__((aligned(8))) float2 { float x; float y; };struct __attribute__((device_builtin)) __attribute__((aligned(16))) float4{float x, y, z, w;};struct __attribute__((device_builtin)) dim3{unsigned int x, y, z;};static __inline__ __attribute__((host)) __attribute__((device)) float4 make_float4(float x, float y, float z, float w){float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;}extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fminf(float x, float y) noexcept (true);extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) float fmaxf(float x, float y) noexcept (true);extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double sqrt(double x) noexcept (true);extern __attribute__((host)) __attribute__((device)) __attribute__((device_builtin)) double log2(double x) noexcept (true);namespace std __attribute__ ((__visibility__ ("default"))){ inline constexpr float sin(float __x){return __builtin_sinf(__x);}constexpr float exp2(float __x){return __builtin_exp2f(__x);}extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sqrt(float);extern __attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) float sin(float);__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float exp2(float a);__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float log2(float a);__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float fmax(float a, float b);__attribute__((host)) __attribute__((device)) __attribute__((cudart_builtin)) constexpr float fmin(float a, float b);}using std::sin, std::exp2;static inline __attribute__((host)) __attribute__((device)) float min(const float a, const float b){return fminf(a, b);}static inline __attribute__((host)) __attribute__((device)) float max(const float a, const float b){return fmaxf(a, b);}uint3 __attribute__((device_builtin)) extern const threadIdx;uint3 __attribute__((device_builtin)) extern const blockIdx;dim3 __attribute__((device_builtin)) extern const blockDim;dim3 __attribute__((device_builtin)) extern const gridDim;int __attribute__((device_builtin)) extern const warpSize;'

class CUDAProgram:
  def __init__(self, name:str, prg:str, binary=False, shared = 0, local_size_override=None):
    if not binary:
      fn = os.path.join(tempfile.gettempdir(), f"tinycuda_{hashlib.md5(prg.encode('utf-8')).hexdigest()}.ii")
      try: 
        if not os.path.exists(fn+".ptx"):
          with open(fn, 'w+') as f: f.write(CUDA_PROGRAM_HEADER + prg)
          subprocess.run([find_cicc_path(),"-arch",f"compute_{arch()[3:]}","--c++17","--allow_managed", "-m64","-ftz=0", "-prec_div=1", "-prec_sqrt=1", "-fmad=1", "-tused", fn, "-o",fn+'.ptx'], check=True, stderr=subprocess.DEVNULL if DEBUG < 3 else None)
        with open(fn+'.ptx', 'r') as f: prg = f.read()
      except Exception as e:
        if DEBUG >= 3: print("FAILED TO BUILD", prg)
        os.remove(fn)
        raise e
    if DEBUG >= 5: print(pretty_ptx(prg))
    if DEBUG >= 6:
      try:
        fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(prg.encode('utf-8')).hexdigest()}").as_posix()
        with open(fn + ".ptx", "wb") as f: f.write(prg.encode('utf-8'))
        subprocess.run(["ptxas", f"-arch={arch()}", "-o", fn, fn+".ptx"], check=True)
        print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
      except Exception as e: print("failed to generate SASS", str(e))
    # TODO: name is wrong, so we get it from the ptx using hacks
    self.prg, self.shared, self.local_size_override = cuda.module_from_buffer(prg.encode('utf-8')).get_function(prg.split(".visible .entry ")[1].split("(")[0]), shared, local_size_override

  def __call__(self, global_size, local_size, *args, wait=False):
    if wait:
      start, end = cuda.Event(), cuda.Event()
      start.record()
    self.prg(*[x._buf if isinstance(x, RawCUDABuffer) else np.int32(x) if (isinstance(x, int) and not getenv("CUDACPU")) else x for x in args], block=tuple(local_size if self.local_size_override is None else self.local_size_override), grid=tuple(global_size), shared=self.shared)
    if wait:
      end.record()
      end.synchronize()
      return start.time_till(end)*1e-3

renderer = functools.partial(uops_to_cstyle, CStyleLanguage(
  kernel_prefix = "__attribute__((global))", smem_prefix = "__attribute__((shared))", smem_prefix_for_cast=False, arg_int_prefix = "const int", barrier = "__syncthreads();", float4 = "make_float4",
  gid = [f'blockIdx.{chr(120+i)}' for i in range(3)],
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)],
  xid = [f'(blockIdx.{chr(120+i)}*blockDim.{chr(120+i)}+threadIdx.{chr(120+i)})' for i in range(3)],
  half_prekernel = """
    #include <cuda_fp16.h>
    struct __align__(8) half4 {
      half2 x, y;
      __device__ __forceinline__ explicit half4(const float4& a): x(make_half2(__float2half(a.x), __float2half(a.y))), y(make_half2(__float2half(a.z),__float2half(a.w))) {}
      __device__ __forceinline__ explicit operator float4() const {return make_float4(__half2float(x.x), __half2float(x.y), __half2float(y.x), __half2float(y.y)); }
    };
    """)) if not getenv("PTX") else fromimport("tinygrad.renderer.assembly_ptx", "uops_to_ptx_asm")
if getenv("TRITON") == 1:
  from tinygrad.renderer.triton import uops_to_triton
  renderer = uops_to_triton
  CUDABuffer = Compiled(RawCUDABuffer, LinearizerOptions(supports_float4=False, supports_float4_alu=False, global_max = [65535, 65535, 2147483647], local_max = [64, 1024, 1024], has_shared=False), renderer, CUDAProgram, cuda.Context.synchronize)
else:
  CUDABuffer = Compiled(RawCUDABuffer, LinearizerOptions(supports_float4=False if getenv("PTX") else True, supports_float4_alu=False, global_max = [65535, 65535, 2147483647], local_max = [64, 1024, 1024]), renderer, CUDAProgram, cuda.Context.synchronize)
