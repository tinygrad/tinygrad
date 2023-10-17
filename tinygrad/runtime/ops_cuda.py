import subprocess, time, re, hashlib, tempfile, functools
from pathlib import Path
from typing import Optional
import numpy as np
from pycuda.compiler import compile as cuda_compile # type: ignore
from tinygrad.helpers import DEBUG, getenv, colored, fromimport
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut, RawMallocBuffer, LRUAllocator
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

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

# TODO: Unify with HIP graph. They are really the same.
class CUDAGraph(BasicBatchExecutor):
  def __init__(self, jit_cache: List[Tuple[Any, Any, Any]]):
    self.info, self.graphs, self.instances = [], [], []

    # Check if CUDAGraph could run the given jit_cache. If not, no hip graph is created and HIPGraph is a BasicBatchExecutor.
    if DEBUG>0 or not all(isinstance(prg, ASTRunner) and isinstance(prg.clprg, CUDAProgram) for prg,_,_ in jit_cache): return # Only CUDAProgram can be captured.
    if len(set([pargs[0]._device for _,pargs,_ in jit_cache])) != 1: return # Only one device is supported now.

    # Splitting the JIT cache into batches to enable parallel execution (cpu+gpu). Batch sizes follow a logarithmic pattern: 4, 8, 16, 32, and so on.
    # This helps push tasks to the GPU while the CPU updates the next graph.
    capture_stream = cuda.Stream()
    capture_stream.begin_capture()
    for j,(prg, pargs, variables) in enumerate(jit_cache):
      # Capture node
      global_size, local_size = prg.launch_dims(variables)
      _, _, graph, deps = capture_stream.get_capture_info_v2()
      cuda_args = [x._buf if isinstance(x, RawCUDABuffer) else np.int32(x) if (isinstance(x, int) and not getenv("CUDACPU")) else x for x in [*pargs, *variables.values()]]
      graph_node = graph.add_kernel_node(*cuda_args, block=tuple(local_size), grid=tuple(global_size), func=prg.clprg.prg, dependencies=deps)
      capture_stream.update_capture_dependencies([graph_node], 1)
      self.info.append((self.__get_batch(j), graph_node, params))

      # If the next batch is different or this is the last entry, finish the graph.
      if self.__get_batch(j) != self.__get_batch(j+1) or j==len(jit_cache)-1:
        self.graphs.append(capture_stream.end_capture())
        self.instances.append(self.graphs[-1].instantiate())
        if j!=len(jit_cache)-1: capture_stream.begin_capture()
  def __update(self, nodeid, inst, prg, pargs, variables, updated_args=None):
    batchid, graph_node, params = self.info[nodeid]
    global_size, local_size = prg.launch_dims(variables)
    cuda_args = [x._buf if isinstance(x, RawCUDABuffer) else np.int32(x) if (isinstance(x, int) and not getenv("CUDACPU")) else x for x in [*pargs, *variables.values()]]
    inst.kernel_node_set_params(*cuda_args, block=tuple(local_size), grid=tuple(global_size), func=prg.clprg.prg, kernel_node=graph_node)
    self.info[nodeid] = (batchid, graph_node, params)

  def exec(self, jit_cache: List[Tuple[Any, Any, Any]], updatable_entries):
    if not self.instances: return super().exec(jit_cache, updatable_entries) # No graph is created switch to basic executor.
    update_keys_per_batch = collections.defaultdict(list)
    for j in updatable_entries.keys(): update_keys_per_batch[self.info[j][0]].append(j)
    for i,inst in enumerate(self.instances):
      for j in update_keys_per_batch[i]: self.__update(j, inst, jit_cache[j][0], jit_cache[j][1], jit_cache[j][2], updated_args=updatable_entries[j])
      inst.launch()
    super().recalc_stat(jit_cache)
  def __get_batch(self, j): return int(math.log(j+4,2)-2) # Batch sizes are logarithmic 4,8,16,32,...

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

class CUDAProgram:
  def __init__(self, name:str, prg:str, binary=False, shared = 0, local_size_override=None):
    if not binary:
      try: prg = cuda_compile(prg, target="ptx", no_extern_c=True, options=['-Wno-deprecated-gpu-targets']).decode('utf-8')
      except cuda.CompileError as e:
        if DEBUG >= 3: print("FAILED TO BUILD", prg)
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
  kernel_prefix = "__global__ ", smem_prefix = "__shared__ ", smem_prefix_for_cast=False, arg_int_prefix = "const int", barrier = "__syncthreads();", float4 = "make_float4",
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
