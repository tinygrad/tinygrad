import numpy as np
import ctypes
import extra.hip_wrapper as hip
from typing import Tuple, List, Any, Dict, cast, Optional, Callable
from tinygrad.helpers import DEBUG, getenv, diskcache
from tinygrad.ops import Compiled, CompiledASTRunner, update_stats
from tinygrad.renderer.hip import HIPRenderer
from tinygrad.runtime.lib import RawBufferCopyInOut, LRUAllocator, RawBufferTransfer, RawBuffer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.shape.symbolic import Variable
from tinygrad.jit import JitItem, get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals, GraphException

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 6:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.

class HIPAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs):
    hip.hipSetDevice(device)
    return hip.hipMalloc(size * dtype.itemsize)
  def _do_free(self, buf): hip.hipFree(buf)
  def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.

class _HIP:
  def __init__(self, device=None):
    self.default_device = device or getenv("HIP_DEFAULT_DEVICE")
    hip.hipSetDevice(self.default_device)
    self.device_count = hip.hipGetDeviceCount()
    self.allocator = HIPAllocator(hip.hipGetDeviceProperties(self.default_device).totalGlobalMem)
HIP = _HIP()

class RawHIPBuffer(RawBufferCopyInOut, RawBufferTransfer):
  def __init__(self, size, dtype, device=HIP.default_device, buf=None, allocator=HIP.allocator): super().__init__(size, dtype, buf=buf, allocator=allocator, **{'device': int(device)})
  def _copyin(self, x:np.ndarray):
    hip.hipSetDevice(self._device)
    hip.hipMemcpyAsync(self._buf, np.require(x, requirements='C').ctypes.data_as(ctypes.c_void_p), self.size * self.dtype.itemsize, hip.hipMemcpyHostToDevice, 0)
  def _copyout(self, x:np.ndarray):
    hip.hipSetDevice(self._device)
    hip.hipMemcpy(x.ctypes.data, self._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToHost)
  def _transfer(self, x:RawBuffer):
    hip.hipSetDevice(x._device)
    hip.hipMemcpy(self._buf, x._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToDevice)

@diskcache
def compile_hip(prg) -> bytes:
  prog = hip.hiprtcCreateProgram(prg, "<null>", [], [])
  hip.hiprtcCompileProgram(prog, [f'--offload-arch={hip.hipGetDeviceProperties(HIP.default_device).gcnArchName}'])
  return hip.hiprtcGetCode(prog)

def time_execution(cb, enable=False):
  if enable:
    start, end = hip.hipEventCreate(), hip.hipEventCreate()
    hip.hipEventRecord(start)
  cb()
  if enable:
    hip.hipEventRecord(end)
    hip.hipEventSynchronize(end)
    ret = hip.hipEventElapsedTime(start, end)*1e-3
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(end)
    return ret

class HIPProgram:
  def __init__(self, name:str, prg:bytes):
    self.modules, self.prgs, self.c_struct_t = [], [], None

    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    for i in range(HIP.device_count):
      hip.hipSetDevice(i)
      self.modules.append(hip.hipModuleLoadData(prg))
      self.prgs.append(hip.hipModuleGetFunction(self.modules[-1], name))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    hip.hipSetDevice(args[0]._device)
    if self.c_struct_t is None: self.c_struct_t = hip.getCStructForType([(ctypes.c_void_p if not isinstance(x, int) else ctypes.c_int) for x in args])
    c_params = cast(Callable, self.c_struct_t)(*[x._buf if not isinstance(x, int) else x for x in args])
    return time_execution(lambda: hip.hipModuleLaunchKernel(self.prgs[args[0]._device], *global_size, *local_size, 0, 0, c_params), enable=wait)

  def __del__(self):
    for module in self.modules: hip.hipModuleUnload(module)

class HIPGraph:
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[RawBuffer], var_vals: Dict[Variable, int]):
    # TODO: Only HIPProgram can be captured for now.
    if not all(isinstance(ji.prg, CompiledASTRunner) and isinstance(ji.prg.clprg, HIPProgram) for ji in jit_cache): raise GraphException

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)
    self.jc_idxs_with_updatable_rawbufs = list(set([x[0] for x in self.input_replace.keys()]))

    self.graph, graph_node = hip.hipGraphCreate(), None
    self.updatable_nodes: Dict[int, Tuple[Any, hip.kernelNodeParamsWrapper]] = {} # Dict[jc index] = tuple(graph_node, node_params)

    for (j,i),input_name in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_name]
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledASTRunner = cast(CompiledASTRunner, ji.prg)
      assert all(x is not None for x in ji.rawbufs) and ji.rawbufs[0] is not None, "buffers could not be None" # for linters

      args = [cast(RawBuffer, x)._buf for x in ji.rawbufs] + [var_vals[x] for x in prg.vars]
      types = [ctypes.c_void_p] * len(ji.rawbufs) + [ctypes.c_int] * len(prg.vars)
      c_params = hip.buildKernelNodeParams(args, types, prg.clprg.prgs[ji.rawbufs[0]._device], *prg.launch_dims(var_vals))
      graph_node = hip.hipGraphAddKernelNode(self.graph, [graph_node] if graph_node else [], c_params)

      if j in self.jc_idxs_with_updatable_launch_dims or j in self.jc_idxs_with_updatable_var_vals or j in self.jc_idxs_with_updatable_rawbufs:
        self.updatable_nodes[j] = (graph_node, c_params)

    self.instance = hip.hipGraphInstantiate(self.graph)

  def __del__(self):
    hip.hipGraphExecDestroy(self.instance)
    hip.hipGraphDestroy(self.graph)

  def __call__(self, input_rawbuffers: List[RawBuffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # Update cached params structs with the new values.
    for (j,i),input_idx in self.input_replace.items():
      hip.setKernelNodeParams(self.updatable_nodes[j][1], [input_rawbuffers[input_idx]._buf], [i])
    for j in self.jc_idxs_with_updatable_launch_dims:
      hip.setKernelNodeLaunchDims(self.updatable_nodes[j][1], *cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals))
    for j in self.jc_idxs_with_updatable_var_vals:
      prg: CompiledASTRunner = cast(CompiledASTRunner, self.jit_cache[j].prg)
      hip.setKernelNodeParams(self.updatable_nodes[j][1], [var_vals[x] for x in prg.vars], list(range(len(self.jit_cache[j].rawbufs), len(self.jit_cache[j].rawbufs) + len(prg.vars))))

    # Update graph nodes with the updated structs.
    for node, params in self.updatable_nodes.values():
      hip.hipGraphExecKernelNodeSetParams(self.instance, node, params)

    et = time_execution(lambda: hip.hipGraphLaunch(self.instance), enable=wait)
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers), jit=jit, num_kernels=len(self.jit_cache))
    return et

HIPBuffer = Compiled(RawHIPBuffer, LinearizerOptions(device="HIP"), HIPRenderer, compile_hip, HIPProgram, hip.hipDeviceSynchronize, graph=HIPGraph)
