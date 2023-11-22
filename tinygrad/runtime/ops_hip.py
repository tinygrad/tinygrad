import numpy as np
import ctypes, collections
import extra.hip_wrapper as hip
from typing import Tuple, List, Union, Dict, cast
from tinygrad.helpers import DEBUG, getenv, diskcache
from tinygrad.ops import Compiled, BatchExecutor, JitItem, CompiledASTRunner, update_stats
from tinygrad.renderer.hip import HIPRenderer
from tinygrad.runtime.lib import RawBufferCopyInOut, LRUAllocator, RawBufferTransfer, RawBuffer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.shape.symbolic import Variable, Node

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

def time_exection(cb, enable=False):
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
    self.modules, self.prgs = [], []

    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    for i in range(HIP.device_count):
      hip.hipSetDevice(i)
      self.modules.append(hip.hipModuleLoadData(prg))
      self.prgs.append(hip.hipModuleGetFunction(self.modules[-1], name))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    hip.hipSetDevice(args[0]._device)
    struct = hip.getStructTypeForArgs(*args)(*[data._buf if not isinstance(data, int) else np.int32(data) for data in args])
    return time_exection(lambda: hip.hipModuleLaunchKernel(self.prgs[args[0]._device], global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct), enable=wait)

  def __del__(self):
    for module in self.modules: hip.hipModuleUnload(module)

class HIPGraph(BatchExecutor):
  UpdateEntry = collections.namedtuple("UpdateEntry", ["node", "params"])

  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: Dict[Union[int, str], RawBuffer], var_vals: Dict[Variable, int]):
    super().__init__(jit_cache, input_rawbuffers, var_vals)

    # TODO: Only HIPProgram can be captured for now.
    if not all(isinstance(ji.prg, CompiledASTRunner) and isinstance(ji.prg.clprg, HIPProgram) for ji in self.jit_cache): return

    self.updatable_var_vals: List[int] = []
    self.updatable_launch_dims: List[int] = []
    self.all_updatable_entries: Dict[int, HIPGraph.UpdateEntry] = {}
    self.graph, graph_node = hip.hipGraphCreate(), None

    for (j,i),input_name in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_name]
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledASTRunner = cast(CompiledASTRunner, ji.prg)
      global_size, local_size = prg.launch_dims(var_vals)

      assert all(x is not None for x in ji.rawbufs) and ji.rawbufs[0] is not None, "buffers could not be None" # for linters
      assert prg.global_size and prg.local_size, "need global and local size to JIT"

      params = hip.buildKernelNodeParams(*ji.rawbufs, *[var_vals[x] for x in prg.vars], func=prg.clprg.prgs[ji.rawbufs[0]._device], grid=global_size, block=local_size)
      graph_node = hip.hipGraphAddKernelNode(self.graph, [graph_node] if graph_node else [], params)

      # Record info needed to update jit cache entry.
      if prg.vars:
        self.updatable_var_vals.append(j)
      if any(isinstance(x, Node) for x in prg.global_size) or any(isinstance(x, Node) for x in prg.local_size):
        self.updatable_launch_dims.append(j)
      if j in self.updatable_launch_dims or j in self.updatable_var_vals or j in [x[0] for x in self.input_replace.keys()]:
        self.all_updatable_entries[j] = HIPGraph.UpdateEntry(graph_node, params)

    self.instance = hip.hipGraphInstantiate(self.graph)
    self.clear_jit_inputs()

  def __del__(self):
    if hasattr(self, 'instance'): hip.hipGraphExecDestroy(self.instance)
    if hasattr(self, 'graph'): hip.hipGraphDestroy(self.graph)

  def __call__(self, input_rawbuffers: Dict[Union[int, str], RawBuffer], var_vals: Dict[Variable, int], wait=False):
    if not hasattr(self, 'instance'): return super().__call__(input_rawbuffers, var_vals, wait)

    # Update cached params structs with the new values.
    for (j,i),input_name in self.input_replace.items():
      hip.setKernelNodeParam(self.all_updatable_entries[j].params, input_rawbuffers[input_name], i)
    for j in self.updatable_launch_dims:
      hip.setKernelNodeLaunchDims(self.all_updatable_entries[j].params, cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals))
    for j in self.updatable_var_vals:
      hip.setKernelNodeParams(self.all_updatable_entries[j].params, [var_vals[x] for x in self.jit_cache[j].prg.vars], list(range(len(self.jit_cache[j].rawbufs), len(self.jit_cache[j].rawbufs) + len(self.jit_cache[j].prg.vars))))

    # Update graph nodes with the updated structs.
    for j in self.all_updatable_entries:
      hip.hipGraphExecKernelNodeSetParams(self.instance, self.all_updatable_entries[j].node, self.all_updatable_entries[j].params)

    et = time_exection(lambda: hip.hipGraphLaunch(self.instance), enable=wait)
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers), jit=True, num_kernels=len(self.jit_cache))
    return et

HIPBuffer = Compiled(RawHIPBuffer, LinearizerOptions(device="HIP"), HIPRenderer, compile_hip, HIPProgram, hip.hipDeviceSynchronize, graph=HIPGraph)