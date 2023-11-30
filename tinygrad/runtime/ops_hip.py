import numpy as np
import ctypes
from typing import Tuple
import gpuctypes.hip as hip
from tinygrad.runtime.ops_cuda import encode_args_cuda_style, time_execution_cuda_style, compile_cuda_style, CUDAGraph
from tinygrad.helpers import DEBUG, getenv, diskcache, ARCWrapper
from tinygrad.device import Compiled
from tinygrad.renderer.hip import HIPRenderer
from tinygrad.runtime.lib import RawBufferCopyInOut, LRUAllocator, RawBufferTransfer, RawBuffer, RawMallocBuffer
from tinygrad.codegen.kernel import LinearizerOptions

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 6:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")
def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)

class HIPAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs):
    check(hip.hipSetDevice(device))
    return (buf := ctypes.c_void_p(), check(hip.hipMalloc(ctypes.byref(buf), size * dtype.itemsize)))[0].value
  def _do_free(self, buf): check(hip.hipFree(buf))
  def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.

MOCKHIP = getenv("MOCKHIP") # for CI. don't run kernels, only check if they compile

class _HIP:
  def __init__(self, device=None):
    self.default_device = device or getenv("HIP_DEFAULT_DEVICE")
    self.device_properties = hip.hipDeviceProp_t()

    if not MOCKHIP:
      check(hip.hipSetDevice(self.default_device))
      check(hip.hipGetDeviceCount(ctypes.byref(dev_count := ctypes.c_int())))
      check(hip.hipGetDeviceProperties(self.device_properties, ctypes.c_int(self.default_device)))

    self.device_count = dev_count.value if not MOCKHIP else 0
    self.allocator = HIPAllocator(self.device_properties.totalGlobalMem) if not MOCKHIP else None
    self.compile_opts = [f'--offload-arch={self.device_properties.gcnArchName.decode("utf-8")}'] if not MOCKHIP else ['--offload-arch=gfx1100']
HIP = _HIP()

class RawHIPBuffer(RawBufferCopyInOut, RawBufferTransfer):
  def __init__(self, size, dtype, device=HIP.default_device, buf=None, allocator=HIP.allocator): super().__init__(size, dtype, buf=buf, allocator=allocator, **{'device': int(device)})
  def _copyin(self, x:np.ndarray):
    check(hip.hipSetDevice(self._device))
    check(hip.hipMemcpyAsync(self._buf, np.require(x, requirements='C').ctypes.data_as(ctypes.c_void_p), self.size * self.dtype.itemsize, hip.hipMemcpyHostToDevice, None))
  def _copyout(self, x:np.ndarray):
    check(hip.hipSetDevice(self._device))
    check(hip.hipMemcpy(x.ctypes.data, self._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToHost))
  def _transfer(self, x:RawBuffer):
    check(hip.hipSetDevice(x._device))
    check(hip.hipMemcpy(self._buf, x._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToDevice))

@diskcache
def compile_hip(prg) -> bytes: return compile_cuda_style(prg, HIP.compile_opts, hip.hiprtcProgram, hip.hiprtcCreateProgram, hip.hiprtcCompileProgram, hip.hiprtcGetCode, hip.hiprtcGetCodeSize, hip.hiprtcGetProgramLog, hip.hiprtcGetProgramLogSize, check)

class HIPProgram:
  def __init__(self, name:str, prg:bytes):
    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    self.modules, self.prgs = [], []
    for i in range(HIP.device_count):
      check(hip.hipSetDevice(i))
      self.modules.append(ARCWrapper((module := hip.hipModule_t(), check(hip.hipModuleLoadData(ctypes.byref(module), prg)))[0], hip.hipModuleUnload))
      self.prgs.append((func := ctypes.POINTER(hip.struct_ihipModuleSymbol_t)(), check(hip.hipModuleGetFunction(ctypes.byref(func), self.modules[-1].obj, name.encode("utf-8"))))[0])

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    if MOCKHIP: return float("inf")
    check(hip.hipSetDevice(args[0]._device))
    return hip_time_execution(lambda: check(hip.hipModuleLaunchKernel(self.prgs[args[0]._device], *global_size, *local_size, 0, None, None, encode_args_cuda_style(args, marks=(1,2,3))[0])), enable=wait)

class HIPGraph(CUDAGraph):
  def launch_params_indicators(self): return (1,2,3)
  def graph_create(self): return ARCWrapper((graph := hip.hipGraph_t(), check(hip.hipGraphCreate(ctypes.byref(graph), 0)))[0], hip.hipGraphDestroy)
  def graph_instantiate(self, graph): return ARCWrapper((instance := hip.hipGraphExec_t(), check(hip.hipGraphInstantiate(ctypes.byref(instance), graph, None, None, 0)))[0], hip.hipGraphExecDestroy)
  def graph_add_kernel_node(self, graph, c_deps, c_params): return (graph_node := hip.hipGraphNode_t(), check(hip.hipGraphAddKernelNode(ctypes.byref(graph_node), graph, c_deps, ctypes.sizeof(c_deps)//8 if c_deps else 0, ctypes.byref(c_params))))[0]
  def graph_launch(self, *args, wait=False): return hip_time_execution(lambda: check(hip.hipGraphLaunch(*args)), enable=wait)
  def graph_exec_kernel_node_set_params(self, *args): return check(hip.hipGraphExecKernelNodeSetParams(*args))

  def build_kernel_node_params(self, ji, prg, global_size, local_size, c_config): return hip.hipKernelNodeParams(hip.dim3(*local_size), c_config, ctypes.cast(prg.clprg.prgs[ji.rawbufs[0]._device], ctypes.c_void_p), hip.dim3(*global_size), None, 0)
  def set_kernel_node_launch_dims(self, node, global_size: Tuple[int, int, int], local_size: Tuple[int, int, int]): node.blockDim.x, node.blockDim.y, node.blockDim.z, node.gridDim.x, node.gridDim.y, node.gridDim.z = *local_size, *global_size

HIPDevice = Compiled(RawHIPBuffer if not MOCKHIP else RawMallocBuffer, LinearizerOptions(device="HIP"), HIPRenderer, compile_hip, HIPProgram, hip.hipDeviceSynchronize, graph=HIPGraph)