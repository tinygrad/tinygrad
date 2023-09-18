import numpy as np
import ctypes, functools
import extra.hip_wrapper as hip
from typing import Tuple, Any, List
from tinygrad.helpers import DEBUG, getenv
from tinygrad.ops import Compiled, BatchExecutor
from tinygrad.runtime.lib import RawBufferCopyInOut, LRUAllocator, RawBufferTransfer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

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
  def __init__(self):
    self.device_count = hip.hipGetDeviceCount()
    self.default_device = getenv("HIP_DEFAULT_DEVICE")
    self.allocator = HIPAllocator(hip.hipGetDeviceProperties(self.default_device).totalGlobalMem)
HIP = _HIP()

class HIPGraph(BatchExecutor):
  def __init__(self):
    self.info: List[Tuple[Any, ...]] = []
    self.capture_stream, self.last_devid, self.graph, self.instance, self.failed = None, None, None, None, False

  def __del__(self):
    if self.capture_stream: hip.hipStreamDestroy(self.capture_stream)
    if self.instance: hip.hipGraphExecDestroy(self.instance)
    if self.graph: hip.hipGraphDestroy(self.graph)

  def capture(self, prg, global_size, local_size, *args) -> int: # Returns nodeid
    devid = args[0]._device
    if self.capture_stream is None:
      self.capture_stream = hip.hipStreamCreate()
      hip.hipStreamBeginCapture(self.capture_stream)

    # TODO: Support multidevice graph. Need to add manual sync between graphs and copy support (_send_rb, _recv_rb)
    if self.last_devid is not None and devid != self.last_devid: self.failed = True
    self.last_devid = devid

    # Adding new node to graph
    hip.hipSetDevice(devid)
    _, _, graph, deps = hip.hipStreamGetCaptureInfo_v2(self.capture_stream)
    struct_type_cached = hip.getStructTypeForArgs(*args)
    params = hip.buildKernelNodeParams(*args, func=prg.prgs[devid], grid=global_size, block=local_size)
    graph_node = hip.hipGraphAddKernelNode(graph, deps, params)
    hip.hipStreamUpdateCaptureDependencies(self.capture_stream, [graph_node], 1)

    self.info.append((prg.prgs[devid], graph_node, struct_type_cached))
    return len(self.info) - 1
  def instantiate(self) -> bool: # Returns True if successful
    assert self.last_devid is not None, "Nothing captured?"
    self.graph = hip.hipStreamEndCapture(self.capture_stream) # Only one graph is supported now
    self.instance = hip.hipGraphInstantiate(self.graph)
    return not self.failed
  def update(self, nodeid, global_size, local_size, *args):
    prg, graph_node, struct_type_cached = self.info[nodeid]
    params = hip.buildKernelNodeParams(*args, func=prg, grid=global_size, block=local_size, argsStructType=struct_type_cached)
    hip.hipGraphExecKernelNodeSetParams(self.instance, graph_node, params)
  def exec(self): hip.hipGraphLaunch(self.instance)

class RawHIPBuffer(RawBufferCopyInOut, RawBufferTransfer):
  def __init__(self, size, dtype, device=str(HIP.default_device)): super().__init__(size, dtype, allocator=HIP.allocator, **{'device': int(device)})
  def _copyin(self, x:np.ndarray):
    x = np.require(x, requirements='C')
    hip.hipMemcpyAsync(self._buf, x.ctypes.data, self.size * self.dtype.itemsize, hip.hipMemcpyHostToDevice, 0)
  def _copyout(self, x:np.ndarray): hip.hipMemcpy(x.ctypes.data, self._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToHost)
  def _transfer(self, x): hip.hipMemcpyAsync(self._buf, x._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToDevice, 0)

class HIPProgram:
  def __init__(self, name:str, prg:str, binary=False):
    try:
      if not binary:
        prog = hip.hiprtcCreateProgram(prg, name, [], [])
        device_properties = hip.hipGetDeviceProperties(HIP.default_device)
        hip.hiprtcCompileProgram(prog, [f'--offload-arch={device_properties.gcnArchName}'])
        prg = hip.hiprtcGetCode(prog)
    except Exception as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    self.prgs = []
    for i in range(HIP.device_count):
      hip.hipSetDevice(i)
      self.prgs.append(hip.hipModuleGetFunction(hip.hipModuleLoadData(prg), name))

  def __call__(self, global_size, local_size, *args, wait=False):
    hip.hipSetDevice(args[0]._device)
    if wait:
      start, end = hip.hipEventCreate(), hip.hipEventCreate()
      hip.hipEventRecord(start)
    class PackageStruct(ctypes.Structure):
      _fields_ = [(f'field{idx}', ctypes.c_void_p if not isinstance(args[idx], int) else ctypes.c_int) for idx in range(len(args))]
    struct = PackageStruct(*[data._buf if not isinstance(data, int) else np.int32(data) for data in args])
    hip.hipModuleLaunchKernel(self.prgs[args[0]._device], global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct)
    if wait:
      hip.hipEventRecord(end)
      hip.hipEventSynchronize(end)
      return hip.hipEventElapsedTime(start, end)*1e-3

renderer = functools.partial(uops_to_cstyle, CStyleLanguage(
  kernel_prefix = "#include <hip/hip_common.h>\n#define INFINITY (__builtin_inff())\n#define NAN (__builtin_nanf(\"\"))" + """
__device__ float4 max(float4 x, float4 y) { return float4(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w)); }
__device__ float4 pow(float x, float4 y) { return float4(pow(x, y.x), pow(x, y.y), pow(x, y.z), pow(x, y.w)); }
__device__ float4 pow(float4 x, float4 y) { return float4(pow(x.x, y.x), pow(x.y, y.y), pow(x.z, y.z), pow(x.w, y.w)); }
__device__ float4 log2(float4 x) { return float4(log2(x.x), log2(x.y), log2(x.z), log2(x.w)); }
__device__ float4 exp2(float4 x) { return float4(exp2(x.x), exp2(x.y), exp2(x.z), exp2(x.w)); }
__device__ float4 sin(float4 x) { return float4(sin(x.x), sin(x.y), sin(x.z), sin(x.w)); }
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
extern "C" __global__
  """, launch_bounds=True,
  smem_prefix = "__shared__ ", barrier = "__syncthreads();", float4 = "make_float4", uses_vload=True, uses_ptr_arithmetic=True, arg_int_prefix = "const int",
  half_prekernel = "#include <hip/hip_fp16.h>\nusing half4 = HIP_vector_type<half, 4>;" + """
__device__ float vload_half(size_t offset, const half *p) { return (float)*(p + offset); }
__device__ float2 vload_half2(size_t offset, const half *p) { return make_float2((float)*(p + offset*2), (float)*(p + offset*2 + 1)); }
__device__ float4 vload_half4(size_t offset, const half *p) { return make_float4((float)*(p + offset*4), (float)*(p + offset*4 + 1), (float)*(p + offset*4 + 2), (float)*(p + offset*4 + 3)); }
__device__ void vstore_half(float data, size_t offset, half *p) { *(p + offset) = (half)data; }
__device__ void vstore_half2(float2 data, size_t offset, half *p) { *(p + offset*2) = (half)data.x; *(p + offset*2 + 1) = (half)data.y; }
__device__ void vstore_half4(float4 data, size_t offset, half *p) { *(p + offset*4) = (half)data.x; *(p + offset*4 + 1) = (half)data.y; *(p + offset*4 + 2) = (half)data.z; *(p + offset*4 + 3) = (half)data.w; }
  """,
  gid = [f'blockIdx.{chr(120+i)}' for i in range(3)],
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)]))
HIPBuffer = Compiled(RawHIPBuffer, LinearizerOptions(device="HIP"), renderer, HIPProgram, hip.hipDeviceSynchronize, HIPGraph)
