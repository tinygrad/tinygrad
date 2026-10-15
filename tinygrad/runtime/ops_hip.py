from types import SimpleNamespace
from tinygrad.helpers import getenv
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite
from tinygrad.device import Compiled
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.autogen import hip
from tinygrad.runtime.support.hcq2 import ccall
from tinygrad.runtime.support.c import init_c_var
from tinygrad.runtime.ops_cuda import CUDAQueue, CUDADevice, check
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

HIPIFY = dict(CUdevice=hip.hipDevice_t, CUcontext=hip.hipCtx_t, CUstream=hip.hipStream_t, CUdeviceptr=hip.hipDeviceptr_t, CUmodule=hip.hipModule_t,
  CUfunction=hip.hipFunction_t, cuInit=hip.hipInit, cuDeviceGet=hip.hipDeviceGet, cuDeviceGetCount=hip.hipGetDeviceCount,
  cuCtxCreate_v2=hip.hipCtxCreate, cuCtxSetCurrent=hip.hipCtxSetCurrent, cuCtxSynchronize=hip.hipDeviceSynchronize, cuMemAlloc_v2=hip.hipMalloc,
  cuStreamCreate=hip.hipStreamCreateWithFlags, cuMemHostGetDevicePointer_v2=hip.hipHostGetDevicePointer, cuMemFree_v2=hip.hipFree,
  cuMemHostAlloc=lambda p, sz, _: hip.hipHostMalloc(p, sz, hip.hipHostMallocCoherent), cuMemFreeHost=hip.hipHostFree,
  cuMemHostRegister_v2=hip.hipHostRegister, cuMemHostUnregister=hip.hipHostUnregister, cuModuleLoadData=hip.hipModuleLoadData,
  cuModuleGetFunction=hip.hipModuleGetFunction, cuLaunchKernel=hip.hipModuleLaunchKernel, cuMemcpyAsync=hip.hipMemcpyAsync,
  cuStreamWaitValue64_v2=hip.hipStreamWaitValue64, cuStreamWriteValue64_v2=hip.hipStreamWriteValue64, cuLaunchHostFunc=hip.hipLaunchHostFunc)

def hipify(c:UOp, f:UOp) -> UOp|None:
  if (fn:=HIPIFY.get(f.arg.name)) is None: return None
  args = list(c.src[1:])
  if fn is hip.hipMemcpyAsync: args.insert(3, UOp.const(hip.hipMemcpyDefault, dtypes.int))
  if fn is hip.hipStreamWaitValue64: args.append(UOp.const((1<<64)-1, dtypes.uint64))
  return ccall(fn, *args)
pm_hipify = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.CUSTOM_FUNCTION, name="f"),), allow_any_len=True, name="c"), hipify)])

def hip_encode(s:UOp) -> UOp: return graph_rewrite(CUDAQueue(s).encode(), pm_hipify, enter_calls=True) # the cuda queues encode hip calls
Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("hip_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), hip_encode),
  (UPat.sink(arg=KernelInfo("hip_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"), hip_encode),
])

class HIPDevice(CUDADevice):
  api, launch_end, renderers = SimpleNamespace(**HIPIFY), 3, [HIPRenderer]

  def get_arch(self, device_id:int) -> str:
    return init_c_var(hip.hipDeviceProp_t, lambda x: check(hip.hipGetDeviceProperties(x, device_id))).gcnArchName.decode().split(":")[0]
