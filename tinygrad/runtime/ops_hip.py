import ctypes, functools
from typing import Tuple, TypeVar
import gpuctypes.hip as hip
from tinygrad.helpers import DEBUG, DType, getenv, diskcache, from_mv, init_c_var, compile_cuda_style, encode_args_cuda_style, time_execution_cuda_style
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator
from tinygrad.renderer.hip import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 6:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.
MOCKHIP = getenv("MOCKHIP") # for CI. don't run kernels, only check if they compile

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)

@diskcache
def compile_hip(prg) -> bytes: return compile_cuda_style(prg, [f'--offload-arch={HIPDevice.default_arch_name}'], hip.hiprtcProgram, hip.hiprtcCreateProgram, hip.hiprtcCompileProgram, hip.hiprtcGetCode, hip.hiprtcGetCodeSize, hip.hiprtcGetProgramLog, hip.hiprtcGetProgramLogSize, check)

class HIPProgram:
  def __init__(self, device:int, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], lib))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    if MOCKHIP: return
    check(hip.hipSetDevice(self.device))
    self.module = init_c_var(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
    self.prg = init_c_var(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))

  def __del__(self):
    if not MOCKHIP: check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    if MOCKHIP: return float("inf")
    check(hip.hipSetDevice(self.device))
    return hip_time_execution(lambda: check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, encode_args_cuda_style(args, hip.hipDeviceptr_t, marks=(1,2,3))[0])), enable=wait)

T = TypeVar("T")
class HIPAllocator(LRUAllocator):
  def __init__(self, device):
    self.device = device
    super().__init__()
  def _alloc(self, size: int, dtype: DType):
    check(hip.hipSetDevice(self.device))
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size * dtype.itemsize)))
  def _free(self, opaque:T): check(hip.hipFree(opaque))
  def copyin(self, dest:T, src: memoryview):
    check(hip.hipSetDevice(self.device))
    check(hip.hipMemcpyAsync(dest, from_mv(src), len(src), hip.hipMemcpyHostToDevice, None))
  def copyout(self, dest:memoryview, src:T):
    check(hip.hipSetDevice(self.device))
    check(hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost))
  def transfer(self, dest:T, src:T, sz:int):
    check(hip.hipSetDevice(self.device))
    check(hip.hipMemcpy(dest, src, sz, hip.hipMemcpyDeviceToDevice))

class HIPDevice(Compiled):
  default_arch_name = "gfx1100"
  def __init__(self, device:str):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    if self.device == 0 and not MOCKHIP: HIPDevice.default_arch_name = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device))).gcnArchName.decode()

    from tinygrad.runtime.graph.hip import HIPGraph
    super().__init__(MallocAllocator if MOCKHIP else HIPAllocator(self.device), LinearizerOptions(device="HIP"), HIPRenderer, compile_hip, functools.partial(HIPProgram, self.device), HIPGraph)
  def synchronize(self): hip.hipDeviceSynchronize()