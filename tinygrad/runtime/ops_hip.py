import ctypes, functools
import extra.hip_wrapper as hip
from typing import Tuple, cast, Callable, TypeVar
from tinygrad.helpers import DEBUG, DType, getenv, diskcache, from_mv
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator
from tinygrad.renderer.hip import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 6:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.
MOCKHIP = getenv("MOCKHIP") # for CI. don't run kernels, only check if they compile

@diskcache
def compile_hip(prg) -> bytes:
  prog = hip.hiprtcCreateProgram(prg, "<null>", [], [])
  hip.hiprtcCompileProgram(prog, [f'--offload-arch={HIPDevice.default_arch_name}'])
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
  def __init__(self, device:int, name:str, prg:bytes, bufs:int, vars:int=0):
    self.device, self.c_struct_t = device, None

    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    if MOCKHIP: return
    hip.hipSetDevice(self.device)
    self.module = hip.hipModuleLoadData(prg)
    self.prg = hip.hipModuleGetFunction(self.module, name)
    self.c_struct_t = hip.getCStructForType([ctypes.c_void_p]*bufs + [ctypes.c_int]*vars)

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    if MOCKHIP: return
    hip.hipSetDevice(self.device)
    c_params = cast(Callable, self.c_struct_t)(*args)
    return time_execution(lambda: hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, 0, c_params), enable=wait)

  def __del__(self):
    if MOCKHIP: return
    hip.hipModuleUnload(self.module)

T = TypeVar("T")
class HIPAllocator(LRUAllocator):
  def __init__(self, device):
    self.device = device
    super().__init__()
  def _alloc(self, size: int, dtype: DType):
    if size == 0: return None
    hip.hipSetDevice(self.device)
    return hip.hipMalloc(size * dtype.itemsize)
  def _free(self, opaque:T): hip.hipFree(opaque)
  def copyin(self, dest:T, src: memoryview):
    hip.hipSetDevice(self.device)
    hip.hipMemcpyAsync(dest, from_mv(src), len(src), hip.hipMemcpyHostToDevice, 0)
  def copyout(self, dest:memoryview, src:T):
    hip.hipSetDevice(self.device)
    hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost)
  def transfer(self, dest:T, src:T, sz:int):
    hip.hipSetDevice(self.device)
    hip.hipMemcpy(dest, src, sz, hip.hipMemcpyDeviceToDevice)

class HIPDevice(Compiled):
  default_arch_name = "gfx1100"
  def __init__(self, device:str):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    if self.device == 0 and not MOCKHIP: HIPDevice.default_arch_name = hip.hipGetDeviceProperties(self.device).gcnArchName
    super().__init__(MallocAllocator if MOCKHIP else HIPAllocator(self.device), LinearizerOptions(device="HIP"), HIPRenderer, compile_hip, functools.partial(HIPProgram, self.device))
  def synchronize(self): hip.hipDeviceSynchronize()