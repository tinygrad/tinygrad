import ctypes
import extra.hip_wrapper as hip
from typing import Tuple, cast, Callable, TypeVar
from tinygrad.helpers import DEBUG, DType, getenv, diskcache, from_mv
from tinygrad.device import Compiled, Allocator, MallocAllocator
from tinygrad.renderer.hip import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions

# TODO: if you fork and exit the child process after creating anything with cl on AMD, it hangs on e.wait()
if DEBUG >= 6:
  from extra.helpers import enable_early_exec
  early_exec = enable_early_exec()

# The default HIP stream is used for everything.
MOCKHIP = getenv("MOCKHIP") # for CI. don't run kernels, only check if they compile

class _HIP:
  def __init__(self, device=None):
    self.default_device = device or getenv("HIP_DEFAULT_DEVICE")
    self.device_count = 0 if MOCKHIP else hip.hipGetDeviceCount()
    if not MOCKHIP: hip.hipSetDevice(self.default_device)
HIP = _HIP()

@diskcache
def compile_hip(prg) -> bytes:
  prog = hip.hiprtcCreateProgram(prg, "<null>", [], [])
  arch = "gfx1100" if MOCKHIP else hip.hipGetDeviceProperties(HIP.default_device).gcnArchName
  hip.hiprtcCompileProgram(prog, [f'--offload-arch={arch}'])
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
    if MOCKHIP: return
    hip.hipSetDevice(args[0]._device)
    if self.c_struct_t is None: self.c_struct_t = hip.getCStructForType([(ctypes.c_void_p if not isinstance(x, int) else ctypes.c_int) for x in args])
    c_params = cast(Callable, self.c_struct_t)(*[x._buf if not isinstance(x, int) else x for x in args])
    return time_execution(lambda: hip.hipModuleLaunchKernel(self.prgs[args[0]._device], *global_size, *local_size, 0, 0, c_params), enable=wait)

  def __del__(self):
    for module in self.modules: hip.hipModuleUnload(module)

T = TypeVar("T")
class HIPAllocator(Allocator):
  def __init__(self, device): self.device = device
  def _alloc(self, size: int, dtype: DType):
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
  def __init__(self, device:str):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    super().__init__(MallocAllocator if MOCKHIP else HIPAllocator(self.device), LinearizerOptions(device="HIP"), HIPRenderer, compile_hip, HIPProgram)
  def synchronize(self): hip.hipDeviceSynchronize()