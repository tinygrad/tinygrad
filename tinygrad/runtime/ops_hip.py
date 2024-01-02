import ctypes, functools, subprocess
from typing import Tuple, TypeVar
import gpuctypes.hip as hip
from tinygrad.helpers import DEBUG, getenv, from_mv, init_c_var, compile_cuda_style, encode_args_cuda_style, time_execution_cuda_style
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions

# The default HIP stream is used for everything.
HIPCPU = getenv("HIPCPU") # for CI. don't run kernels, only check if they compile

if HIPCPU:
  remu = ctypes.CDLL("/root/code/tinygrad/rdna/target/release/libremu.so")
  hip.hipSetDevice = lambda x: x
  hip.hipDeviceptr_t = lambda: ctypes.c_void_p(None)
  hip.hipModule_t = lambda: ctypes.c_void_p(None)
  hip.hipDeviceSynchronize = lambda: 0

  hip.hipMalloc = remu.hipMalloc
  hip.hipMemcpy = remu.hipMemcpy
  hip.hipModuleLaunchKernel.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
  hip.hipModuleLaunchKernel = lambda prg, *args: remu.hipModuleLaunchKernel(prg, len(prg), *args)

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

# TODO: remove these helpers, they increase complexity
def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)  # noqa: E501

def compile_hip(prg) -> bytes: return compile_cuda_style(prg, [f'--offload-arch={HIPDevice.default_arch_name}'], hip.hiprtcProgram, hip.hiprtcCreateProgram, hip.hiprtcCompileProgram, hip.hiprtcGetCode, hip.hiprtcGetCodeSize, hip.hiprtcGetProgramLog, hip.hiprtcGetProgramLogSize, check)  # noqa: E501

class HIPProgram:
  def __init__(self, device:int, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    if not HIPCPU:
      self.module = init_c_var(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
      prg = init_c_var(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))
    self.prg = prg if not HIPCPU else lib

  def __del__(self):
    if not HIPCPU: check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], vals:Tuple[int, ...]=(), wait=False):
    check(hip.hipSetDevice(self.device))
    args = (*args, *vals)
    return hip_time_execution(lambda: check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))), enable=wait)  # noqa: E501

T = TypeVar("T")
class HIPAllocator(LRUAllocator):
  def __init__(self, device):
    self.device = device
    super().__init__()
  def _alloc(self, size:int):
    check(hip.hipSetDevice(self.device))
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _free(self, opaque:T): check(hip.hipFree(opaque))
  def copyin(self, dest:T, src: memoryview):
    check(hip.hipSetDevice(self.device))
    # TODO: have to make sure src isn't freed to make this async
    check(hip.hipMemcpy(dest, from_mv(src), len(src), hip.hipMemcpyHostToDevice))
  def copyout(self, dest:memoryview, src:T):
    check(hip.hipSetDevice(self.device))
    check(hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost))
  def transfer(self, dest:T, src:T, sz:int):
    check(hip.hipSetDevice(self.device))
    # TODO: hipMemcpyAsync, but you have to track the "src" buffer to not free it
    check(hip.hipMemcpy(dest, src, sz, hip.hipMemcpyDeviceToDevice))

class HIPDevice(Compiled):
  default_arch_name = "gfx1100"
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    if self.device == 0 and not HIPCPU: HIPDevice.default_arch_name = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device))).gcnArchName.decode()  # noqa: E501

    from tinygrad.runtime.graph.hip import HIPGraph
    super().__init__(HIPAllocator(self.device), LinearizerOptions(device="HIP"), HIPRenderer,
                     compile_hip, functools.partial(HIPProgram, self.device), HIPGraph)
  def synchronize(self):
    check(hip.hipSetDevice(self.device))
    check(hip.hipDeviceSynchronize())