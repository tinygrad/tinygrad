from __future__ import annotations
import ctypes, functools, subprocess, io
from typing import Tuple, TypeVar, List
import gpuctypes.hip as hip
from tinygrad.helpers import DEBUG, getenv, init_c_var, compile_cuda_style, encode_args_cuda_style, time_execution_cuda_style
from tinygrad.helpers import from_mv, round_up, to_mv
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions

# The default HIP stream is used for everything.
MOCKHIP = getenv("MOCKHIP") # for CI. don't run kernels, only check if they compile

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

# TODO: remove these helpers, they increase complexity
def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)  # noqa: E501

def compile_hip(prg) -> bytes: return compile_cuda_style(prg, *HIPDevice._comp_args)
def alloc_call(f, *args): return init_c_var(f.argtypes[0]._type_(), lambda x: check(f(ctypes.byref(x), *args)))

class HIPProgram:
  def __init__(self, device:HIPDevice, name:str, lib:bytes):
    self.hip_scall, self.name, self.lib = device.hip_scall, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    if MOCKHIP: return
    self.module = self.hip_scall(alloc_call, hip.hipModuleLoadData, lib, chk=False)
    self.prg = alloc_call(hip.hipModuleGetFunction, self.module, name.encode("utf-8"))

  def __del__(self):
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], vals:Tuple[int, ...]=(), wait=False):
    if MOCKHIP: return float("inf")
    self.hip_scall(hip_time_execution, lambda: check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None,
                  encode_args_cuda_style(args, vals, hip.hipDeviceptr_t, marks=(1,2,3))[0])), wait, chk=False)

T = TypeVar("T")
CHUNK_SIZE, PAGE_SIZE = 256*1024*1024, 0x1000
class HIPAllocator(LRUAllocator):
  def __init__(self, device:HIPDevice):
    self.device = device
    super().__init__()
  def _alloc(self, size:int): return self.device.hip_scall(alloc_call, hip.hipMalloc, size, chk=False)
  def _free(self, opaque:T): check(hip.hipFree(opaque))
  def _hostalloc(self, size:int): return alloc_call(hip.hipHostMalloc, size, 0) # <- watchout, this one does not set the device
  def copy_from_fd(self, dest, fd, offset, size):
    check(hip.hipSetDevice(self.device.device))
    if not hasattr(self, 'hb'): self.hb = [self._hostalloc(CHUNK_SIZE) for _ in range(2)]
    fo = io.FileIO(fd, "a+b", closefd=False)
    fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(round_up(size+minor_offset, PAGE_SIZE)-local_offset, CHUNK_SIZE)
      fo.readinto(to_mv(self.hb[0], local_size))
      check(hip.hipDeviceSynchronize())
      check(hip.hipMemcpyAsync(ctypes.c_void_p(dest.value + copied_in), ctypes.c_void_p(self.hb[0].value + minor_offset),
                               copy_size:=min(local_size-minor_offset, size-copied_in), hip.hipMemcpyHostToDevice, None))
      copied_in += copy_size
      self.hb = self.hb[1:] + [self.hb[0]]
      minor_offset = 0 # only on the first
  def copyin(self, dest:T, src: memoryview):
    host_mem = self.device.hip_scall(self._hostalloc, len(src), chk=False)
    self.device.pending_copyin.append(host_mem)
    ctypes.memmove(host_mem, from_mv(src), len(src))
    check(hip.hipMemcpyAsync(dest, host_mem, len(src), hip.hipMemcpyHostToDevice, None))
  def copyout(self, dest:memoryview, src:T): self.device.hip_scall(hip.hipMemcpy, from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost)
  # TODO: hipMemcpyAsync, but you have to track the "src" buffer to not free it
  def transfer(self, dest:T, src:T, sz:int): self.device.hip_scall(hip.hipMemcpy, dest, src, sz, hip.hipMemcpyDeviceToDevice)

class HIPDevice(Compiled):
  default_arch_name = "gfx1100"
  _comp_args = ([], hip.hiprtcProgram, hip.hiprtcCreateProgram, hip.hiprtcCompileProgram, hip.hiprtcGetCode,
           hip.hiprtcGetCodeSize, hip.hiprtcGetProgramLog, hip.hiprtcGetProgramLogSize, check) # type: ignore[var-annotated]
  def __init__(self, device:str=""):
    self.device, bas = int(device.split(":")[1]) if ":" in device else 0, (None, '', 'unavailable',)
    self.pending_copyin: List[hip.hipDeviceptr_t] = []

    if self.device == 0 and not MOCKHIP: HIPDevice.default_arch_name = alloc_call(hip.hipGetDeviceProperties, self.device).gcnArchName.decode()
    if not (o:=HIPDevice._comp_args[0]) and (a:=HIPDevice.default_arch_name) not in bas: o.extend(('-I/opt/rocm/include', f'--offload-arch={a}'))

    from tinygrad.runtime.graph.hip import HIPGraph
    super().__init__(MallocAllocator if MOCKHIP else HIPAllocator(self), LinearizerOptions("HIP"), HIPRenderer,
                     compile_hip, functools.partial(HIPProgram, self), HIPGraph)
  def synchronize(self):
    self.hip_scall(hip.hipDeviceSynchronize)
    for opaque in self.pending_copyin: check(hip.hipFree(opaque))
    self.pending_copyin.clear()
  def hip_scall(self, f, *args, chk=True): return (check(hip.hipSetDevice(self.device)), check(f(*args)) if chk else f(*args))[1]
