from __future__ import annotations
import os, ctypes, ctypes.util, functools, subprocess
from typing import Tuple, TypeVar, List
import gpuctypes.hip as hip
from tinygrad.helpers import DEBUG, getenv, from_mv, init_c_var, compile_cuda_style, encode_args_cuda_style, time_execution_cuda_style, round_up
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.read.restype = ctypes.c_size_t

# The default HIP stream is used for everything.
MOCKHIP = getenv("MOCKHIP") # for CI. don't run kernels, only check if they compile

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

# TODO: remove these helpers, they increase complexity
def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)  # noqa: E501

def compile_hip(prg) -> bytes: return compile_cuda_style(prg, [f'--offload-arch={HIPDevice.default_arch_name}', '-I/opt/rocm/include'], hip.hiprtcProgram, hip.hiprtcCreateProgram, hip.hiprtcCompileProgram, hip.hiprtcGetCode, hip.hiprtcGetCodeSize, hip.hiprtcGetProgramLog, hip.hiprtcGetProgramLogSize, check)  # noqa: E501

class HIPProgram:
  def __init__(self, device:int, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    if MOCKHIP: return
    check(hip.hipSetDevice(self.device))
    self.module = init_c_var(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
    self.prg = init_c_var(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))

  def __del__(self):
    if not MOCKHIP: check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], vals:Tuple[int, ...]=(), wait=False):
    if MOCKHIP: return float("inf")
    check(hip.hipSetDevice(self.device))
    return hip_time_execution(lambda: check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, encode_args_cuda_style(args, vals, hip.hipDeviceptr_t, marks=(1,2,3))[0])), enable=wait)  # noqa: E501

T = TypeVar("T")
CHUNK_SIZE, PAGE_SIZE = 128*1024*1024, 0x1000
class HIPAllocator(LRUAllocator):
  def __init__(self, device:HIPDevice):
    self.device = device
    check(hip.hipSetDevice(self.device.device))
    self.hb = [self._hostalloc(CHUNK_SIZE), self._hostalloc(CHUNK_SIZE)]
    super().__init__()
  def _alloc(self, size:int):
    check(hip.hipSetDevice(self.device.device))
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _free(self, opaque:T): check(hip.hipFree(opaque))
  def _hostalloc(self, size:int): return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipHostMalloc(ctypes.byref(x), size, 0)))
  def _copyin_async(self, dest:T, src:T, size:int): check(hip.hipMemcpyAsync(dest, src, size, hip.hipMemcpyHostToDevice, None))
  def _copy_from_fd(self, dest, fd, offset, size):
    minor_offset = offset % PAGE_SIZE
    offset -= minor_offset
    real_size = round_up(size+minor_offset, PAGE_SIZE)
    os.lseek(fd, offset, os.SEEK_SET)
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(real_size-local_offset, CHUNK_SIZE)
      ret = libc.read(fd, self.hb[0], local_size)
      assert ret == local_size, f"{ret} != {local_size}"
      self.device.synchronize()
      copy_size = min(local_size-minor_offset, size-copied_in)
      self._copyin_async(ctypes.c_void_p(dest.value + copied_in), ctypes.c_void_p(self.hb[0].value + minor_offset), copy_size)
      copied_in += copy_size
      self.hb = self.hb[::-1]
      minor_offset = 0 # only on the first
    assert copied_in == size, f"{copied_in} != {size}"
  def copyin(self, dest:T, src: memoryview):
    check(hip.hipSetDevice(self.device.device))
    host_mem = self._hostalloc(len(src))
    self.device.pending_copyin.append(host_mem)
    ctypes.memmove(host_mem, from_mv(src), len(src))
    check(hip.hipMemcpyAsync(dest, host_mem, len(src), hip.hipMemcpyHostToDevice, None))
  def copyout(self, dest:memoryview, src:T):
    check(hip.hipSetDevice(self.device.device))
    check(hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost))
  def transfer(self, dest:T, src:T, sz:int):
    check(hip.hipSetDevice(self.device.device))
    # TODO: hipMemcpyAsync, but you have to track the "src" buffer to not free it
    check(hip.hipMemcpy(dest, src, sz, hip.hipMemcpyDeviceToDevice))

class HIPDevice(Compiled):
  default_arch_name = "gfx1100"
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    self.pending_copyin: List[hip.hipDeviceptr_t] = []
    if self.device == 0 and not MOCKHIP: HIPDevice.default_arch_name = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device))).gcnArchName.decode()  # noqa: E501

    from tinygrad.runtime.graph.hip import HIPGraph
    super().__init__(MallocAllocator if MOCKHIP else HIPAllocator(self), LinearizerOptions(device="HIP"), HIPRenderer,
                     compile_hip, functools.partial(HIPProgram, self.device), HIPGraph)
  def synchronize(self):
    check(hip.hipSetDevice(self.device))
    check(hip.hipDeviceSynchronize())
    for opaque in self.pending_copyin: check(hip.hipFree(opaque))
    self.pending_copyin.clear()