from __future__ import annotations
import ctypes, functools, subprocess, io
from typing import Tuple, TypeVar, List, Any, cast, Set
import gpuctypes.hip as hip
from tinygrad.helpers import DEBUG, getenv, init_c_var, compile_cuda_style, encode_args_cuda_style, time_execution_cuda_style
from tinygrad.helpers import from_mv, round_up, to_mv, colored
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator, BufferOptions, JITRunner, Device, Buffer, update_stats
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions

# The default HIP stream is used for everything.
MOCKHIP = getenv("MOCKHIP") # for CI. don't run kernels, only check if they compile

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

# TODO: remove these helpers, they increase complexity
def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)  # noqa: E501

def compile_hip(prg:str, arch="gfx1100") -> bytes: return compile_cuda_style(prg, [f'--offload-arch={arch}', '-I/opt/rocm/include'], hip.hiprtcProgram, hip.hiprtcCreateProgram, hip.hiprtcCompileProgram, hip.hiprtcGetCode, hip.hiprtcGetCodeSize, hip.hiprtcGetProgramLog, hip.hiprtcGetProgramLogSize, check)  # noqa: E501

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
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], vals:Tuple[int, ...]=(), wait=False):
    if MOCKHIP: return float("inf")
    check(hip.hipSetDevice(self.device))
    return hip_time_execution(lambda: check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, encode_args_cuda_style(args, vals, hip.hipDeviceptr_t, marks=(1,2,3))[0])), enable=wait)  # noqa: E501

T = TypeVar("T")
CHUNK_SIZE, PAGE_SIZE = 256*1024*1024, 0x1000
class HIPAllocator(LRUAllocator):
  def __init__(self, device:HIPDevice):
    self.device = device
    self.track_cross_device: List[HIPDevice] = []
    super().__init__()
  def free_cache(self):
    self.device.synchronize()
    for x in self.track_cross_device: x.synchronize()
    return super().free_cache()
  def _alloc(self, size:int):
    check(hip.hipSetDevice(self.device.device))
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _alloc_with_options(self, size:int, options:BufferOptions):
    check(hip.hipSetDevice(self.device.device))
    if options.uncached:
      return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipExtMallocWithFlags(ctypes.byref(x), size, 3)))  # hipDeviceMallocUncached = 3
    elif options.host:
      return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipHostMalloc(ctypes.byref(x), size, 2 if options.signal else 0)))
    else:
      raise Exception("no options")
  def _free(self, opaque:T): check(hip.hipFree(opaque))
  def copy_from_fd(self, dest, fd, offset, size):
    check(hip.hipSetDevice(self.device.device))
    if not hasattr(self, 'hb'):
      self.hb = [self._alloc_with_options(CHUNK_SIZE, BufferOptions(host=True)) for _ in range(2)]
      self.hb_events = [None, None]
      self.hb_polarity = 0
    fo = io.FileIO(fd, "a+b", closefd=False)
    fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(round_up(size+minor_offset, PAGE_SIZE)-local_offset, CHUNK_SIZE)
      if self.hb_events[self.hb_polarity] is not None:
        # NOTE: block doesn't work here because we modify the CPU memory
        check(hip.hipEventSynchronize(self.hb_events[self.hb_polarity]))
        check(hip.hipEventDestroy(self.hb_events[self.hb_polarity]))
        self.hb_events[self.hb_polarity] = None
      fo.readinto(to_mv(self.hb[self.hb_polarity], local_size))
      check(hip.hipMemcpyAsync(ctypes.c_void_p(dest.value + copied_in), ctypes.c_void_p(self.hb[self.hb_polarity].value + minor_offset),
                               copy_size:=min(local_size-minor_offset, size-copied_in), hip.hipMemcpyHostToDevice, None))
      self.hb_events[self.hb_polarity] = init_c_var(hip.hipEvent_t(), lambda x: check(hip.hipEventCreate(ctypes.byref(x))))
      check(hip.hipEventRecord(self.hb_events[self.hb_polarity], None))
      copied_in += copy_size
      self.hb_polarity = (self.hb_polarity+1) % len(self.hb)
      minor_offset = 0 # only on the first
  def copyin(self, dest:T, src: memoryview):
    check(hip.hipSetDevice(self.device.device))
    host_mem = self._alloc_with_options(len(src), BufferOptions(host=True))
    self.device.pending_copyin.append(host_mem)
    ctypes.memmove(host_mem, from_mv(src), len(src))
    check(hip.hipMemcpyAsync(dest, host_mem, len(src), hip.hipMemcpyHostToDevice, None))
  def copyout(self, dest:memoryview, src:T):
    self.device.synchronize()
    check(hip.hipSetDevice(self.device.device))
    check(hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost))
  def transfer(self, dest:T, src:T, sz:int):
    check(hip.hipSetDevice(self.device.device))
    check(hip.hipMemcpyAsync(dest, src, sz, hip.hipMemcpyDeviceToDevice, None))

class HIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    self.arch = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device))).gcnArchName.decode() if not MOCKHIP else "gfx1100"  # noqa: E501
    self.pending_copyin: List[hip.hipDeviceptr_t] = []
    self.track_cross_buffer: List[Any] = []
    self.peers: Set[int] = set()

    from tinygrad.runtime.graph.hip import HIPGraph
    super().__init__(device, MallocAllocator if MOCKHIP else HIPAllocator(self), LinearizerOptions("HIP"), HIPRenderer,
                     functools.partial(compile_hip,arch=self.arch), f"compile_hip_{self.arch}", functools.partial(HIPProgram, self.device), HIPGraph)
  def synchronize(self):
    check(hip.hipSetDevice(self.device))
    check(hip.hipDeviceSynchronize())
    for opaque in self.pending_copyin: check(hip.hipFree(opaque))
    self.track_cross_buffer.clear()
    self.pending_copyin.clear()
  def enable_peer(self, dnum):
    if self.device == dnum or dnum in self.peers: return
    check(hip.hipSetDevice(self.device))
    check(hip.hipDeviceEnablePeerAccess(dnum, 0))
    self.peers.add(dnum)

class HIPSyncEvent(JITRunner):
  def __init__(self, lb):
    self.lb, self.device, self.dname = lb, cast(HIPDevice, Device[lb.device]), lb.device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals, wait=False, jit=False):
    to_mv(rawbufs[0]._buf, 4).cast("I")[0] = 0
    check(hip.hipSetDevice(self.device.device))
    check(hip.hipStreamWriteValue32(None, rawbufs[0]._buf, 1, 0))
    update_stats(colored("sync", "red"), 0, 0, {}, None, 1, device=self.dname)

class HIPWaitEvent(JITRunner):
  def __init__(self, device):
    self.device, self.dname = cast(HIPDevice, Device[device]), device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals, wait=False, jit=False):
    check(hip.hipSetDevice(self.device.device))
    check(hip.hipStreamWaitValue32(None, rawbufs[0]._buf, 1, 1, 0xFFFFFFFF))
    update_stats(colored("wait", "RED"), 0, 0, {}, None, 1, device=self.dname)
