import numpy as np
import ctypes
import extra.hip_wrapper as hip
from typing import Tuple
from tinygrad.helpers import DEBUG, getenv, diskcache
from tinygrad.ops import Compiled
from tinygrad.renderer.hip import HIPRenderer
from tinygrad.runtime.lib import RawBuffer, RawBufferCopyInOut, LRUAllocator, RawBufferTransfer
from tinygrad.codegen.kernel import LinearizerOptions

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
  def __init__(self, device=None):
    self.default_device = device or getenv("HIP_DEFAULT_DEVICE")
    hip.hipSetDevice(self.default_device)
    self.device_count = hip.hipGetDeviceCount()
    self.allocator = HIPAllocator(hip.hipGetDeviceProperties(self.default_device).totalGlobalMem)
HIP = _HIP()

class RawHIPBuffer(RawBufferCopyInOut, RawBufferTransfer):
  def __init__(self, size, dtype, device=HIP.default_device, buf=None, allocator=HIP.allocator): super().__init__(size, dtype, buf=buf, allocator=allocator, **{'device': int(device)})
  def _copyin(self, x:np.ndarray):
    hip.hipSetDevice(self._device)
    hip.hipMemcpyAsync(self._buf, np.require(x, requirements='C').ctypes.data_as(ctypes.c_void_p), self.size * self.dtype.itemsize, hip.hipMemcpyHostToDevice, 0)
  def _copyout(self, x:np.ndarray):
    hip.hipSetDevice(self._device)
    hip.hipMemcpy(x.ctypes.data, self._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToHost)
  def _transfer(self, x:RawBuffer):
    hip.hipSetDevice(x._device)
    hip.hipMemcpy(self._buf, x._buf, self.size * self.dtype.itemsize, hip.hipMemcpyDeviceToDevice)

@diskcache
def compile_hip(prg) -> bytes:
  prog = hip.hiprtcCreateProgram(prg, "<null>", [], [])
  hip.hiprtcCompileProgram(prog, [f'--offload-arch={hip.hipGetDeviceProperties(HIP.default_device).gcnArchName}'])
  return hip.hiprtcGetCode(prog)

class HIPProgram:
  def __init__(self, name:str, prg:bytes):
    self.modules, self.prgs = [], []

    if DEBUG >= 6:
      asm = early_exec((["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], prg))
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    for i in range(HIP.device_count):
      hip.hipSetDevice(i)
      self.modules.append(hip.hipModuleLoadData(prg))
      self.prgs.append(hip.hipModuleGetFunction(self.modules[-1], name))

  def __call__(self, *args, global_size:Tuple[int,int,int], local_size:Tuple[int,int,int], wait=False):
    hip.hipSetDevice(args[0]._device)
    if wait:
      start, end = hip.hipEventCreate(), hip.hipEventCreate()
      hip.hipEventRecord(start)
    struct = hip.getStructTypeForArgs(*args)(*[data._buf if not isinstance(data, int) else np.int32(data) for data in args])
    hip.hipModuleLaunchKernel(self.prgs[args[0]._device], global_size[0], global_size[1], global_size[2], local_size[0], local_size[1], local_size[2], 0, 0, struct)
    if wait:
      hip.hipEventRecord(end)
      hip.hipEventSynchronize(end)
      ret = hip.hipEventElapsedTime(start, end)*1e-3
      hip.hipEventDestroy(start)
      hip.hipEventDestroy(end)
      return ret

  def __del__(self):
    for module in self.modules: hip.hipModuleUnload(module)

HIPBuffer = Compiled(RawHIPBuffer, LinearizerOptions(device="HIP"), HIPRenderer, compile_hip, HIPProgram, hip.hipDeviceSynchronize)
