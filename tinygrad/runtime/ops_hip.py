import ctypes
from tinygrad.helpers import mv_address, getenv, suppress_finalizing
from tinygrad.device import BufferStorage, Compiled, Allocator, BufferSpec, Program, TinyELF
from tinygrad.runtime.autogen import hip
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.c import init_c_var
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

class HIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.arch = init_c_var(hip.hipDeviceProp_t, lambda x: check(hip.hipGetDeviceProperties(x, self.device_id))).gcnArchName.decode()
    self.time_event_st, self.time_event_en = [init_c_var(hip.hipEvent_t, lambda x: hip.hipEventCreate(ctypes.byref(x), 0)) for _ in range(2)]

    super().__init__(device, HIPAllocator(self), [HIPRenderer], HIPProgram, arch=self.arch)

  def count(self) -> int: return init_c_var(ctypes.c_int, lambda x: check(hip.hipGetDeviceCount(x))).value

  def synchronize(self, timeout:int|None=None):
    check(hip.hipSetDevice(self.device_id))
    check(hip.hipDeviceSynchronize())

class HIPProgram(Program[HIPDevice]):
  def __init__(self, dev:HIPDevice, obj:TinyELF):
    self.dev, self.name, self.lib, self.signature = dev, obj.name, obj.lib, obj.signature
    check(hip.hipSetDevice(self.dev.device_id))
    self.module = init_c_var(hip.hipModule_t, lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), obj.lib)))
    self.prg = init_c_var(hip.hipFunction_t, lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, obj.name.encode("utf-8"))))

  @suppress_finalizing
  def __del__(self):
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False, **kw):
    args = (*args, *vals)
    check(hip.hipSetDevice(self.dev.device_id))
    c_args = TinyELF.pack(self.signature, args)
    vargs = (ctypes.c_void_p * 5)(1, mv_address(c_args), 2, ctypes.addressof(_arg_size:=ctypes.c_size_t(len(c_args))), 3)

    if wait: check(hip.hipEventRecord(self.dev.time_event_st, None))

    check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, vargs))

    if wait:
      check(hip.hipEventRecord(self.dev.time_event_en, None))
      check(hip.hipEventSynchronize(self.dev.time_event_en))
      check(hip.hipEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), self.dev.time_event_st, self.dev.time_event_en))
      return ret.value * 1e-3

class HIPAllocator(Allocator[HIPDevice]):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    check(hip.hipSetDevice(self.dev.device_id))
    return BufferStorage(init_c_var(hip.hipDeviceptr_t, lambda x: check(hip.hipMalloc(ctypes.byref(x), size))))

  def _free(self, storage:BufferStorage, options:BufferSpec): check(hip.hipFree(storage.buf))
  def _copyin(self, dest, src: memoryview):
    check(hip.hipSetDevice(self.dev.device_id))
    check(hip.hipMemcpy(dest, mv_address(src), len(src), hip.hipMemcpyHostToDevice))
  def _copyout(self, dest:memoryview, src):
    self.dev.synchronize()
    check(hip.hipMemcpy(mv_address(dest), src, len(dest), hip.hipMemcpyDeviceToHost))
  def _offset(self, buf, size:int, offset:int): return hip.hipDeviceptr_t(buf.value + offset)
