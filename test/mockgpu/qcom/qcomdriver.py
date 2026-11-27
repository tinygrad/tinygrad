import ctypes, functools, mmap
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile
from test.mockgpu.qcom.qcomgpu import QCOMGPU

def _ioctl_nr(ioctl:functools.partial) -> int: return ioctl.args[2]

class EmulatorError(Exception): pass # Compiled.synchronize turns a RuntimeError into "hang detected"

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.kgsl_ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return libc.mmap(start, sz, prot, flags|mmap.MAP_ANONYMOUS, -1, 0)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.tracked_files += [VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self))]
    self.gpu, self.next_fd, self.next_id, self.timestamp = QCOMGPU(), 1 << 30, 1, 0

  def open(self, name, flags, mode, virtfile):
    self.next_fd += 1
    return virtfile.fdcls(self.next_fd)

  def kgsl_ioctl(self, req, argp):
    nr = req & 0xFF
    self.gpu.progress()
    # GPU_COMMAND runs in a ctypes callback that can't raise, errors surface on the next wait
    if self.gpu.error is not None and nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID):
      err, self.gpu.error = self.gpu.error, None
      self.gpu.progress(drain=True)
      raise EmulatorError(str(err)) from err
    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE): kgsl.struct_kgsl_drawctxt_create.from_address(argp).drawctxt_id = 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if prop.type != kgsl.KGSL_PROP_DEVICE_INFO: raise RuntimeError(f"unsupported kgsl property {prop.type}")
      ctypes.cast(prop.value, ctypes.POINTER(kgsl.struct_kgsl_devinfo)).contents.chip_id = 0x06030000
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      alloc = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      alloc.id, self.next_id = self.next_id, self.next_id + 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
      mi = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      mi.gpuaddr = mi.hostptr
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      cmd = kgsl.struct_kgsl_gpu_command.from_address(argp)
      for i in range(cmd.numcmds):
        obj = kgsl.struct_kgsl_command_object.from_address(cmd.cmdlist + i * cmd.cmdsize)
        self.gpu.submit(obj.gpuaddr, obj.size // 4)
      self.timestamp += 1
      cmd.timestamp = self.timestamp
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID):
      kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid.from_address(argp).timestamp = self.timestamp
    elif nr not in {_ioctl_nr(i) for i in (kgsl.IOCTL_KGSL_SETPROPERTY, kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, kgsl.IOCTL_KGSL_GPUOBJ_FREE,
                                           kgsl.IOCTL_KGSL_SHAREDMEM_FREE)}:
      raise RuntimeError(f"unsupported kgsl ioctl {nr:#x}")
    return 0
