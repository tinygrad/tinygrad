import ctypes, functools, mmap, platform
from typing import Any, cast
from tinygrad.helpers import unwrap
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile, TextFileDesc
from test.mockgpu.qcom.qcomgpu import QCOMGPU

def _ioctl_nr(ioctl): return ioctl.args[2]

class QCOMFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset):
    addr = libc.mmap(0, sz, prot, flags|mmap.MAP_ANONYMOUS, -1, 0)
    self.driver.gpu.map_range(addr, sz)
    return addr

class _NoFlush:
  @staticmethod
  def fxn(*_): return None

def _map_cpu(self, buf): return self.dev._gpu_map(buf.host.addr, buf.nbytes)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.gpu, self.next_fd, self.next_id, self.ctx, self._hooked, self._err = QCOMGPU(), 1 << 28, 1, 1, False, None
    self.tracked_files += [VirtFile('/dev/kgsl-3d0', functools.partial(QCOMFileDesc, driver=self)),
                           VirtFile('/sys/class/kgsl/kgsl-3d0/idle_timer', functools.partial(TextFileDesc, text='4294967276\n'))]

  def open(self, name, flags, mode, virtfile):
    self.next_fd += 1
    return virtfile.fdcls(self.next_fd - 1)

  def _hook(self):
    if self._hooked: return
    self._hooked = True
    import tinygrad.runtime.ops_qcom as qcom
    if platform.machine() != 'aarch64': cast(Any, qcom).dcache_flush = lambda: _NoFlush()
    setattr(qcom.QCOMAllocator, '_map', _map_cpu)
    qcom.QCOMDevice.wait_timeout_ms = 50
    @ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
    def ioctl(fd, req, arg):
      try: return self.ioctl(int(req), int(arg))
      except Exception as e:
        self._err = e
        return -1
    self._cb = ioctl
    import tinygrad.runtime.support.hcq2 as hcq2
    hcq2.cfunc_buf('libc', 'ioctl').host.view(fmt='Q')[0] = unwrap(ctypes.cast(ioctl, ctypes.c_void_p).value)

  def ioctl(self, req, argp):
    self._hook()
    if self._err is not None:
      err, self._err = self._err, None
      raise err
    nr = req & 0xff
    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
      kgsl.struct_kgsl_drawctxt_create.from_address(argp).drawctxt_id = self.ctx
      self.ctx += 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if prop.type == kgsl.KGSL_PROP_DEVICE_INFO:
        info = kgsl.struct_kgsl_devinfo.from_address(cast(int, prop.value))
        info.device_id, info.chip_id, info.mmu_enabled, info.gpu_id, info.gmem_sizebytes = 0, 0x06030001, 1, 630, 1 << 20
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      kgsl.struct_kgsl_gpuobj_alloc.from_address(argp).id = self.next_id
      self.next_id += 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
      mapped = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      mapped.gpuaddr = mapped.hostptr
      self.gpu.map_range(mapped.hostptr, mapped.len)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      cmd = kgsl.struct_kgsl_gpu_command.from_address(argp)
      for i in range(cmd.numcmds):
        obj = kgsl.struct_kgsl_command_object.from_address(cmd.cmdlist + i * max(cmd.cmdsize, 1))
        cmd.timestamp = self.gpu.submit(obj.gpuaddr, obj.size)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID):
      kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid.from_address(argp).timestamp = self.gpu.timestamp
    elif nr in (_ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY), _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE), _ioctl_nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE),
                _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID)): pass
    else: raise RuntimeError(f'unhandled kgsl ioctl {nr:#x}')
    return 0
