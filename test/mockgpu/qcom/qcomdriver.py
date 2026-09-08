import ctypes, functools, mmap
from typing import Any
from tinygrad.runtime.autogen import kgsl, libc
from tinygrad.runtime.support import hcq2
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc, TextFileDesc, DirFileDesc
from test.mockgpu.qcom.qcomgpu import QCOMGPU

A630_CHIP_ID = 0x6030001

# HCQ2 QCOM submit is ccall(libc.dll.ioctl, ...)
_real_ioctl = libc.dll.ioctl
_qcom_drivers: list = []
def _dispatch_ioctl(fd, request, argp):
  for d in _qcom_drivers:
    if fd in d.kgsl_fds: return int(d.kgsl_ioctl(request, argp or 0) or 0)
  return int(_real_ioctl(fd, request, argp))
_mock_ioctl = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p)(_dispatch_ioctl)

_orig_cfunc_buf = hcq2.cfunc_buf
@functools.cache
def _cfunc_buf(lib:str, name:str):
  b = _orig_cfunc_buf(lib, name)
  if lib == "libc" and name == "ioctl":
    b._buf.view.view(fmt='Q')[0] = ctypes.cast(_mock_ioctl, ctypes.c_void_p).value
  return b

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver
    driver.kgsl_fds.add(fd)

  def ioctl(self, fd, request, argp):
    return self.driver.kgsl_ioctl(request, argp)

  def mmap(self, start, sz, prot, flags, fd, offset):
    addr = int(libc.mmap(start, sz, prot, flags | mmap.MAP_ANONYMOUS, -1, 0) or 0)
    obj = self.driver.objects.get(offset // 0x1000)
    if obj is not None:
      obj['va'] = addr
      if (obj['flags'] >> kgsl.KGSL_CACHEMODE_SHIFT) & 3 == kgsl.KGSL_CACHEMODE_UNCACHED:
        self.driver.track_address(addr, addr + sz, lambda mv, off: None, lambda mv, off: self.driver._emulate_execute())
    return addr

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.tracked_files += [
      VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self)),
      VirtFile('/sys/class/kgsl', functools.partial(DirFileDesc, child_names=['kgsl-3d0'])),
      VirtFile('/sys/class/kgsl/kgsl-3d0', functools.partial(DirFileDesc, child_names=['idle_timer'])),
      VirtFile('/sys/class/kgsl/kgsl-3d0/idle_timer', functools.partial(TextFileDesc, text='10\n')),
    ]
    self.gpu = QCOMGPU(0)
    self.next_fd = 1 << 30
    self.next_id = 1
    self.next_ts = 0
    self.objects: dict[int, dict] = {}
    self.kgsl_fds: set[int] = set()
    self._executing = False
    _qcom_drivers.append(self)
    hcq2.cfunc_buf = _cfunc_buf

  def _alloc_fd(self):
    fd = self.next_fd
    self.next_fd += 1
    return fd

  def open(self, name, flags, mode, virtfile):
    return virtfile.fdcls(self._alloc_fd())

  def _emulate_execute(self):
    if self._executing: return
    self._executing = True
    try: self.gpu.execute()
    finally: self._executing = False

  def kgsl_ioctl(self, req, argp):
    nr = req & 0xFF
    st: Any
    if nr == kgsl.IOCTL_KGSL_GPUOBJ_ALLOC.args[2]:
      st = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      st.id = self.next_id
      self.next_id += 1
      self.objects[st.id] = {'size': st.size, 'va': None, 'flags': st.flags}
    elif nr == kgsl.IOCTL_KGSL_DRAWCTXT_CREATE.args[2]:
      kgsl.struct_kgsl_drawctxt_create.from_address(argp).drawctxt_id = 1
    elif nr == kgsl.IOCTL_KGSL_SETPROPERTY.args[2]:
      pass
    elif nr == kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY.args[2]:
      st = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if st.type == kgsl.KGSL_PROP_DEVICE_INFO:
        kgsl.struct_kgsl_devinfo.from_address(st.value).chip_id = A630_CHIP_ID
    elif nr == kgsl.IOCTL_KGSL_MAP_USER_MEM.args[2]:
      st = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      st.gpuaddr = st.hostptr
    elif nr == kgsl.IOCTL_KGSL_GPUOBJ_FREE.args[2]:
      self.objects.pop(kgsl.struct_kgsl_gpuobj_free.from_address(argp).id, None)
    elif nr == kgsl.IOCTL_KGSL_SHAREDMEM_FREE.args[2]:
      pass
    elif nr == kgsl.IOCTL_KGSL_GPU_COMMAND.args[2]:
      st = kgsl.struct_kgsl_gpu_command.from_address(argp)
      obj = kgsl.struct_kgsl_command_object.from_address(st.cmdlist)
      self.gpu.submit_ib(obj.gpuaddr, obj.size)
      self.next_ts += 1
      st.timestamp = self.next_ts
      self._emulate_execute()
    elif nr == kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID.args[2]:
      self._emulate_execute()
      kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid.from_address(argp).timestamp = self.next_ts
    elif nr == kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID.args[2]:
      self._emulate_execute()
    else:
      raise RuntimeError(f"unknown kgsl ioctl {nr:#x}")
    return 0
