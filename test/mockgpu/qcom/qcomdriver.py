from __future__ import annotations
import ctypes, errno, functools, mmap
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile
from test.mockgpu.qcom.qcomgpu import A630GPU

def _ioctl_nr(ioctl: functools.partial) -> int: return ioctl.args[2]

A630_CHIP_ID = 0x06030001

kgsl_ioctl_info = {
  _ioctl_nr(ioctl): (name, ioctl.args[3]) for name, ioctl in vars(kgsl).items()
  if name.startswith("IOCTL_KGSL_") and isinstance(ioctl, functools.partial) and ioctl.args[3] is not None
}

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver
  def ioctl(self, fd, request, argp): return self.driver.kgsl_ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return self.driver.mmap_gpuobj(offset, sz)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.tracked_files += [VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self))]
    self.next_fd = 1 << 29
    self.next_id = 1
    self.next_ctx = 1
    self.next_ts = 1
    self.objects: dict[int, dict] = {}
    self.host_maps: dict[int, int] = {}
    self.gpu = A630GPU(self.check_range)

  def check_range(self, addr:int, size:int) -> None:
    if addr < 0 or size < 0: raise RuntimeError(f"invalid qcom memory range {addr:#x}+{size:#x}")
    end = addr + size
    for obj in self.objects.values():
      if obj['ptr'] <= addr and end <= obj['ptr'] + obj['size']: return
    for ptr, mapped_size in self.host_maps.items():
      if ptr <= addr and end <= ptr + mapped_size: return
    raise RuntimeError(f"unmapped qcom memory range {addr:#x}+{size:#x}")

  def _alloc_fd(self):
    fd = self.next_fd
    self.next_fd += 1
    return fd

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())

  def mmap_gpuobj(self, offset, sz):
    obj_id = offset // 0x1000
    obj = self.objects.get(obj_id)
    if obj is None: raise OSError(errno.EINVAL, f"mmap unknown kgsl id {obj_id}")
    return obj['ptr']

  def _anon(self, size:int) -> int:
    ptr = libc.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    if ptr == 0xFFFFFFFFFFFFFFFF or ptr is None: raise OSError("mmap failed")
    return int(ctypes.cast(ptr, ctypes.c_void_p).value or ptr)

  def kgsl_ioctl(self, req, argp):
    nr = req & 0xFF
    if nr not in kgsl_ioctl_info: raise RuntimeError(f"unknown kgsl ioctl nr={nr}")
    name, struct_type = kgsl_ioctl_info[nr]
    s = struct_type.from_address(argp)

    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
      s.drawctxt_id = self.next_ctx
      self.next_ctx += 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY):
      pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY):
      pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      if s.type == kgsl.KGSL_PROP_DEVICE_INFO:
        info = kgsl.struct_kgsl_devinfo.from_address(s.value)
        info.device_id = kgsl.KGSL_DEVICE_3D0
        info.chip_id = A630_CHIP_ID
        info.mmu_enabled = 1
        info.gpu_id = 630
        info.gmem_sizebytes = 1024 * 1024
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      size = int(s.size) or int(s.mmapsize) or 0x1000
      ptr = self._anon(size)
      obj_id = self.next_id
      self.next_id += 1
      s.id = obj_id
      s.mmapsize = size
      s.va_len = size
      self.objects[obj_id] = {'ptr': ptr, 'size': size}
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE):
      obj = self.objects.pop(s.id, None)
      if obj is not None: libc.munmap(obj['ptr'], obj['size'])
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
      s.gpuaddr = s.hostptr
      self.host_maps[s.gpuaddr] = int(s.len)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE):
      self.host_maps.pop(s.gpuaddr, None)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      ts = self.next_ts
      self.next_ts += 1
      s.timestamp = ts
      cmd_t = kgsl.struct_kgsl_command_object
      for i in range(s.numcmds):
        cmd = cmd_t.from_address(s.cmdlist + i * ctypes.sizeof(cmd_t))
        self.gpu.execute_ib(int(cmd.gpuaddr), int(cmd.size))
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID):
      self.gpu.resume()
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_INFO):
      obj = self.objects.get(s.id)
      if obj is None: return -1
      s.gpuaddr = obj['ptr']
      s.size = obj['size']
      s.va_addr = obj['ptr']
      s.va_len = obj['size']
    else:
      raise RuntimeError(f"unsupported kgsl ioctl {nr} {name}")
    return 0
