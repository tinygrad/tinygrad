import functools, mmap
from typing import cast
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile
from test.mockgpu.qcom.qcomgpu import QCOMGPU, CHIP_ID

def _ioctl_nr(ioctl: functools.partial) -> int: return ioctl.args[2]

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.kgsl_ioctl(request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return self.driver.kgsl_mmap(start, sz, prot, flags, offset)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.tracked_files += [VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self))]

    self.gpu = QCOMGPU(0)
    self.next_fd, self.next_id, self.next_ctx, self.timestamp = (1 << 30), 1, 1, 0
    self.objs: dict[int, tuple[int, int]] = {}  # id -> (size, va_addr), va_addr filled in on mmap

  def open(self, name, flags, mode, virtfile): return virtfile.fdcls(self._alloc_fd())

  def _alloc_fd(self):
    self.next_fd += 1
    return self.next_fd - 1

  def kgsl_mmap(self, start, sz, prot, flags, offset):
    if (obj_id:=offset // 0x1000) not in self.objs: raise RuntimeError(f"mmap of unknown gpuobj id {obj_id}")
    va = libc.mmap(start, sz, prot, flags | mmap.MAP_ANONYMOUS, -1, 0)
    self.objs[obj_id] = (self.objs[obj_id][0], va)
    self.gpu.map_range(va, sz)
    return va

  def kgsl_ioctl(self, req, argp):
    nr = req & 0xFF
    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
      kgsl.struct_kgsl_drawctxt_create.from_address(argp).drawctxt_id = self.next_ctx
      self.next_ctx += 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      st = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      st.id = self.next_id
      self.next_id += 1
      # KGSL_MEMFLAGS_USE_CPU_MAP: the gpu address is the address the object gets mmap'd at, filled in by kgsl_mmap
      self.objs[st.id] = (st.mmapsize, 0)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE):
      if (obj:=self.objs.pop(kgsl.struct_kgsl_gpuobj_free.from_address(argp).id, None)) is not None and obj[1]:
        self.gpu.unmap_range(obj[1], obj[0])
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
      mm = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      mm.gpuaddr = mm.hostptr  # the mock gpu shares the host address space
      self.gpu.map_range(mm.hostptr, mm.len)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE): pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY): pass
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      gp = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if gp.type != kgsl.KGSL_PROP_DEVICE_INFO: raise RuntimeError(f"unsupported getproperty type {gp.type}")
      info = kgsl.struct_kgsl_devinfo.from_address(cast(int, gp.value))
      info.device_id, info.chip_id, info.gpu_id, info.mmu_enabled = 1, CHIP_ID, 630, 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      cmd = kgsl.struct_kgsl_gpu_command.from_address(argp)
      for i in range(cmd.numcmds):
        cobj = kgsl.struct_kgsl_command_object.from_address(cmd.cmdlist + i * cmd.cmdsize)
        if cobj.flags & kgsl.KGSL_CMDLIST_IB: self.gpu.execute(cobj.gpuaddr, cobj.size)
      self.timestamp += 1
      cmd.timestamp = self.timestamp
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID): pass  # commands complete synchronously in the mock
    else: raise RuntimeError(f"unsupported kgsl ioctl nr {nr:#x}")
    return 0
