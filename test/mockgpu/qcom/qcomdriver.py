import ctypes, functools, mmap, os
from dataclasses import dataclass
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc
from test.mockgpu.qcom.qcomgpu import QCOMGPU

IOCTLS = {value.args[2]: (name, value.args[3]) for name,value in vars(kgsl).items()
          if name.startswith('IOCTL_KGSL_') and isinstance(value, functools.partial)}

@dataclass
class Allocation:
  size:int
  pointer:int = 0

class QCOMFileDesc(VirtFileDesc):
  def __init__(self, fd:int, driver:'QCOMDriver'):
    super().__init__(fd)
    self.driver = driver
  def ioctl(self, fd, request, argp):
    self.raise_if_failed()
    return self.driver.ioctl(request, argp)
  def mmap(self, start, size, prot, flags, fd, offset):
    allocation = self.driver.allocations[offset//0x1000]
    if allocation.pointer or size != allocation.size: raise ValueError('Invalid QCOM allocation mapping')
    pointer = libc.mmap(0, size, prot, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    if pointer == ctypes.c_void_p(-1).value: raise OSError('QCOM mock mmap failed')
    allocation.pointer = pointer
    return pointer
  def close(self, fd): os.close(fd)

class QCOMDriver(VirtDriver):
  def __init__(self):
    super().__init__()
    self.tracked_files = [VirtFile('/dev/kgsl-3d0', functools.partial(QCOMFileDesc, driver=self))]
    self.allocations:dict[int, Allocation] = {}
    self.external:dict[int, int] = {}
    self.next_allocation = self.next_context = 1
    self.contexts:set[int] = set()
    self.gpu = QCOMGPU(self.ranges)
  def ranges(self) -> tuple[tuple[int, int], ...]:
    return tuple((a.pointer, a.size) for a in self.allocations.values() if a.pointer) + tuple(self.external.items())
  def open(self, name, flags, mode, virtfile):
    # An OS-allocated placeholder descriptor cannot collide with another mock driver's descriptor range.
    return virtfile.fdcls(os.open(os.devnull, os.O_RDWR))
  def ioctl(self, request:int, pointer:int) -> int:
    name, record_type = IOCTLS[request & 255]
    record = record_type.from_address(pointer)
    if name == 'IOCTL_KGSL_DRAWCTXT_CREATE':
      record.drawctxt_id = self.next_context
      self.contexts.add(self.next_context)
      self.next_context += 1
    elif name == 'IOCTL_KGSL_DRAWCTXT_DESTROY': self.contexts.remove(record.drawctxt_id)
    elif name == 'IOCTL_KGSL_DEVICE_GETPROPERTY':
      if record.type != kgsl.KGSL_PROP_DEVICE_INFO: raise ValueError(f'Unsupported QCOM property {record.type}')
      info = kgsl.struct_kgsl_devinfo.from_address(record.value)
      info.device_id, info.chip_id, info.gpu_id, info.gmem_sizebytes, info.mmu_enabled = 1, 0x06030001, 630, 1 << 20, 1
    elif name == 'IOCTL_KGSL_SETPROPERTY':
      if record.type != kgsl.KGSL_PROP_PWR_CONSTRAINT: raise ValueError(f'Unsupported QCOM property {record.type}')
    elif name == 'IOCTL_KGSL_GPUOBJ_ALLOC':
      record.id = self.next_allocation
      record.mmapsize = record.size
      self.allocations[record.id] = Allocation(record.size)
      self.next_allocation += 1
    elif name == 'IOCTL_KGSL_GPUOBJ_FREE': del self.allocations[record.id]
    elif name == 'IOCTL_KGSL_MAP_USER_MEM':
      record.gpuaddr = record.hostptr
      self.external[record.hostptr] = record.len
    elif name == 'IOCTL_KGSL_SHAREDMEM_FREE': del self.external[record.gpuaddr]
    elif name == 'IOCTL_KGSL_GPU_COMMAND':
      if record.context_id not in self.contexts: raise ValueError('Unknown QCOM context')
      for i in range(record.numcmds):
        command = kgsl.struct_kgsl_command_object.from_address(record.cmdlist+i*record.cmdsize)
        self.gpu.submit(command.gpuaddr+command.offset, command.size)
      record.timestamp = self.gpu.submissions
    elif name == 'IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID': record.timestamp = self.gpu.submissions
    elif name == 'IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID':
      if record.timestamp > self.gpu.submissions: raise RuntimeError('QCOM timestamp has not completed')
    else: raise ValueError(f'Unsupported QCOM request {name}')
    return 0
