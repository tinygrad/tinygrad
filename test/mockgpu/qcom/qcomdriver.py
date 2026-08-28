from __future__ import annotations
import ctypes, functools, mmap
from dataclasses import dataclass

from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc, TextFileDesc
from test.mockgpu.qcom.qcomgpu import A630GPU

def _kgsl_nr(fn) -> int: return fn.args[2]


class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd:int, driver:QCOMDriver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp): return self.driver.ioctl(request, argp)
  def mmap(self, start, size, prot, flags, fd, offset): return self.driver.mmap(start, size, prot, flags, offset)


@dataclass
class KGSLObject:
  size: int
  address: int|None = None
  mapped_size: int = 0


class QCOMDriver(VirtDriver):
  CHIP_ID = 0x06030001

  def __init__(self):
    super().__init__()
    self.gpu = A630GPU(0)
    self.tracked_files.append(VirtFile("/dev/kgsl-3d0", functools.partial(KGSLFileDesc, driver=self)))
    self.tracked_files.append(VirtFile("/sys/class/kgsl/kgsl-3d0/idle_timer", functools.partial(TextFileDesc, text="10000\n")))
    self.next_fd, self.next_object_id, self.next_context_id = 1 << 30, 1, 1
    self.objects:dict[int, KGSLObject] = {}
    self.external_mappings:dict[int, list[int]] = {}
    self.contexts:set[int] = set()
    self.submitted_timestamps:dict[int, int] = {}
    self.submitted_timestamp = self.completed_timestamp = 0
    self._executing = False

  def open(self, name, flags, mode, virtfile):
    fd = self.next_fd
    self.next_fd += 1
    return virtfile.fdcls(fd)

  def mmap(self, start:int, size:int, prot:int, flags:int, offset:int) -> int:
    if offset % 0x1000: raise ValueError(f"KGSL mmap offset {offset:#x} is not page aligned")
    object_id = offset // 0x1000
    if object_id not in self.objects: raise ValueError(f"unknown KGSL object id {object_id}")
    obj = self.objects[object_id]
    if obj.address is not None: raise ValueError(f"KGSL object {object_id} is already mapped")
    if size != obj.size: raise ValueError(f"invalid KGSL mmap size {size} for allocation {obj.size}")
    address = libc.mmap(start, size, prot, (flags & ~mmap.MAP_SHARED) | mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    if address in {0, ctypes.c_void_p(-1).value}: raise OSError("anonymous KGSL mmap failed")
    obj.address, obj.mapped_size = address, size
    self.gpu.map_range(address, size)
    self.track_address(address, address + size, lambda mv, index: None, lambda mv, index: self._emulate_execute())
    return address

  def _emulate_execute(self):
    if self._executing: return
    self._executing = True
    try: self.completed_timestamp = self.gpu.progress()
    finally: self._executing = False

  def _untrack_address(self, address:int, size:int):
    for index in range(len(self.tracked_addresses) - 1, -1, -1):
      start, end, _, _ = self.tracked_addresses[index]
      if start == address and end == address + size:
        self.tracked_addresses.pop(index)
        return
    raise RuntimeError(f"KGSL mapping {address:#x}..{address + size:#x} was not tracked")

  def ioctl(self, request:int, argp:int) -> int:
    number = request & 0xff
    if number == _kgsl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      getproperty = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if getproperty.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f"unsupported KGSL property {getproperty.type}")
      if getproperty.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_devinfo): raise ValueError("KGSL device-info buffer is too small")
      if (info_address:=ctypes.cast(getproperty.value, ctypes.c_void_p).value) is None: raise ValueError("KGSL device-info pointer is null")
      info = kgsl.struct_kgsl_devinfo.from_address(info_address)
      info.device_id, info.chip_id, info.mmu_enabled = 0, self.CHIP_ID, 1
      info.gpu_id, info.gmem_gpubaseaddr, info.gmem_sizebytes = 630, 0, 0x100000
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID):
      wait = kgsl.struct_kgsl_device_waittimestamp_ctxtid.from_address(argp)
      if wait.context_id not in self.contexts: raise ValueError(f"unknown KGSL context {wait.context_id}")
      self._emulate_execute()
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
      context = kgsl.struct_kgsl_drawctxt_create.from_address(argp)
      context.drawctxt_id = self.next_context_id
      self.contexts.add(context.drawctxt_id)
      self.submitted_timestamps[context.drawctxt_id] = 0
      self.next_context_id += 1
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY):
      destroy = kgsl.struct_kgsl_drawctxt_destroy.from_address(argp)
      if destroy.drawctxt_id not in self.contexts: raise ValueError(f"unknown KGSL context {destroy.drawctxt_id}")
      self.contexts.remove(destroy.drawctxt_id)
      del self.submitted_timestamps[destroy.drawctxt_id]
      self.gpu.pending[:] = [stream for stream in self.gpu.pending if stream.context_id != destroy.drawctxt_id]
      self.gpu.context_states.pop(destroy.drawctxt_id, None)
      self.gpu.completed_timestamps.pop(destroy.drawctxt_id, None)
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_MAP_USER_MEM):
      usermap = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      if usermap.hostptr == 0 or usermap.len == 0: raise ValueError("KGSL user mapping requires a non-empty host range")
      if usermap.hostptr + usermap.len > 1 << 64: raise ValueError("KGSL user mapping overflows the 64-bit address space")
      usermap.gpuaddr = usermap.hostptr
      self.external_mappings.setdefault(usermap.gpuaddr, []).append(usermap.len)
      self.gpu.map_range(usermap.gpuaddr, usermap.len)
      self.track_address(usermap.gpuaddr, usermap.gpuaddr + usermap.len,
                         lambda mv, index: None, lambda mv, index: self._emulate_execute())
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_SHAREDMEM_FREE):
      shared = kgsl.struct_kgsl_sharedmem_free.from_address(argp)
      if not (sizes:=self.external_mappings.get(shared.gpuaddr)):
        raise ValueError(f"unknown KGSL external mapping {shared.gpuaddr:#x}")
      size = sizes.pop()
      if not sizes: del self.external_mappings[shared.gpuaddr]
      self.gpu.unmap_range(shared.gpuaddr, size)
      self._untrack_address(shared.gpuaddr, size)
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_SETPROPERTY):
      setproperty = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if setproperty.type != kgsl.KGSL_PROP_PWR_CONSTRAINT: raise NotImplementedError(f"unsupported KGSL property {setproperty.type}")
      if setproperty.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_device_constraint): raise ValueError("KGSL power constraint is too small")
      if (constraint_address:=ctypes.cast(setproperty.value, ctypes.c_void_p).value) is None: raise ValueError("KGSL constraint pointer is null")
      constraint = kgsl.struct_kgsl_device_constraint.from_address(constraint_address)
      if constraint.context_id not in self.contexts: raise ValueError(f"unknown KGSL context {constraint.context_id}")
      if constraint.type != kgsl.KGSL_CONSTRAINT_PWRLEVEL: raise NotImplementedError(f"unsupported KGSL constraint {constraint.type}")
      if (power_address:=ctypes.cast(constraint.data, ctypes.c_void_p).value) is None or \
         constraint.size < ctypes.sizeof(kgsl.struct_kgsl_device_constraint_pwrlevel):
        raise ValueError("KGSL power-level constraint has no value")
      power = kgsl.struct_kgsl_device_constraint_pwrlevel.from_address(power_address)
      if power.level not in {kgsl.KGSL_CONSTRAINT_PWR_MIN, kgsl.KGSL_CONSTRAINT_PWR_MAX}:
        raise ValueError(f"invalid KGSL power level {power.level}")
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      allocation = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      if allocation.size == 0: raise ValueError("KGSL allocation size must be positive")
      allocation_size = (allocation.size + 0xfff) & ~0xfff
      if allocation_size > 0xffffffff: raise ValueError(f"KGSL allocation size {allocation.size} exceeds the A630 limit after page alignment")
      object_id = self.next_object_id
      self.next_object_id += 1
      allocation.size, allocation.id, allocation.mmapsize = allocation_size, object_id, allocation_size
      self.objects[object_id] = KGSLObject(allocation_size)
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE):
      free = kgsl.struct_kgsl_gpuobj_free.from_address(argp)
      if free.id not in self.objects: raise ValueError(f"unknown KGSL object id {free.id}")
      obj = self.objects.pop(free.id)
      if obj.address is not None:
        self.gpu.unmap_range(obj.address, obj.mapped_size)
        self._untrack_address(obj.address, obj.mapped_size)
    elif number == _kgsl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      submission = kgsl.struct_kgsl_gpu_command.from_address(argp)
      if submission.context_id not in self.contexts: raise ValueError(f"unknown KGSL context {submission.context_id}")
      if submission.cmdsize < ctypes.sizeof(kgsl.struct_kgsl_command_object): raise ValueError("KGSL command object stride is too small")
      if submission.numcmds and submission.cmdlist == 0: raise ValueError("KGSL command list pointer is null")
      words:list[int] = []
      for index in range(submission.numcmds):
        command = kgsl.struct_kgsl_command_object.from_address(submission.cmdlist + index * submission.cmdsize)
        if command.size == 0: raise ValueError("KGSL command object is empty")
        if command.size % 4: raise ValueError(f"A630 command buffer size must be dword aligned, got {command.size}")
        address = command.gpuaddr + command.offset
        self.gpu._validate_memory(address, command.size)
        command_words = list((ctypes.c_uint32 * (command.size // 4)).from_address(address))
        self.gpu.validate_words(command_words)
        words.extend(command_words)
      timestamp = self.submitted_timestamps[submission.context_id] + 1
      self.gpu.submit(words, timestamp, submission.context_id)
      self.submitted_timestamps[submission.context_id] = timestamp
      self.submitted_timestamp = max(self.submitted_timestamp, timestamp)
      submission.timestamp = timestamp
      self._emulate_execute()
    else: raise NotImplementedError(f"unsupported KGSL ioctl number {number:#x}")
    return 0
