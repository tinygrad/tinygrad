from __future__ import annotations
import ctypes, functools, mmap
from dataclasses import dataclass

from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc
from test.mockgpu.qcom.qcomgpu import A630GPU


_PAGE_SIZE, _U32_MAX, _U64_LIMIT = 0x1000, (1 << 32) - 1, 1 << 64
_MAX_COMMAND_OBJECTS, _MAX_COMMAND_BYTES = 1 << 20, 1 << 28


def _range_end(address:int, size:int, name:str) -> int:
  if address < 0 or address >= _U64_LIMIT or size <= 0 or size > _U64_LIMIT - address:
    raise ValueError(f"{name} overflows the 64-bit address space")
  return address + size


def _checked_product(lhs:int, rhs:int, name:str) -> int:
  if lhs < 0 or rhs < 0 or (rhs and lhs > _U64_LIMIT // rhs):
    raise ValueError(f"{name} overflows the 64-bit address space")
  return lhs * rhs


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
    self.next_fd, self.next_object_id, self.next_context_id = 1 << 30, 1, 1
    self.objects:dict[int, KGSLObject] = {}
    self.external_mappings:dict[int, list[int]] = {}
    self.contexts:set[int] = set()
    self.submitted_timestamps:dict[int, int] = {}
    self.submitted_timestamp = self.completed_timestamp = 0
    self.last_error:Exception|None = None
    self._executing = False

  def open(self, name, flags, mode, virtfile):
    fd = self.next_fd
    self.next_fd += 1
    return virtfile.fdcls(fd)

  def mmap(self, start:int, size:int, prot:int, flags:int, offset:int) -> int:
    if offset < 0 or offset >= _U64_LIMIT: raise ValueError(f"invalid KGSL mmap offset {offset}")
    if offset % _PAGE_SIZE: raise ValueError(f"KGSL mmap offset {offset:#x} is not page aligned")
    object_id = offset // _PAGE_SIZE
    if object_id not in self.objects: raise ValueError(f"unknown KGSL object id {object_id}")
    obj = self.objects[object_id]
    if obj.address is not None: raise ValueError(f"KGSL object {object_id} is already mapped")
    if size != obj.size: raise ValueError(f"invalid KGSL mmap size {size} for allocation {obj.size}")
    _range_end(0, size, "KGSL mmap size")
    address = libc.mmap(start, size, prot, (flags & ~mmap.MAP_SHARED) | mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    if address in {0, ctypes.c_void_p(-1).value}: raise OSError("anonymous KGSL mmap failed")
    mapped = False
    try:
      self.gpu.map_range(address, size)
      mapped = True
      self.track_address(address, address + size, lambda mv, index: None, lambda mv, index: self._emulate_execute())
    except Exception:
      if mapped: self.gpu.unmap_range(address, size)
      libc.munmap(address, size)
      raise
    obj.address, obj.mapped_size = address, size
    return address

  def _emulate_execute(self):
    if self._executing: return
    self._executing = True
    try: self.completed_timestamp = self.gpu.progress()
    finally: self._executing = False

  def _tracked_address_index(self, address:int, size:int) -> int:
    for index in range(len(self.tracked_addresses) - 1, -1, -1):
      start, end, _, _ = self.tracked_addresses[index]
      if start == address and end == address + size: return index
    raise RuntimeError(f"KGSL mapping {address:#x}..{address + size:#x} was not tracked")

  def _untrack_address(self, address:int, size:int): self.tracked_addresses.pop(self._tracked_address_index(address, size))

  def _free_mapping(self, address:int, size:int):
    # Look up the tracker before changing the GPU mapping so a malformed free has no side effects.
    tracked_index = self._tracked_address_index(address, size)
    self.gpu.unmap_range(address, size)
    self.tracked_addresses.pop(tracked_index)

  def ioctl(self, request:int, argp:int) -> int:
    number = request & 0xff
    if number == 0x02:
      getproperty = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if getproperty.type != kgsl.KGSL_PROP_DEVICE_INFO: raise NotImplementedError(f"unsupported KGSL property {getproperty.type}")
      if getproperty.sizebytes < ctypes.sizeof(kgsl.struct_kgsl_devinfo): raise ValueError("KGSL device-info buffer is too small")
      if (info_address:=ctypes.cast(getproperty.value, ctypes.c_void_p).value) is None: raise ValueError("KGSL device-info pointer is null")
      info = kgsl.struct_kgsl_devinfo.from_address(info_address)
      info.device_id, info.chip_id, info.mmu_enabled = 0, self.CHIP_ID, 1
      info.gpu_id, info.gmem_gpubaseaddr, info.gmem_sizebytes = 630, 0, 0x100000
    elif number == 0x07:
      wait = kgsl.struct_kgsl_device_waittimestamp_ctxtid.from_address(argp)
      if wait.context_id not in self.contexts: raise ValueError(f"unknown KGSL context {wait.context_id}")
      try: self._emulate_execute()
      except Exception as exc:
        self.last_error = exc
        raise
      if not self.gpu.is_timestamp_complete(wait.context_id, wait.timestamp):
        if wait.timestamp > self.submitted_timestamps[wait.context_id]:
          raise ValueError(f"KGSL timestamp {wait.timestamp} was not submitted in context {wait.context_id}")
        if wait.timeout == 0xffffffff: return 0
        raise TimeoutError(f"KGSL context {wait.context_id} is blocked before timestamp {wait.timestamp}")
    elif number == 0x13:
      context = kgsl.struct_kgsl_drawctxt_create.from_address(argp)
      if self.next_context_id > _U32_MAX: raise RuntimeError("KGSL context ids are exhausted")
      context_id = self.next_context_id
      self.contexts.add(context_id)
      self.submitted_timestamps[context_id] = 0
      context.drawctxt_id = context_id
      self.next_context_id += 1
    elif number == 0x14:
      destroy = kgsl.struct_kgsl_drawctxt_destroy.from_address(argp)
      if destroy.drawctxt_id not in self.contexts: raise ValueError(f"unknown KGSL context {destroy.drawctxt_id}")
      self.gpu.drop_context(destroy.drawctxt_id)
      self.contexts.remove(destroy.drawctxt_id)
      del self.submitted_timestamps[destroy.drawctxt_id]
      self.submitted_timestamp = max(self.submitted_timestamps.values(), default=0)
      self.completed_timestamp = self.gpu.completed_timestamp
    elif number == 0x15:
      usermap = kgsl.struct_kgsl_map_user_mem.from_address(argp)
      if usermap.hostptr == 0 or usermap.len == 0: raise ValueError("KGSL user mapping requires a non-empty host range")
      _range_end(usermap.hostptr, usermap.len, "KGSL user mapping")
      self.gpu.map_range(usermap.hostptr, usermap.len)
      try:
        self.track_address(usermap.hostptr, usermap.hostptr + usermap.len,
                           lambda mv, index: None, lambda mv, index: self._emulate_execute())
      except Exception:
        self.gpu.unmap_range(usermap.hostptr, usermap.len)
        raise
      self.external_mappings.setdefault(usermap.hostptr, []).append(usermap.len)
      usermap.gpuaddr = usermap.hostptr
    elif number == 0x21:
      shared = kgsl.struct_kgsl_sharedmem_free.from_address(argp)
      if not (sizes:=self.external_mappings.get(shared.gpuaddr)):
        raise ValueError(f"unknown KGSL external mapping {shared.gpuaddr:#x}")
      size = sizes[-1]
      self._free_mapping(shared.gpuaddr, size)
      sizes.pop()
      if not sizes: del self.external_mappings[shared.gpuaddr]
    elif number == 0x32:
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
    elif number == 0x45:
      allocation = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      if allocation.size == 0: raise ValueError("KGSL allocation size must be positive")
      allocation_size = (allocation.size + _PAGE_SIZE - 1) & ~(_PAGE_SIZE - 1)
      if allocation_size > 0xffffffff: raise ValueError(f"KGSL allocation size {allocation.size} exceeds the A630 limit after page alignment")
      if self.next_object_id > _U32_MAX: raise RuntimeError("KGSL object ids are exhausted")
      object_id = self.next_object_id
      self.objects[object_id] = KGSLObject(allocation_size)
      try: allocation.size, allocation.id, allocation.mmapsize = allocation_size, object_id, allocation_size
      except Exception:
        del self.objects[object_id]
        raise
      self.next_object_id += 1
    elif number == 0x46:
      free = kgsl.struct_kgsl_gpuobj_free.from_address(argp)
      object_id = free.id
      if object_id not in self.objects: raise ValueError(f"unknown KGSL object id {object_id}")
      obj = self.objects[object_id]
      if obj.address is not None:
        self._free_mapping(obj.address, obj.mapped_size)
      del self.objects[object_id]
    elif number == 0x4a:
      submission = kgsl.struct_kgsl_gpu_command.from_address(argp)
      if submission.context_id not in self.contexts: raise ValueError(f"unknown KGSL context {submission.context_id}")
      if submission.cmdsize < ctypes.sizeof(kgsl.struct_kgsl_command_object): raise ValueError("KGSL command object stride is too small")
      if submission.cmdsize % ctypes.alignment(ctypes.c_uint64): raise ValueError("KGSL command object stride is not naturally aligned")
      if submission.numcmds > _MAX_COMMAND_OBJECTS: raise ValueError(f"KGSL command list has too many objects: {submission.numcmds}")
      if submission.numcmds and submission.cmdlist == 0: raise ValueError("KGSL command list pointer is null")
      if submission.numcmds:
        command_list_size = _checked_product(submission.numcmds, submission.cmdsize, "KGSL command list")
        _range_end(submission.cmdlist, command_list_size, "KGSL command list")
      words:list[int] = []
      for index in range(submission.numcmds):
        command = kgsl.struct_kgsl_command_object.from_address(submission.cmdlist + index * submission.cmdsize)
        if command.size == 0: raise ValueError("KGSL command object is empty")
        if command.size % 4: raise ValueError(f"A630 command buffer size must be dword aligned, got {command.size}")
        if command.size > _MAX_COMMAND_BYTES: raise ValueError(f"A630 command buffer is too large: {command.size} bytes")
        if len(words) > (_MAX_COMMAND_BYTES - command.size) // 4:
          raise ValueError("A630 command list is too large")
        address = _range_end(command.gpuaddr, command.offset + 1, "KGSL command object") - 1
        if address % 4: raise ValueError(f"A630 command buffer address must be dword aligned, got {address:#x}")
        self.gpu._validate_memory(address, command.size)
        command_words = list((ctypes.c_uint32 * (command.size // 4)).from_address(address))
        self.gpu.validate_words(command_words)
        words.extend(command_words)
      if self.submitted_timestamps[submission.context_id] >= _U32_MAX:
        raise RuntimeError(f"KGSL timestamps are exhausted in context {submission.context_id}")
      timestamp = self.submitted_timestamps[submission.context_id] + 1
      self.gpu.submit(words, timestamp, submission.context_id)
      self.submitted_timestamps[submission.context_id] = timestamp
      self.submitted_timestamp = max(self.submitted_timestamp, timestamp)
      submission.timestamp = timestamp
      try: self._emulate_execute()
      except Exception as exc:
        self.last_error = exc
        raise
    else: raise NotImplementedError(f"unsupported KGSL ioctl number {number:#x}")
    return 0
