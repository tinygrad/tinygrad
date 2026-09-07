"""The KGSL boundary for the A630 mock; PM4 runs against staged, mapped memory."""
from __future__ import annotations
import collections, ctypes, dataclasses, functools, mmap, os, struct, sys, threading, time
from typing import Callable
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile

# Read from the outside inward:
#   native ioctl -> QCOMDriver -> Context queue -> qcomgpu.execute_command
#   PM4 actions / IR3 lanes -> MemoryTransaction -> AddressSpace -> host memory
# The driver owns resource lifetimes and queue progress. The PM4 layer interprets
# commands; the IR3 layer advances shader lanes. Only transaction.commit publishes
# their staged memory effects back to the host program.


# Host transport: copy bytes across the native ABI without using a guest address
# as an unchecked Python pointer. Both read and write select endpoints for the
# same kernel copy operation; the local bytearray remains alive for that call.

class IOVec(ctypes.Structure):
  _fields_ = [("base", ctypes.c_void_p), ("size", ctypes.c_size_t)]


def host_transfer(address:int, data:bytearray, write:bool):
  # Kernel copy APIs reject stale/unmapped host pointers instead of dereferencing
  # them in Python. Guest addresses additionally need an AddressSpace mapping.
  if address <= 0 or len(data) > 1 << 30 or address + len(data) > 1 << 64:
    raise ValueError("invalid host memory span")
  if not data:
    return
  local = (ctypes.c_ubyte * len(data)).from_buffer(data)
  source, destination = (ctypes.addressof(local), address) if write else (address, ctypes.addressof(local))
  if sys.platform == "darwin":
    task, copied = ctypes.c_uint.in_dll(libc.dll, "mach_task_self_").value, ctypes.c_uint64()
    result = libc.dll.mach_vm_read_overwrite(ctypes.c_uint(task), ctypes.c_uint64(source), ctypes.c_uint64(len(data)),
                                             ctypes.c_uint64(destination), ctypes.byref(copied))
    complete = result == 0 and copied.value == len(data)
  elif sys.platform == "linux":
    source_vec, destination_vec = IOVec(source, len(data)), IOVec(destination, len(data))
    libc.dll.process_vm_readv.restype = ctypes.c_ssize_t
    copied_bytes = libc.dll.process_vm_readv(ctypes.c_int(os.getpid()), ctypes.byref(destination_vec), ctypes.c_ulong(1),
                                           ctypes.byref(source_vec), ctypes.c_ulong(1), ctypes.c_ulong(0))
    complete = copied_bytes == len(data)
  else:
    raise NotImplementedError("QCOM mock host memory needs Linux or macOS")
  if not complete:
    raise ValueError(f"host memory {'write' if write else 'read'} failed at {address:#x}")


def read_host(address:int, size:int) -> bytes:
  if not 0 <= size <= 1 << 30:
    raise ValueError("invalid host memory size")
  data = bytearray(size)
  host_transfer(address, data, False)
  return bytes(data)


def write_host(address:int, data:bytes|bytearray):
  host_transfer(address, bytearray(data), True)


def validate_host_mapping(address:int, size:int, write:bool=True):
  # Walk the complete requested interval, not just its first page. This checks
  # current host permissions; AddressSpace separately checks guest registration.
  end, cursor = address + size, address
  if sys.platform == "linux":
    with open("/proc/self/maps") as maps:
      for line in maps:
        extent, permissions = line.split()[:2]
        start, stop = (int(part, 16) for part in extent.split("-"))
        if start <= cursor < stop and permissions[0] == "r" and (not write or permissions[1] == "w"):
          cursor = min(stop, end)
        if cursor == end:
          return
  elif sys.platform == "darwin":
    task = ctypes.c_uint.in_dll(libc.dll, "mach_task_self_").value
    while cursor < end:
      start, length, count = ctypes.c_uint64(cursor), ctypes.c_uint64(), ctypes.c_uint(9)
      info, object_name = (ctypes.c_int * 9)(), ctypes.c_uint()
      result = libc.dll.mach_vm_region(ctypes.c_uint(task), ctypes.byref(start), ctypes.byref(length), ctypes.c_int(9),
                                        info, ctypes.byref(count), ctypes.byref(object_name))
      if object_name.value:
        libc.dll.mach_port_deallocate(ctypes.c_uint(task), object_name)
      required = 3 if write else 1
      if result or start.value > cursor or not length.value or info[0] & required != required:
        break
      cursor = min(start.value + length.value, end)
    if cursor == end:
      return
  raise ValueError("GPU access needs a live host range with the requested permissions")


@dataclasses.dataclass
class Mapping:
  # Address/allocation indexes share one record; owner_fd names its descriptor.
  # Object identity matters: replacing a mapping at the same address must not
  # silently revive a transaction that observed the earlier allocation.
  base:int
  size:int
  owner:object # None means borrowed storage; True denotes a native allocation.
  owner_fd:int|None=None


class AddressSpace:
  # Guest registration retains an optional owner but does not allocate or pin
  # host storage. Host permissions and validity still need separate checks.
  def __init__(self):
    self.mappings:dict[int, Mapping] = {}

  def map(self, base:int, size:int, owner:object=None, owner_fd:int|None=None) -> Mapping:
    if base <= 0 or not 0 < size <= 1 << 30 or base + size > 1 << 64:
      raise ValueError("invalid GPU mapping")
    if any(base < entry.base + entry.size and entry.base < base + size for entry in self.mappings.values()):
      raise ValueError("overlapping GPU mapping")
    validate_host_mapping(base, size)
    self.mappings[base] = entry = Mapping(base, size, owner, owner_fd)
    return entry

  def unmap(self, base:int):
    if base not in self.mappings:
      raise ValueError("unknown GPU mapping")
    del self.mappings[base]

  def mapping(self, address:int, size:int) -> Mapping:
    if size < 0:
      raise ValueError("negative GPU access size")
    for entry in self.mappings.values():
      if entry.base <= address and address + size <= entry.base + entry.size:
        return entry
    raise ValueError(f"unmapped GPU memory span {address:#x}+{size}")

  def transaction(self) -> MemoryTransaction:
    return MemoryTransaction(self)


class MemoryTransaction:
  # A transaction is one scheduler segment's working memory. Small writes use
  # ordered overlays; IR3 requests a whole-region snapshot for NumPy views.
  # Reads remain live until that region is snapshotted, so a later queue pass can
  # observe a signal written by another context instead of retaining an old wait.
  def __init__(self, space:AddressSpace):
    self.space = space
    self.snapshots:dict[int, bytearray] = {}
    self.overlays:list[tuple[int, bytes]] = []
    self.entries:dict[int, Mapping] = {}
    self.finished = False

  def mapping(self, address:int, size:int) -> Mapping:
    if self.finished:
      raise ValueError("transaction already committed")
    entry = self.space.mapping(address, size)
    # Keep the first allocation identity even after a later lookup. Old overlays
    # and snapshots belong to that record, so reject replacement before exposing
    # cached data or staging any further access against a new allocation.
    if self.entries.setdefault(entry.base, entry) is not entry:
      raise ValueError("GPU mapping changed during transaction")
    return entry

  def read(self, address:int, size:int) -> bytes:
    entry = self.mapping(address, size)
    if entry.base in self.snapshots:
      return bytes(self.snapshots[entry.base][address-entry.base:address-entry.base+size])
    data = bytearray(read_host(address, size))
    for start, payload in self.overlays:
      low, high = max(address, start), min(address + size, start + len(payload))
      if low < high:
        data[low-address:high-address] = payload[low-start:high-start]
    return bytes(data)

  def validate(self, address:int, size:int, write:bool=False):
    self.mapping(address, size)
    validate_host_mapping(address, size, write)

  def write(self, address:int, data:bytes|bytearray):
    entry, payload = self.mapping(address, len(data)), bytes(data)
    if entry.base in self.snapshots:
      self.snapshots[entry.base][address-entry.base:address-entry.base+len(payload)] = payload
    else:
      self.overlays.append((address, payload))

  def region(self, address:int, size:int) -> tuple[int, bytearray]:
    # Fold earlier overlays into the snapshot once. The returned bytearray owns
    # the storage behind IR3 views and stays alive until this segment finishes.
    entry = self.mapping(address, size)
    if entry.base not in self.snapshots:
      self.snapshots[entry.base] = bytearray(self.read(entry.base, entry.size))
      self.overlays = [(start, data) for start, data in self.overlays if not entry.base <= start < entry.base + entry.size]
    return entry.base, self.snapshots[entry.base]

  def commit(self):
    # Validate every retained mapping identity and write destination before the
    # first copy. Publication itself is a sequence of host writes, not an OS-wide
    # atomic transaction; a host mapping race can still make a later copy fail.
    if self.finished:
      raise ValueError("transaction already committed")
    for base, entry in self.entries.items():
      if self.space.mappings.get(base) is not entry:
        raise ValueError("GPU mapping changed during transaction")
      if entry.owner is None:
        validate_host_mapping(base, entry.size)
    writes = [*self.snapshots.items(), *self.overlays]
    for address, data in writes:
      validate_host_mapping(address, len(data), write=True)
    for address, data in writes:
      write_host(address, data)
    self.finished = True


@dataclasses.dataclass
class Context:
  # queued advances on acceptance. consumed records the front command's first
  # successful scheduler segment, including a wait at cursor zero; retired waits
  # for its final action. pending holds (words, timestamp, continuation cursor).
  owner_fd:int
  queued:int=0
  consumed:int=0
  retired:int=0
  pending:collections.deque=dataclasses.field(default_factory=collections.deque)


class QCOMFileDesc(VirtFileDesc):
  # Adapt the generic mock-file interface to one shared driver instance. The
  # descriptor is the ownership key used when locating contexts and allocations.
  def __init__(self, fd:int, driver:QCOMDriver):
    super().__init__(fd)
    self.driver = driver

  def ioctl(self, fd, request, argp):
    return self.driver.ioctl(fd, request, argp)
  def mmap(self, start, size, prot, flags, fd, offset):
    return self.driver.mmap(fd, start, size, prot, flags, offset)
  def close(self, fd):
    self.driver.close(fd)


def ioctl_number(fn):
  direction, group, number, typ = fn.args
  return direction << 30 | ctypes.sizeof(typ) << 16 | group << 8 | number


# Decode request numbers from the existing generated ABI. The table selects a
# request name and structure type; ioctl below still validates its allowed fields.
IOCTLS = {
  ioctl_number(fn): (name.removeprefix("IOCTL_KGSL_"), fn.args[3])
  for name, fn in vars(kgsl).items()
  if name.startswith("IOCTL_KGSL_") and isinstance(fn, functools.partial) and fn.args[3] is not None
}
CONTEXT_FLAGS = (kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE |
                 kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC | kgsl.KGSL_CONTEXT_PRIORITY_MASK | kgsl.KGSL_CONTEXT_PREEMPT_STYLE_MASK)
ALLOCATION_FLAGS = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | kgsl.KGSL_MEMALIGN_MASK | kgsl.KGSL_CACHEMODE_MASK


def read_struct(address:int, typ, writable:bool=False):
  alignment = max(ctypes.alignment(field[1]) for field in typ._real_fields_)
  if address % alignment:
    raise ValueError("misaligned KGSL structure")
  if writable:
    validate_host_mapping(address, ctypes.sizeof(typ), write=True)
  return typ.from_buffer_copy(read_host(address, ctypes.sizeof(typ)))


class QCOMDriver(VirtDriver):
  # Control plane: serialize descriptor/resource operations and queue execution.
  # allocations and borrowed select the same Mapping representation, but only
  # allocations may release driver-owned native storage when a descriptor closes.
  def __init__(self, execute:Callable|None=None):
    super().__init__()
    self.tracked_files = [VirtFile("/dev/kgsl-3d0", QCOMFileDesc)]
    self.memory, self.execute = AddressSpace(), execute
    self.allocations:dict[int, Mapping] = {}
    self.borrowed:dict[int, Mapping] = {}
    self.contexts:dict[int, Context] = {}
    self.next_fd, self.next_id, self.next_context = 1 << 28, 1, 1
    self.lock, self.executing = threading.RLock(), False
    self.error:Exception|None = None

  def open(self, name, flags, mode, virtfile):
    with self.lock:
      fd = QCOMFileDesc(self.next_fd, self)
      self.next_fd += 1
      return fd

  def check_error(self):
    # A ctypes callback cannot propagate a Python exception through native HCQ2.
    # make_ioctl_callback saves it; the Python execution boundary re-raises it.
    if self.error is not None:
      raise RuntimeError(f"QCOM mock submission failed: {self.error}") from self.error

  def context(self, fd:int, context_id:int) -> Context:
    if context_id not in self.contexts or self.contexts[context_id].owner_fd != fd:
      raise ValueError("unknown KGSL context")
    return self.contexts[context_id]

  def mmap(self, fd, start, size, prot, flags, offset):
    # Allocation reservation and native mapping are separate lifetime steps.
    # On registration failure, release the new host mapping before propagating.
    with self.lock:
      self.check_error()
      allocation = self.allocations.get(offset // 4096)
      if start or offset % 4096 or allocation is None or allocation.owner_fd != fd or allocation.base or size != allocation.size:
        raise ValueError("invalid KGSL mmap")
      if prot != mmap.PROT_READ | mmap.PROT_WRITE or flags != mmap.MAP_SHARED:
        raise ValueError("unsupported KGSL mmap flags")
      address = libc.mmap(0, size, prot, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE, -1, 0)
      if address == ctypes.c_void_p(-1).value:
        raise MemoryError("QCOM mock mmap failed")
      try:
        entry = self.memory.map(address, size, owner=True, owner_fd=fd)
      except Exception:
        libc.munmap(address, size)
        raise
      self.allocations[offset // 4096] = entry
      return address

  def close(self, fd:int):
    # Closing discards this descriptor's contexts and unregisters its resources.
    # Borrowed CPU storage survives; native allocations owned here are unmapped.
    with self.lock:
      self.contexts = {key: context for key, context in self.contexts.items() if context.owner_fd != fd}
      for resources, owned in ((self.borrowed, False), (self.allocations, True)):
        for key, allocation in list(resources.items()):
          if allocation.owner_fd != fd:
            continue
          if allocation.base:
            self.memory.unmap(allocation.base)
            if owned:
              libc.munmap(allocation.base, allocation.size)
          del resources[key]

  def drain(self):
    # Cooperative scheduling: advance each context's front command until a full
    # pass makes no progress. A wait yields at its cursor; another context may
    # publish the signal that lets it resume. The re-entry latch prevents nested
    # callbacks from starting a second scheduler over these same queues.
    if self.executing:
      return
    self.executing = True
    try:
      progress = True
      while progress:
        progress = False
        for context in self.contexts.values():
          if not context.pending:
            continue
          words, timestamp, cursor = context.pending[0]
          transaction = self.memory.transaction()
          if self.execute is None:
            from test.mockgpu.qcom.qcomgpu import execute_command
            self.execute = execute_command
          complete, next_cursor = self.execute(words, transaction, time.monotonic_ns() * 192 // 10000, cursor)
          if not isinstance(complete, bool) or not isinstance(next_cursor, int) or next_cursor < cursor:
            raise ValueError("invalid PM4 execution progress")
          # A valid prefix's signals must be visible to other contexts before a
          # later wait can resume. Rejected segments never reach this commit.
          transaction.commit()
          # Start-of-pipeline is observed at successful segment publication in
          # this synchronous mock, not at a physical GPU cycle. A planning or
          # commit rejection publishes no new start; retries retain the same one.
          context.consumed = timestamp
          if complete:
            context.pending.popleft()
            context.retired, progress = timestamp, True
          else:
            context.pending[0] = words, timestamp, next_cursor
            progress |= next_cursor != cursor
    finally:
      self.executing = False

  def ioctl(self, fd:int, request:int, argp:int):
    # Copy the caller's ABI structure, select a supported transition, then return
    # its updated fields. Validation stays beside the state/effect it protects.
    # Read the branches in groups: contexts, device properties, memory lifetime,
    # command submission, and completion observation.
    with self.lock:
      self.check_error()
      if request not in IOCTLS:
        raise ValueError(f"unknown KGSL ioctl {request:#x}")
      name, typ = IOCTLS[request]
      req = read_struct(argp, typ, writable=True)
      if name == "DRAWCTXT_CREATE":
        if req.flags & ~CONTEXT_FLAGS or self.next_context > 0xffffffff:
          raise ValueError("unsupported KGSL context")
        req.drawctxt_id = self.next_context
        self.contexts[self.next_context] = Context(fd)
        self.next_context += 1
      elif name == "DRAWCTXT_DESTROY":
        if self.context(fd, req.drawctxt_id).pending:
          raise ValueError("KGSL context has pending commands")
        del self.contexts[req.drawctxt_id]
      elif name == "DEVICE_GETPROPERTY":
        # Device identity and the narrow power contract expected by QCOMDevice.
        if req.type != kgsl.KGSL_PROP_DEVICE_INFO or req.sizebytes != ctypes.sizeof(kgsl.struct_kgsl_devinfo):
          raise ValueError("unsupported KGSL property")
        write_host(req.value, bytes(kgsl.struct_kgsl_devinfo(chip_id=0x06030000, gpu_id=630, mmu_enabled=1)))
      elif name == "SETPROPERTY":
        if req.type != kgsl.KGSL_PROP_PWR_CONSTRAINT or req.sizebytes != 24:
          raise ValueError("unsupported KGSL power property")
        kind, context_id, power, power_size = struct.unpack("=IIQQ", read_host(req.value, 24))
        self.context(fd, context_id)
        if kind != 1 or power_size != 4 or struct.unpack("=I", read_host(power, 4))[0] != 1:
          raise ValueError("unsupported KGSL power constraint")
      elif name == "GPUOBJ_ALLOC":
        # Reserve metadata now; mmap supplies the native address later.
        if not 0 < req.size <= 1 << 30 or req.size % 4096 or req.mmapsize != req.size or req.metadata_len:
          raise ValueError("unsupported KGSL allocation")
        if req.flags & ~ALLOCATION_FLAGS or not req.flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP:
          raise ValueError("KGSL mock needs supported CPU mapped allocation flags")
        if (req.flags & kgsl.KGSL_MEMALIGN_MASK) >> kgsl.KGSL_MEMALIGN_SHIFT > 12:
          raise ValueError("KGSL mock allocations guarantee at most 4096-byte alignment")
        if req.va_len or req.metadata or self.next_id > 0xffffffff:
          raise ValueError("unsupported KGSL allocation metadata")
        req.id = self.next_id
        self.allocations[req.id] = Mapping(0, req.size, owner=True, owner_fd=fd)
        self.next_id += 1
      elif name == "MAP_USER_MEM":
        # Borrow a caller-owned range; registration does not transfer ownership.
        if req.memtype != kgsl.KGSL_USER_MEM_TYPE_ADDR or req.offset or req.flags:
          raise ValueError("unsupported KGSL user mapping")
        self.borrowed[req.hostptr] = self.memory.map(req.hostptr, req.len, owner_fd=fd)
        req.gpuaddr = req.hostptr
      elif name in ("GPUOBJ_FREE", "SHAREDMEM_FREE"):
        # Remove guest visibility only when no queued work can still reference it.
        owned = name == "GPUOBJ_FREE"
        resources, key = (self.allocations, req.id) if owned else (self.borrowed, req.gpuaddr)
        allocation = resources.get(key)
        if allocation is None or allocation.owner_fd != fd or (owned and (req.flags or req.priv or req.type or req.len)):
          raise ValueError("unknown or unsupported KGSL free" if owned else "unknown KGSL user mapping")
        if not owned and ((mapping := self.memory.mappings.get(key)) is None or mapping.owner is not None):
          raise ValueError("unknown KGSL user mapping")
        if any(context.pending for context in self.contexts.values()):
          raise ValueError(f"cannot {'free' if owned else 'unmap'} with pending commands")
        if allocation.base:
          self.memory.unmap(allocation.base)
        del resources[key] # Explicit free retains CPU storage; QCOMDevice._gpu_free owns its CPU munmap.
      elif name == "GPU_COMMAND":
        # Capture command words at submission, assign their ordered timestamp,
        # and let drain advance PM4/IR3 work through transaction segments.
        context = self.context(fd, req.context_id)
        if (req.numcmds != 1 or req.cmdsize != ctypes.sizeof(kgsl.struct_kgsl_command_object) or req.flags or
            req.objlist or req.objsize or req.numobjs or req.synclist or req.syncsize or req.numsyncs):
          raise ValueError("unsupported KGSL command list")
        obj = read_struct(req.cmdlist, kgsl.struct_kgsl_command_object)
        if obj.flags != kgsl.KGSL_CMDLIST_IB or obj.offset or obj.id or not 0 < obj.size <= 64 << 20 or obj.size % 4 or obj.gpuaddr % 4:
          raise ValueError("unsupported KGSL command object")
        if context.queued == 0xffffffff:
          raise ValueError("KGSL timestamp exhausted")
        data = self.memory.transaction().read(obj.gpuaddr, obj.size)
        context.queued += 1
        req.timestamp = context.queued
        context.pending.append((struct.unpack(f"<{len(data)//4}I", data), req.timestamp, 0))
        self.drain()
      elif name == "CMDSTREAM_READTIMESTAMP_CTXTID":
        # Polling also gives queues another opportunity to observe live signals.
        context = self.context(fd, req.context_id)
        if req.type not in (kgsl.KGSL_TIMESTAMP_QUEUED, kgsl.KGSL_TIMESTAMP_RETIRED, kgsl.KGSL_TIMESTAMP_CONSUMED):
          raise ValueError("unsupported KGSL timestamp type")
        self.drain()
        req.timestamp = {
          kgsl.KGSL_TIMESTAMP_QUEUED:context.queued,
          kgsl.KGSL_TIMESTAMP_CONSUMED:context.consumed,
          kgsl.KGSL_TIMESTAMP_RETIRED:context.retired,
        }[req.type]
      elif name == "DEVICE_WAITTIMESTAMP_CTXTID":
        context = self.context(fd, req.context_id)
        self.drain()
        if context.retired < req.timestamp:
          raise TimeoutError("QCOM mock wait has no runnable producer")
      else:
        raise NotImplementedError(f"unsupported KGSL ioctl {name}")
      write_host(argp, bytes(req))
      return 0


def make_ioctl_callback(tracked_fds:dict, real_ioctl):
  # Data-plane entry from native HCQ2. Untracked descriptors use the real ioctl;
  # tracked QCOM descriptors enter the driver. Retain the first Python failure so
  # it reaches the caller at the Python boundary instead of disappearing in C.
  # ioctl is variadic: Darwin arm64 needs its fixed arguments declared so the
  # native fallback places the third argument according to the variadic ABI.
  real_ioctl.argtypes = (ctypes.c_int, ctypes.c_ulong)
  real_ioctl.restype = ctypes.c_int
  @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p)
  def callback(fd, request, argp):
    target = tracked_fds.get(fd)
    if not isinstance(target, QCOMFileDesc):
      return real_ioctl(ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_void_p(argp))
    try:
      return target.ioctl(fd, request, argp)
    except Exception as error:
      if target.driver.error is None:
        target.driver.error = error
      return -1
  return callback
