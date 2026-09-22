import ctypes, errno, functools, mmap, os
from collections.abc import Callable
from typing import cast
from tinygrad.runtime.autogen import kgsl, libc
from test.mockgpu.driver import VirtDriver, VirtFileDesc, VirtFile
from test.mockgpu.qcom.compute import QCOMCompute
from test.mockgpu.qcom.errors import ErrorCode, ModelError, ModelInputError
from test.mockgpu.qcom.state import read_buffer

def _ioctl_nr(ioctl: functools.partial) -> int: return ioctl.args[2]
def _ioctl_request(ioctl) -> int:
  idir, base, nr, struct = ioctl.args
  return (idir << 30) | (ctypes.sizeof(struct) << 16) | (base << 8) | nr

# match the full 32-bit ioctl encoding, not just nr (pins the struct ABI)
_SUPPORTED_IOCTL_REQUESTS = { _ioctl_nr(x): _ioctl_request(x) for x in (kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY,
                             kgsl.IOCTL_KGSL_SETPROPERTY, kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY,
                             kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.IOCTL_KGSL_GPUOBJ_FREE, kgsl.IOCTL_KGSL_GPU_COMMAND,
                             kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID, kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID) }

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd, driver):
    super().__init__(fd)
    self.driver = driver
  def ioctl(self, fd, request, argp): return self.driver.ioctl(fd, request, argp)
  def mmap(self, st, sz, prot, flags, fd, off): return self.driver.mmap(fd, sz, prot, flags, off)
  def close(self, fd): return self.driver.close(fd)

class QCOMDriver(VirtDriver):
  # A630's major/minor/patch fields occupy chip_id bits 31:24/23:16/15:8.
  chip_id = 0x06030000

  def __init__(self):
    super().__init__()
    self.next_fd, self.next_context, self.next_object = 1 << 28, 1, 1
    self.fds: set[int] = set()
    self.contexts: dict[int, int] = {}
    self.objects: dict[int, dict[str, int]] = {}
    self.maps: dict[int, int] = {}
    self.gpus: dict[int, QCOMCompute] = {}
    self.timestamps: dict[int, int] = {}
    self.constraints: dict[int, int] = {}
    self.last_submission_stats: dict[str, int] = {}
    self._instrumentation: Callable[[dict[str, object]], None] | None = None
    self.tracked_files = [VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self))]

  # Observation only; dispatch still runs through QCOMCompute.
  @property
  def instrumentation(self) -> Callable[[dict[str, object]], None] | None:
    return self._instrumentation

  @instrumentation.setter
  def instrumentation(self, callback: Callable[[dict[str, object]], None] | None):
    self._instrumentation = callback
    for gpu in self.gpus.values(): gpu.instrumentation = callback

  def open(self, name, flags, mode, virtfile):
    fd, self.next_fd = self.next_fd, self.next_fd + 1
    self.fds.add(fd)
    return virtfile.fdcls(fd)

  def _check_fd(self, fd):
    if fd not in self.fds: raise OSError(errno.EBADF, f"invalid KGSL fd {fd}")

  def mmap(self, fd, size, prot, flags, offset):
    self._check_fd(fd)
    if offset < 0 or offset % 0x1000: raise OSError(errno.EINVAL, f"invalid KGSL mmap offset={offset:#x}")
    if flags != mmap.MAP_SHARED: raise OSError(errno.EINVAL, f"unsupported KGSL mmap flags {flags:#x}")
    if prot & ~(mmap.PROT_READ | mmap.PROT_WRITE): raise OSError(errno.EINVAL, f"unsupported KGSL mmap prot {prot:#x}")
    # per-fd ownership: free revokes future maps, not existing ones
    obj = self.objects.get(offset // 0x1000)
    if obj is None or obj['fd'] != fd: raise OSError(errno.EINVAL, f"invalid KGSL mmap offset={offset:#x}")
    if size != obj['size']: raise OSError(errno.ERANGE, f"KGSL mmap size {size:#x} != {obj['size']:#x}")
    addr = libc.mmap(0, size, prot, flags, obj['backing_fd'], 0)
    if addr == ctypes.c_void_p(-1).value: raise OSError(ctypes.get_errno(), f"KGSL mmap object {offset // 0x1000} size={size:#x}")
    self.maps[addr] = offset // 0x1000
    return addr

  def mapped_objects(self, fd) -> dict[int, dict[str, int]]:
    self._check_fd(fd)
    with open('/proc/self/maps') as maps: lines = [line.split() for line in maps]
    regions: dict[int, dict[str, int]] = {}
    for addr, ident in self.maps.items():
      obj = self.objects[ident]
      if obj['fd'] != fd or not obj['flags'] & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP: continue
      stat = os.fstat(obj['backing_fd'])
      for span, perms, offset, device, inode, *_ in lines:
        start, end = (int(x, 16) for x in span.split('-'))
        major, minor = (int(x, 16) for x in device.split(':'))
        if start <= addr and addr + obj['size'] <= end and int(offset, 16) + addr - start == 0 \
            and perms.endswith('s') and int(inode) == stat.st_ino and os.makedev(major, minor) == stat.st_dev:
          if any(other is obj for other in regions.values()):
            raise OSError(errno.EINVAL, f"KGSL object {ident} has multiple GPU mappings")
          regions[addr] = obj
          break
    return regions

  def buffers(self, fd, regions=None) -> dict[int, bytearray]:
    if regions is None: regions = self.mapped_objects(fd)
    buffers = {}
    for addr, obj in regions.items():
      data = os.pread(obj['backing_fd'], obj['size'], 0)
      if len(data) != obj['size']: raise OSError(errno.EFAULT, f"short KGSL backing read at {addr:#x}")
      buffers[addr] = bytearray(data)
    return buffers

  def publish(self, regions, before, after):
    attempted = []
    try:
      for addr, blob in after.items():
        if blob == before[addr]: continue
        attempted.append(addr)
        if os.pwrite(regions[addr]['backing_fd'], blob, 0) != len(blob): raise OSError(errno.EIO, f'short KGSL write at {addr:#x}')
    except OSError as error:
      setattr(error, 'model_code', ErrorCode.ROLLBACK)
      failed = []
      restored = 0
      for addr in attempted:
        try:
          if os.pwrite(regions[addr]['backing_fd'], before[addr], 0) != len(before[addr]): failed.append(addr)
          else: restored += len(before[addr])
        except OSError: failed.append(addr)
      self.last_submission_stats['restored_bytes'] = restored
      if failed:
        rollback = OSError(errno.EIO, f'KGSL rollback failed at {", ".join(hex(addr) for addr in failed)}')
        setattr(rollback, 'model_code', ErrorCode.ROLLBACK)
        raise rollback from error
      raise

  def close(self, fd):
    self._check_fd(fd)
    self.fds.remove(fd)
    self.contexts = {ctx: owner for ctx, owner in self.contexts.items() if owner != fd}
    self.gpus = {ctx: gpu for ctx, gpu in self.gpus.items() if ctx in self.contexts}
    self.timestamps = {ctx: ts for ctx, ts in self.timestamps.items() if ctx in self.contexts}
    self.constraints = {ctx: level for ctx, level in self.constraints.items() if ctx in self.contexts}
    for obj, alloc in list(self.objects.items()):
      if alloc['fd'] == fd: self._release_object(obj)
    return 0

  def _release_object(self, obj):
    self.maps = {addr: ident for addr, ident in self.maps.items() if ident != obj}
    os.close(self.objects.pop(obj)['backing_fd'])

  def ioctl(self, fd, request, argp):
    self._check_fd(fd)
    nr = request & 0xFF
    if request != _SUPPORTED_IOCTL_REQUESTS.get(nr): raise OSError(errno.ENOTTY, f"unsupported KGSL ioctl {request:#x}")
    if not argp: raise OSError(errno.EFAULT, f"KGSL ioctl {request:#x} argument pointer is NULL")
    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND):
      cmd = kgsl.struct_kgsl_gpu_command.from_address(argp)
      if self.contexts.get(cmd.context_id) != fd:
        raise OSError(errno.EINVAL, f"KGSL submit context {cmd.context_id} not owned by fd {fd}")
      if cmd.flags: raise OSError(errno.EINVAL, f"unsupported KGSL submit flags {cmd.flags:#x}")
      if cmd.numcmds != 1 or cmd.cmdsize != ctypes.sizeof(kgsl.struct_kgsl_command_object):
        raise OSError(errno.EINVAL, f"unsupported KGSL command list count={cmd.numcmds} size={cmd.cmdsize}")
      if cmd.objlist or cmd.objsize or cmd.numobjs or cmd.synclist or cmd.syncsize or cmd.numsyncs:
        raise OSError(errno.EINVAL, "unsupported KGSL submit object or sync list")
      if not cmd.cmdlist: raise OSError(errno.EFAULT, "KGSL command list pointer is NULL")
      # Copy through the kernel; reject inaccessible pointers without Python dereference.
      try:
        with open('/proc/self/mem', 'rb', buffering=0) as mem: data = os.pread(mem.fileno(), cmd.cmdsize, cmd.cmdlist)
      except (OSError, OverflowError) as e:
        raise OSError(errno.EFAULT, f"unreadable KGSL command list {cmd.cmdlist:#x}") from e
      if len(data) != cmd.cmdsize: raise OSError(errno.EFAULT, f"truncated KGSL command list {cmd.cmdlist:#x}")
      ib = kgsl.struct_kgsl_command_object.from_buffer_copy(data)
      if ib.flags != kgsl.KGSL_CMDLIST_IB or ib.offset or ib.id:
        raise OSError(errno.EINVAL, f"unsupported KGSL command flags={ib.flags:#x} offset={ib.offset:#x} id={ib.id}")
      if not ib.size or ib.size % 4 or ib.gpuaddr % 4 or ib.gpuaddr + ib.size > 1 << 64:
        raise OSError(errno.EINVAL, f"invalid KGSL command range {ib.gpuaddr:#x}+{ib.size:#x}")
      if self.timestamps[cmd.context_id] == 0xffffffff: raise OSError(errno.EOVERFLOW, 'KGSL timestamp wrap unsupported')
      regions = self.mapped_objects(fd)
      before = self.buffers(fd, regions)
      try: stream = read_buffer(before, ib.gpuaddr, ib.size, 'KGSL command')
      except ModelInputError as e:
        error = OSError(errno.EFAULT, str(e))
        setattr(error, 'model_code', e.code)
        raise error from e
      except ValueError as e:
        error = OSError(errno.EFAULT, str(e))
        setattr(error, 'model_code', ErrorCode.INPUT)
        raise error from e
      transaction = self.gpus[cmd.context_id].begin_transaction(before)
      try: transaction.gpu.execute_command_stream(stream, transaction.buffers)
      except ModelInputError as e:
        error = OSError(errno.EINVAL, f'KGSL context {cmd.context_id}: {e}')
        setattr(error, 'model_code', e.code)
        raise error from e
      except ModelError as e:
        error = OSError(errno.EIO, f'KGSL context {cmd.context_id}: {e}')
        setattr(error, 'model_code', e.code)
        raise error from e
      except ValueError as e:
        error = OSError(errno.EIO, f'KGSL context {cmd.context_id}: emulator failure: {e}')
        setattr(error, 'model_code', ErrorCode.EXECUTION)
        raise error from e
      except Exception as e:
        error = OSError(errno.EIO, f'KGSL context {cmd.context_id}: emulator failure: {e}')
        setattr(error, 'model_code', ErrorCode.EXECUTION)
        raise error from e
      if self.mapped_objects(fd) != regions: raise OSError(errno.EFAULT, 'KGSL mappings changed during submission')
      self.last_submission_stats = {
        'objects': len(transaction.before),
        'snapshot_bytes': sum(len(blob) for blob in transaction.before.values()),
        'changed_bytes': transaction.changed_bytes(),
        'restored_bytes': 0,
      }
      self.publish(regions, transaction.before, transaction.buffers)
      transaction.commit_state()
      self.timestamps[cmd.context_id] += 1
      cmd.timestamp = self.timestamps[cmd.context_id]
    elif nr in (_ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID), _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID)):
      if nr == _ioctl_nr(kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID):
        ts = kgsl.struct_kgsl_cmdstream_readtimestamp_ctxtid.from_address(argp)
        if self.contexts.get(ts.context_id) != fd: raise OSError(errno.EINVAL, f'KGSL timestamp context {ts.context_id} not owned by fd {fd}')
        if ts.type not in (kgsl.KGSL_TIMESTAMP_QUEUED, kgsl.KGSL_TIMESTAMP_CONSUMED, kgsl.KGSL_TIMESTAMP_RETIRED):
          raise OSError(errno.EINVAL, f'unsupported KGSL timestamp type {ts.type}')
        ts.timestamp = self.timestamps[ts.context_id]
      else:
        wait = kgsl.struct_kgsl_device_waittimestamp_ctxtid.from_address(argp)
        if self.contexts.get(wait.context_id) != fd:
          raise OSError(errno.EINVAL, f'KGSL timestamp context {wait.context_id} not owned by fd {fd}')
        if wait.timestamp > self.timestamps[wait.context_id]:
          raise OSError(errno.ETIMEDOUT, f'KGSL timestamp {wait.timestamp} not completed')
    if nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE):
      ctx = kgsl.struct_kgsl_drawctxt_create.from_address(argp)
      flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE \
        | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC | kgsl.KGSL_CONTEXT_PRIORITY_MASK | kgsl.KGSL_CONTEXT_PREEMPT_STYLE_MASK
      if ctx.flags & ~flags: raise OSError(errno.EINVAL, f"unsupported KGSL context flags {ctx.flags:#x}")
      style = (ctx.flags & kgsl.KGSL_CONTEXT_PREEMPT_STYLE_MASK) >> kgsl.KGSL_CONTEXT_PREEMPT_STYLE_SHIFT
      if style not in (0, kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN):
        raise OSError(errno.EINVAL, f"unsupported KGSL context preempt style {style}")
      ctx.drawctxt_id, self.next_context = self.next_context, self.next_context + 1
      self.contexts[ctx.drawctxt_id] = fd
      self.gpus[ctx.drawctxt_id], self.timestamps[ctx.drawctxt_id] = QCOMCompute(self.instrumentation), 0
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY):
      destroy = kgsl.struct_kgsl_drawctxt_destroy.from_address(argp)
      owner = self.contexts.get(destroy.drawctxt_id)
      if owner is None: raise OSError(errno.EINVAL, f"unknown KGSL context {destroy.drawctxt_id}")
      if owner != fd: raise OSError(errno.EINVAL, f"KGSL context {destroy.drawctxt_id} not owned by fd {fd}")
      del self.contexts[destroy.drawctxt_id]
      del self.gpus[destroy.drawctxt_id]
      del self.timestamps[destroy.drawctxt_id]
      self.constraints.pop(destroy.drawctxt_id, None)
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_SETPROPERTY):
      prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if prop.type != kgsl.KGSL_PROP_PWR_CONSTRAINT: raise OSError(errno.EINVAL, f"unsupported KGSL property {prop.type:#x}")
      if not prop.value: raise OSError(errno.EFAULT, "KGSL PWR constraint payload pointer is NULL")
      if prop.sizebytes != ctypes.sizeof(kgsl.struct_kgsl_device_constraint):
        raise OSError(errno.EINVAL, f"KGSL PWR constraint size {prop.sizebytes} != {ctypes.sizeof(kgsl.struct_kgsl_device_constraint)}")
      constraint = kgsl.struct_kgsl_device_constraint.from_address(cast(int, prop.value))
      if constraint.context_id not in self.contexts:
        raise OSError(errno.EINVAL, f"KGSL PWR constraint context {constraint.context_id} not found")
      if self.contexts[constraint.context_id] != fd:
        raise OSError(errno.EINVAL, f"KGSL PWR constraint context {constraint.context_id} not owned by fd {fd}")
      if constraint.type != kgsl.KGSL_CONSTRAINT_PWRLEVEL:
        raise OSError(errno.EINVAL, f"unsupported KGSL PWR constraint type {constraint.type}")
      if constraint.size != ctypes.sizeof(kgsl.struct_kgsl_device_constraint_pwrlevel):
        raise OSError(errno.EINVAL, f"unsupported KGSL PWR constraint size {constraint.size}")
      if not constraint.data: raise OSError(errno.EFAULT, "KGSL PWR constraint data pointer is NULL")
      level = kgsl.struct_kgsl_device_constraint_pwrlevel.from_address(cast(int, constraint.data)).level
      if level not in (kgsl.KGSL_CONSTRAINT_PWR_MIN, kgsl.KGSL_CONSTRAINT_PWR_MAX):
        raise OSError(errno.EINVAL, f"unsupported KGSL PWR constraint level {level}")
      self.constraints[constraint.context_id] = level
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY):
      prop = kgsl.struct_kgsl_device_getproperty.from_address(argp)
      if prop.type != kgsl.KGSL_PROP_DEVICE_INFO or prop.sizebytes != ctypes.sizeof(kgsl.struct_kgsl_devinfo):
        raise OSError(errno.EINVAL, f"unsupported KGSL property {prop.type:#x} size={prop.sizebytes}")
      if not prop.value: raise OSError(errno.EFAULT, "KGSL property payload pointer is NULL")
      info = kgsl.struct_kgsl_devinfo.from_address(cast(int, prop.value))
      info.device_id, info.chip_id, info.mmu_enabled = kgsl.KGSL_DEVICE_3D0, self.chip_id, 1
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC):
      obj = kgsl.struct_kgsl_gpuobj_alloc.from_address(argp)
      if obj.size == 0 or obj.size > 0xFFFFFFFF: raise OSError(errno.EINVAL, f"KGSL allocation size {obj.size:#x} out of range")
      flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (12 << kgsl.KGSL_MEMALIGN_SHIFT)
      if obj.flags not in (0, flags) or obj.va_len or obj.metadata_len or obj.metadata:
        raise OSError(errno.EINVAL, f"unsupported KGSL allocation flags={obj.flags:#x} va_len={obj.va_len} metadata_len={obj.metadata_len}")
      backing_fd = os.memfd_create(f"mockkgsl-{self.next_object}", os.MFD_CLOEXEC)
      try: os.ftruncate(backing_fd, obj.size)
      except OSError:
        os.close(backing_fd)
        raise
      obj.id, self.next_object = self.next_object, self.next_object + 1
      obj.mmapsize = obj.size
      self.objects[obj.id] = {'fd': fd, 'size': obj.size, 'backing_fd': backing_fd, 'flags': obj.flags}
    elif nr == _ioctl_nr(kgsl.IOCTL_KGSL_GPUOBJ_FREE):
      free = kgsl.struct_kgsl_gpuobj_free.from_address(argp)
      if free.flags or free.priv or free.type or free.len:
        raise OSError(errno.EINVAL, f"unsupported KGSL free object {free.id} flags={free.flags:#x} type={free.type} len={free.len}")
      alloc = self.objects.get(free.id)
      if alloc is None: raise OSError(errno.EINVAL, f"unknown KGSL object {free.id}")
      if alloc['fd'] != fd: raise OSError(errno.EINVAL, f"KGSL object {free.id} not owned by fd {fd}")
      self._release_object(free.id)
    return 0
