import ctypes, errno, mmap, os, subprocess, sys, textwrap, unittest
from typing import Any, cast
from unittest.mock import patch
from tinygrad.runtime.autogen import kgsl, libc
from tinygrad.runtime.support.system import FileIOInterface
from test.mockgpu.qcom.qcomdriver import QCOMDriver, _ioctl_request

class TestQCOMDriver(unittest.TestCase):
  def setUp(self):
    self.driver = QCOMDriver()
    self.fd = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0])

  def tearDown(self):
    for fd in list(self.driver.fds): self.driver.close(fd)

  def call(self, op, arg, fd=None):
    self.assertEqual(self.driver.ioctl(self.fd.fd if fd is None else fd, _ioctl_request(op), ctypes.addressof(arg)), 0)
    return arg

  def map(self, obj, fd=None):
    addr = self.driver.mmap(self.fd.fd if fd is None else fd, obj.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, obj.id * 0x1000)
    self.addCleanup(libc.munmap, addr, obj.size)
    return addr

  def check(self, no, msg, fn):
    with self.assertRaisesRegex(OSError, msg) as e: fn()
    self.assertEqual(e.exception.errno, no)

  def test_context_and_a630_device_info(self):
    ctx = kgsl.struct_kgsl_drawctxt_create()
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx)
    self.assertEqual(ctx.drawctxt_id, 1)
    info = kgsl.struct_kgsl_devinfo()
    prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.call(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY, prop)
    self.assertEqual(info.chip_id, QCOMDriver.chip_id)

  def test_drawctxt_destroy_removes_context(self):
    ctx = kgsl.struct_kgsl_drawctxt_create()
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx)
    destroy = kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=ctx.drawctxt_id)
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, destroy)
    self.assertEqual(self.driver.contexts, {})
    self.check(errno.EINVAL, "unknown KGSL context 1", lambda: self.call(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, destroy))

  def test_cross_fd_ownership_rejected(self):
    ctx = kgsl.struct_kgsl_drawctxt_create()
    alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, alloc)
    other = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0])
    destroy = kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=ctx.drawctxt_id)
    self.check(errno.EINVAL, "not owned by fd", lambda: self.call(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, destroy, other.fd))
    free = kgsl.struct_kgsl_gpuobj_free(id=alloc.id)
    self.check(errno.EINVAL, "not owned by fd", lambda: self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free, other.fd))
    self.check(errno.EINVAL, "invalid KGSL mmap",
               lambda: self.driver.mmap(other.fd, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000))

  def test_allocate_map_and_free(self):
    alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, alloc)
    addr = self.driver.mmap(self.fd.fd, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)
    self.addCleanup(libc.munmap, addr, alloc.size)
    ctypes.c_uint32.from_address(addr).value = 0xA630
    self.assertEqual(ctypes.c_uint32.from_address(addr).value, 0xA630)
    free = kgsl.struct_kgsl_gpuobj_free(id=alloc.id)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free)
    self.assertEqual(self.driver.objects, {})
    self.check(errno.EINVAL, f"unknown KGSL object {alloc.id}", lambda: self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free))
    self.assertEqual(ctypes.c_uint32.from_address(addr).value, 0xA630)

  def test_repeated_map_shares_object_and_free_revokes_future_maps(self):
    alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, alloc)
    flags = mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED
    first = self.driver.mmap(self.fd.fd, 0x1000, *flags, alloc.id * 0x1000)
    second = self.driver.mmap(self.fd.fd, 0x1000, *flags, alloc.id * 0x1000)
    self.addCleanup(libc.munmap, first, alloc.size)
    self.addCleanup(libc.munmap, second, alloc.size)
    ctypes.c_uint32.from_address(first).value = 0xA630
    self.assertEqual(ctypes.c_uint32.from_address(second).value, 0xA630)
    free = kgsl.struct_kgsl_gpuobj_free(id=alloc.id)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free)
    self.check(errno.EINVAL, "invalid KGSL mmap", lambda: self.driver.mmap(self.fd.fd, 0x1000, *flags, alloc.id * 0x1000))

  def test_mmap_size_mismatch_erange(self):
    alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, alloc)
    self.check(errno.ERANGE, "0x800 != 0x1000",
               lambda: self.driver.mmap(self.fd.fd, 0x800, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000))

  def test_failed_setup_leaves_no_resource(self):
    for size in (0, 0x100000000):
      alloc = kgsl.struct_kgsl_gpuobj_alloc(size=size)
      self.check(errno.EINVAL, f"size {size:#x} out of range", lambda a=alloc: self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, a))
    self.assertEqual(self.driver.objects, {})

  def test_unknown_object_fails(self):
    free = kgsl.struct_kgsl_gpuobj_free(id=1)
    self.check(errno.EINVAL, "unknown KGSL object 1", lambda: self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free))

  def test_pwr_constraint_payload_contract(self):
    ctx = kgsl.struct_kgsl_drawctxt_create()
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx)
    level = kgsl.struct_kgsl_device_constraint_pwrlevel(level=1)
    constraint = kgsl.struct_kgsl_device_constraint(type=1, context_id=ctx.drawctxt_id, data=ctypes.addressof(level), size=4)
    prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_PWR_CONSTRAINT,
                                            value=ctypes.addressof(constraint), sizebytes=ctypes.sizeof(constraint))
    self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop)
    prop.sizebytes = ctypes.sizeof(constraint) - 8
    self.check(errno.EINVAL, "size 16 != 24", lambda: self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop))
    prop.sizebytes = ctypes.sizeof(constraint)
    constraint.context_id = 5
    self.check(errno.EINVAL, "context 5 not found", lambda: self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop))
    prop.value = ctypes.c_void_p()
    self.check(errno.EFAULT, "NULL", lambda: self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop))

  def test_getproperty_requires_full_devinfo(self):
    info = kgsl.struct_kgsl_devinfo()
    prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info) - 1)
    self.check(errno.EINVAL, "unsupported KGSL property", lambda: self.call(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY, prop))

  def test_gpu_command_rejected(self):
    cmd = kgsl.struct_kgsl_gpu_command()
    self.check(errno.EINVAL, "submit context 0 not owned", lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))

  def submission(self):
    ctx = self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, kgsl.struct_kgsl_drawctxt_create())
    obj = kgsl.struct_kgsl_command_object(gpuaddr=0x100001000, size=4, flags=kgsl.KGSL_CMDLIST_IB)
    cmd = kgsl.struct_kgsl_gpu_command(context_id=ctx.drawctxt_id, cmdlist=ctypes.addressof(obj), numcmds=1,
                                       cmdsize=ctypes.sizeof(obj), timestamp=0xdeadbeef)
    return cmd, obj

  def test_unmapped_submit_never_reports_completion(self):
    cmd, obj = self.submission()
    before = bytes(cmd), bytes(obj), self.driver.contexts.copy()
    open_fds = set(os.listdir('/proc/self/fd'))
    for _ in range(2):
      self.check(errno.EFAULT, 'KGSL command.*backing buffers', lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
      self.assertEqual((bytes(cmd), bytes(obj), self.driver.contexts), before)
      self.assertEqual(self.driver.objects, {})
      self.assertEqual(set(os.listdir('/proc/self/fd')), open_fds)

  def test_submit_list_contract(self):
    cmd, obj = self.submission()
    for field, value, message in (('flags', 1, 'submit flags'), ('flags', 1 << 63, 'submit flags'),
                                   ('numcmds', 0, 'count=0'), ('numcmds', 2, 'count=2'), ('numcmds', 0xffffffff, 'count=4294967295'),
                                   ('cmdsize', 0, 'size=0'), ('cmdsize', 31, 'size=31'), ('cmdsize', 33, 'size=33')):
      with self.subTest(field=field, value=value):
        old = getattr(cmd, field)
        setattr(cmd, field, value)
        self.check(errno.EINVAL, message, lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
        self.assertEqual(cmd.timestamp, 0xdeadbeef)
        setattr(cmd, field, old)
    for field in ('objlist', 'objsize', 'numobjs', 'synclist', 'syncsize', 'numsyncs'):
      with self.subTest(field=field):
        setattr(cmd, field, 1)
        self.check(errno.EINVAL, 'object or sync list', lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
        setattr(cmd, field, 0)
    self.assertEqual(cmd.cmdlist, ctypes.addressof(obj))

  def test_submit_command_contract(self):
    cmd, obj = self.submission()
    for field, value in (('flags', 0), ('flags', 5), ('offset', 4), ('id', 1), ('size', 0), ('size', 3),
                          ('size', 1 << 63), ('gpuaddr', 0x1001)):
      with self.subTest(field=field, value=value):
        old = getattr(obj, field)
        setattr(obj, field, value)
        if field == 'size' and value == 1 << 63: obj.gpuaddr = (1 << 64)-4
        self.check(errno.EINVAL, '(unsupported KGSL command|invalid KGSL command range)',
                   lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
        self.assertEqual(cmd.timestamp, 0xdeadbeef)
        setattr(obj, field, old)
        obj.gpuaddr = 0x100001000

  def test_submit_bad_command_pointer(self):
    cmd, obj = self.submission()
    open_fds = set(os.listdir('/proc/self/fd'))
    for pointer, message in ((0, 'pointer is NULL'), (1, 'unreadable'), ((1 << 64)-1, 'unreadable')):
      with self.subTest(pointer=pointer):
        cmd.cmdlist = pointer
        self.check(errno.EFAULT, message, lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
    cmd.cmdlist = ctypes.addressof(obj)
    with patch('os.pread', return_value=bytes(31)):
      self.check(errno.EFAULT, 'truncated', lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
    request = _ioctl_request(kgsl.IOCTL_KGSL_GPU_COMMAND)
    self.check(errno.EFAULT, 'argument pointer is NULL', lambda: self.driver.ioctl(self.fd.fd, request, 0))
    self.check(errno.ENOTTY, 'unsupported KGSL ioctl', lambda: self.driver.ioctl(self.fd.fd, request ^ (1 << 16), ctypes.addressof(cmd)))
    self.assertEqual(set(os.listdir('/proc/self/fd')), open_fds)

  def test_submit_owner_and_destroyed_context(self):
    cmd, obj = self.submission()
    other = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0])
    self.check(errno.EINVAL, f'context {cmd.context_id} not owned', lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd, other.fd))
    destroy = kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=cmd.context_id)
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, destroy)
    self.check(errno.EINVAL, f'context {cmd.context_id} not owned', lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
    self.driver.close(self.fd.fd)
    self.check(errno.EBADF, 'invalid KGSL fd', lambda: self.call(kgsl.IOCTL_KGSL_GPU_COMMAND, cmd))
    self.assertEqual(cmd.cmdlist, ctypes.addressof(obj))
    self.assertEqual(cmd.timestamp, 0xdeadbeef)

  def test_close_releases_fd_owned_resources(self):
    ctx = kgsl.struct_kgsl_drawctxt_create()
    alloc = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx)
    self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, alloc)
    self.fd.close(self.fd.fd)
    self.assertEqual(self.driver.contexts, {})
    self.assertEqual(self.driver.objects, {})

  def test_unknown_property_and_malformed_request_fail(self):
    prop = kgsl.struct_kgsl_device_getproperty(type=0xFFFF, value=1, sizebytes=1)
    self.check(errno.EINVAL, "unsupported KGSL property", lambda: self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop))
    alloc = kgsl.struct_kgsl_gpuobj_alloc()
    self.check(errno.ENOTTY, "unsupported KGSL ioctl",
               lambda: self.driver.ioctl(self.fd.fd, _ioctl_request(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC) ^ (1 << 16), ctypes.addressof(alloc)))

  def test_closed_and_unknown_fd_rejected(self):
    self.driver.close(self.fd.fd)
    for fd in (self.fd.fd, 123):
      ctx = kgsl.struct_kgsl_drawctxt_create()
      self.check(errno.EBADF, f"KGSL fd {fd}", lambda: self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx, fd))
      self.check(errno.EBADF, f"KGSL fd {fd}", lambda: self.driver.mmap(fd, 0x1000, 3, mmap.MAP_SHARED, 0x1000))
      self.check(errno.EBADF, f"KGSL fd {fd}", lambda: self.driver.close(fd))
    self.assertEqual(self.driver.contexts, {})

  def test_allocation_failure_closes_backing_fd(self):
    before = set(os.listdir('/proc/self/fd'))
    for fn in ('memfd_create', 'ftruncate'):
      obj = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000)
      with patch(f'test.mockgpu.qcom.qcomdriver.os.{fn}', side_effect=OSError(errno.ENOSPC, fn)):
        self.check(errno.ENOSPC, fn, lambda: self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, obj))
      self.assertEqual(self.driver.objects, {})
      self.assertEqual((obj.id, obj.mmapsize), (0, 0))
      self.assertEqual(set(os.listdir('/proc/self/fd')), before)
    obj = self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=0x1000))
    self.assertEqual(obj.id, 1)

  def test_mapping_contract_and_failure(self):
    obj = self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=0x1000))
    for off in (-0x1000, obj.id * 0x1000 + 1):
      self.check(errno.EINVAL, f"offset={off:#x}", lambda: self.driver.mmap(self.fd.fd, obj.size, 3, mmap.MAP_SHARED, off))
    self.check(errno.EINVAL, "mmap flags", lambda: self.driver.mmap(self.fd.fd, obj.size, 3, mmap.MAP_PRIVATE, obj.id * 0x1000))
    with patch('test.mockgpu.qcom.qcomdriver.libc.mmap', return_value=ctypes.c_void_p(-1).value):
      with patch('test.mockgpu.qcom.qcomdriver.ctypes.get_errno', return_value=errno.ENOMEM):
        self.check(errno.ENOMEM, f"object {obj.id}", lambda: self.map(obj))
    self.check(errno.EINVAL, 'mmap prot', lambda: self.driver.mmap(self.fd.fd, obj.size, mmap.PROT_EXEC, mmap.MAP_SHARED, obj.id * 0x1000))
    self.assertIn(obj.id, self.driver.objects)
    ctypes.c_uint32.from_address(self.map(obj)).value = 0xA630

  def test_close_isolates_owners_and_keeps_existing_mapping(self):
    other = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0])
    ctx = self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, kgsl.struct_kgsl_drawctxt_create(), other.fd)
    obj = self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=0x1000))
    own = self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=0x1000), other.fd)
    backing = self.driver.objects[obj.id]['backing_fd']
    addr = self.map(obj)
    self.driver.close(self.fd.fd)
    self.assertEqual(self.driver.contexts, {ctx.drawctxt_id: other.fd})
    self.assertEqual(set(self.driver.objects), {own.id})
    self.check(errno.EBADF, "Bad file descriptor", lambda: os.fstat(backing))
    ctypes.c_uint32.from_address(addr).value = 0xA630
    self.assertEqual(ctypes.c_uint32.from_address(addr).value, 0xA630)
    self.map(own, other.fd)

  def test_context_destroy_preserves_fd_allocation(self):
    ctx = self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, kgsl.struct_kgsl_drawctxt_create())
    obj = self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=0x1000))
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=ctx.drawctxt_id))
    self.assertEqual(self.driver.contexts, {})
    self.assertIn(obj.id, self.driver.objects)
    self.map(obj)

  def test_pwr_constraint_owner_and_fields(self):
    ctx = self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, kgsl.struct_kgsl_drawctxt_create())
    other = self.driver.open('/dev/kgsl-3d0', 0, 0, self.driver.tracked_files[0])
    level = kgsl.struct_kgsl_device_constraint_pwrlevel(level=1)
    constraint = kgsl.struct_kgsl_device_constraint(type=1, context_id=ctx.drawctxt_id, data=ctypes.addressof(level), size=4)
    prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_PWR_CONSTRAINT,
                                            value=ctypes.addressof(constraint), sizebytes=ctypes.sizeof(constraint))
    self.check(errno.EINVAL, f"context {ctx.drawctxt_id} not owned by fd {other.fd}",
               lambda: self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop, other.fd))
    self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop)
    self.assertEqual(self.driver.constraints, {ctx.drawctxt_id: 1})
    for field, value, no, msg in (('type', 99, errno.EINVAL, 'constraint type 99'),
                                 ('size', 8, errno.EINVAL, 'constraint size 8'), ('data', 0, errno.EFAULT, 'NULL')):
      old = getattr(constraint, field)
      setattr(constraint, field, value)
      self.check(no, msg, lambda: self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop))
      setattr(constraint, field, old)
    level.level = 2
    self.check(errno.EINVAL, "constraint level 2", lambda: self.call(kgsl.IOCTL_KGSL_SETPROPERTY, prop))
    self.assertEqual(self.driver.constraints, {ctx.drawctxt_id: 1})
    self.call(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY, kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=ctx.drawctxt_id))
    self.assertEqual(self.driver.constraints, {})

  def test_unsupported_resource_modes(self):
    for fields in ({'flags': 1 << 63}, {'va_len': 0x1000}, {'metadata_len': 1}, {'metadata': 1}):
      obj = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000, **fields)
      self.check(errno.EINVAL, 'unsupported KGSL allocation', lambda: self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, obj))
      self.assertEqual(self.driver.objects, {})
    obj = self.call(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, kgsl.struct_kgsl_gpuobj_alloc(size=0x1000))
    for fields in ({'flags': 1}, {'priv': 1}, {'type': 1}, {'len': 1}):
      free = kgsl.struct_kgsl_gpuobj_free(id=obj.id, **fields)
      self.check(errno.EINVAL, f'unsupported KGSL free object {obj.id}', lambda: self.call(kgsl.IOCTL_KGSL_GPUOBJ_FREE, free))
      self.assertIn(obj.id, self.driver.objects)
    ctx = kgsl.struct_kgsl_drawctxt_create(flags=1 << kgsl.KGSL_CONTEXT_PREEMPT_STYLE_SHIFT)
    self.check(errno.EINVAL, 'preempt style 1', lambda: self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx))
    ctx = kgsl.struct_kgsl_drawctxt_create(flags=1 << 31)
    self.check(errno.EINVAL, 'context flags', lambda: self.call(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, ctx))

  def test_null_ioctl_pointer(self):
    req = _ioctl_request(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE)
    self.check(errno.EFAULT, f'{req:#x}.*NULL', lambda: self.driver.ioctl(self.fd.fd, req, 0))

  def test_qcom_wait_failure_preserves_context(self):
    from unittest.mock import patch
    from tinygrad.runtime.ops_qcom import QCOMDevice
    device = object.__new__(QCOMDevice)
    device.fd, device.ctx, device.wait_timeout_ms = cast(FileIOInterface, cast(Any, object())), 7, 11
    timestamp = type('Timestamp', (), {'timestamp': 13})()
    with patch('tinygrad.runtime.ops_qcom.kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID', return_value=timestamp), \
         patch('tinygrad.runtime.ops_qcom.kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID', side_effect=OSError(errno.ETIMEDOUT, 'not completed')), \
         patch('tinygrad.runtime.ops_qcom.Compiled._wait_signal') as parent:
      with self.assertRaisesRegex(OSError, r'QCOM wait context=7 timestamp=13: \[Errno 110\] not completed'):
        device._wait_signal(memoryview(bytearray([0])), 1)
      parent.assert_not_called()

  def run_code(self, code, dev='MOCK+QCOM'):
    out = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=os.getcwd(),
                         env=os.environ | {'DEV': dev, 'PYTHONPATH': '.'}, capture_output=True, text=True, timeout=60)
    self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
    self.assertEqual(out.stderr, '')

  def test_mock_device_routes_and_closes(self):
    self.run_code("""
      import ctypes, gc
      from tinygrad import Device
      from tinygrad.runtime.autogen import kgsl
      d = Device['QCOM']
      import test.mockgpu.mockgpu as mockgpu
      driver = mockgpu.drivers[0]
      assert mockgpu.tracked_fds[d.fd.fd].driver is driver
      assert (d.gpu_id, d.arch) == ((6, 3, 0), 'a630')
      assert driver.contexts == {d.ctx: d.fd.fd}
      assert driver.constraints == {d.ctx: 1}
      for _ in range(3):
        buf = d._gpu_alloc(16)
        assert buf.meta[0].id in driver.objects
        ctypes.c_uint32.from_address(buf.buf).value = 0xA630
        assert ctypes.c_uint32.from_address(buf.buf).value == 0xA630
        d._gpu_free(buf)
        assert not driver.objects
      kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(d.fd, drawctxt_id=d.ctx)
      assert not driver.contexts and not driver.constraints
      d.finalize()
      del d.fd
      gc.collect()
      assert not driver.fds and not mockgpu.tracked_fds
    """)

  def test_mock_file_close_releases_driver_resources(self):
    self.run_code("""
      import gc, os
      from tinygrad.runtime.autogen import kgsl
      from tinygrad.runtime.support.system import FileIOInterface
      import test.mockgpu.mockgpu as mockgpu
      driver = mockgpu.drivers[0]
      before = set(os.listdir('/proc/self/fd'))
      for _ in range(3):
        f = FileIOInterface('/dev/kgsl-3d0')
        kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(f, flags=0)
        kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(f, size=0x1000)
        del f
        gc.collect()
        assert not driver.contexts and not driver.objects and not driver.fds
        assert not mockgpu.tracked_fds
        assert set(os.listdir('/proc/self/fd')) == before
    """)

  def test_mock_file_close_is_idempotent_after_runtime_close(self):
    self.run_code("""
      import gc
      from tinygrad.runtime.support.system import FileIOInterface
      import test.mockgpu.mockgpu as mockgpu
      first = FileIOInterface('/dev/kgsl-3d0')
      old_fd = first.fd
      mockgpu.runtime.close()
      second = FileIOInterface('/dev/kgsl-3d0')
      assert second.fd != old_fd
      first.close()
      assert mockgpu.tracked_fds[second.fd].driver is mockgpu.drivers[0]
      first.close()
      del first
      gc.collect()
      second.close()
      assert not mockgpu.tracked_fds and not mockgpu.drivers[0].fds
    """)

  def test_failed_device_setup_closes_file(self):
    self.run_code("""
      import gc
      from unittest.mock import patch
      from tinygrad.runtime.ops_qcom import QCOMDevice
      from tinygrad.runtime.autogen import kgsl
      from test.mockgpu.qcom.qcomdriver import QCOMDriver, _ioctl_request
      import test.mockgpu.mockgpu as mockgpu
      ioctl = QCOMDriver.ioctl
      def fail(self, fd, request, argp):
        if request == _ioctl_request(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY): raise OSError('injected device-info failure')
        return ioctl(self, fd, request, argp)
      with patch.object(QCOMDriver, 'ioctl', fail):
        try: QCOMDevice('QCOM')
        except OSError as e: assert 'injected device-info failure' in str(e)
        else: raise AssertionError('setup succeeded')
      gc.collect()
      driver = mockgpu.drivers[0]
      assert not driver.contexts and not driver.constraints and not driver.objects and not driver.fds
      assert not mockgpu.tracked_fds
    """)

  def test_failed_gpu_alloc_releases_object(self):
    self.run_code("""
      from unittest.mock import patch
      from tinygrad.runtime.ops_qcom import QCOMDevice
      import test.mockgpu.mockgpu as mockgpu
      d = QCOMDevice('QCOM')
      driver = mockgpu.tracked_fds[d.fd.fd].driver
      with patch.object(d.fd, 'mmap', side_effect=OSError('injected mmap failure')):
        try: d._gpu_alloc(64)
        except OSError as error: assert 'injected mmap failure' in str(error)
        else: raise AssertionError('allocation succeeded')
      assert not driver.objects and not driver.maps
      from tinygrad.runtime.autogen import kgsl
      kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(d.fd, drawctxt_id=d.ctx)
      d.fd.close()
      assert not driver.contexts and not driver.fds
    """)

  def test_runtime_file_owners_are_isolated(self):
    self.run_code("""
      import tinygrad.runtime.support.system
      from test.mockgpu.mockgpu import MockRuntime, MockFileIOInterface
      first, second = MockRuntime(), MockRuntime()
      one = MockFileIOInterface('/dev/kgsl-3d0', runtime=first)
      two = MockFileIOInterface('/dev/kgsl-3d0', runtime=second)
      assert one.mock_runtime is first and two.mock_runtime is second
      assert first.tracked_fds[one.fd].driver is first.drivers[0]
      assert second.tracked_fds[two.fd].driver is second.drivers[0]
      assert first.tracked_fds is not second.tracked_fds
      first.close()
      assert not first.tracked_fds and two.fd in second.tracked_fds
      two.close()
      assert not second.tracked_fds
    """)

  def test_amd_qcom_and_native_fd_isolation(self):
    self.run_code("""
      import ctypes, errno, gc, mmap, os, tempfile
      from tinygrad.runtime.support.system import FileIOInterface
      from tinygrad.runtime.autogen import kgsl, kfd
      import test.mockgpu.mockgpu as mockgpu
      q = FileIOInterface('/dev/kgsl-3d0')
      a = FileIOInterface('/dev/kfd')
      assert q.fd != a.fd
      qdriver = mockgpu.tracked_fds[q.fd].driver
      adriver = mockgpu.tracked_fds[a.fd].driver
      assert qdriver is not adriver
      version = kfd.AMDKFD_IOC_GET_VERSION(a)
      assert version.major_version == 1
      kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(q, flags=0)
      obj = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(q, size=0x1000)
      addr = q.mmap(0, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, obj.id * 0x1000)
      FileIOInterface.munmap(addr, 0x1000)
      with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b'kgsl')
        tmp.flush()
        native = FileIOInterface(tmp.name, os.O_RDONLY)
        assert native.fd not in mockgpu.tracked_fds
        assert native.read() == 'kgsl'
        try: native.ioctl(0, ctypes.c_uint32())
        except OSError as e: assert e.errno == errno.ENOTTY
        else: raise AssertionError('native ioctl succeeded')
        addr = native.mmap(0, 4, mmap.PROT_READ, mmap.MAP_SHARED, 0)
        assert ctypes.string_at(addr, 4) == b'kgsl'
        FileIOInterface.munmap(addr, 4)
        native_fd = native.fd
        del native
        gc.collect()
        try: os.fstat(native_fd)
        except OSError: pass
        else: raise AssertionError('native fd leaked')
      del q
      gc.collect()
      assert not qdriver.contexts and not qdriver.objects and not qdriver.fds
      assert mockgpu.tracked_fds[a.fd].driver is adriver
      assert kfd.AMDKFD_IOC_GET_VERSION(a).major_version == version.major_version
      del a
      gc.collect()
      assert not mockgpu.tracked_fds
    """, dev='MOCK+QCOM;MOCK+AMD')

if __name__ == '__main__': unittest.main()
