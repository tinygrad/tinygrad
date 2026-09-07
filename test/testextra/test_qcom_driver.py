import ctypes, importlib, mmap, os, pathlib, subprocess, sys, tempfile, unittest
from tinygrad.helpers import DEV
from tinygrad.runtime.autogen import kgsl, libc


def request_number(fn):
  direction, group, number, typ = fn.args
  return direction << 30 | ctypes.sizeof(typ) << 16 | group << 8 | number


class QCOMDriverTests(unittest.TestCase):
  def setUp(self):
    path = pathlib.Path(__file__).parents[1] / "mockgpu/qcom/qcomdriver.py"
    self.assertTrue(path.is_file(), "QCOM memory and driver implementation is missing")
    self.module = importlib.import_module("test.mockgpu.qcom.qcomdriver")
    self.storage = (ctypes.c_ubyte * 64)(*range(64))
    self.base = ctypes.addressof(self.storage)
    self.space = self.module.AddressSpace()
    self.space.map(self.base, 64, self.storage)

  def test_transaction_commit_and_discard(self):
    # Removing staging would expose partial command writes before commit.
    tx = self.space.transaction()
    tx.write(self.base + 3, b"abc")
    self.assertEqual(bytes(self.storage)[3:6], b"\x03\x04\x05")
    self.assertEqual(tx.read(self.base + 2, 5), b"\x02abc\x06")
    tx.commit()
    self.assertEqual(bytes(self.storage)[3:6], b"abc")
    later = self.space.transaction()
    later.write(self.base + 3, b"bad")
    with self.assertRaises(ValueError): later.write(self.base + 63, b"xx")
    self.assertEqual(bytes(self.storage)[3:6], b"abc")

  def test_ordered_overlays_and_numpy_region_share_changes(self):
    # Losing overlay order or keeping disconnected region snapshots changes data.
    tx = self.space.transaction()
    tx.write(self.base + 2, b"ABCD")
    tx.write(self.base + 3, b"xy")
    base, data = tx.region(self.base + 2, 4)
    self.assertEqual(base, self.base)
    self.assertEqual(data[2:6], b"AxyD")
    data[4] = ord("Z")
    self.assertEqual(tx.read(self.base + 2, 4), b"AxZD")
    tx.write(self.base + 3, b"!")
    self.assertEqual(data[2:6], b"A!ZD")
    tx.commit()
    self.assertEqual(bytes(self.storage)[2:6], b"A!ZD")

  def test_mappings_reject_overlap_and_out_of_range_without_reads(self):
    # A guest address outside registered storage must never reach a native read.
    with self.assertRaises(ValueError): self.space.map(self.base + 1, 8, self.storage)
    with self.assertRaises(ValueError): self.space.transaction().read(self.base + 64, 1)
    with self.assertRaises(ValueError): self.space.transaction().write(self.base - 1, b"x")
    self.space.unmap(self.base)
    with self.assertRaises(ValueError): self.space.transaction().read(self.base, 1)

  def test_changed_mapping_cannot_commit_stale_bytes(self):
    # Reusing an address after unmap must not accept an old transaction's writes.
    tx = self.space.transaction()
    tx.write(self.base, b"bad")
    self.space.unmap(self.base)
    self.space.map(self.base, 64, self.storage)
    with self.assertRaises(ValueError): tx.commit()
    self.assertEqual(bytes(self.storage)[:3], b"\x00\x01\x02")

  def test_readonly_host_memory_is_rejected_as_gpu_mapping(self):
    # An accepted mapping must allow staged GPU writes to commit safely.
    address = libc.mmap(0, 4096, mmap.PROT_READ, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    self.addCleanup(libc.munmap, address, 4096)
    with self.assertRaises(ValueError): self.space.map(address, 4096)

  def test_commit_preflights_all_write_permissions_before_publication(self):
    address = libc.mmap(0, 4096, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    self.addCleanup(libc.munmap, address, 4096)
    # Retaining an owner does not make a later protection change writable.
    self.space.map(address, 4096, self.storage)
    transaction = self.space.transaction()
    transaction.write(self.base, b"changed")
    transaction.write(address, b"later")
    libc.mprotect(address, 4096, mmap.PROT_READ)
    with self.assertRaises(ValueError): transaction.commit()
    self.assertEqual(bytes(self.storage)[:7], bytes(range(7)))

  def test_ioctl_output_span_is_validated_before_resource_creation(self):
    driver, fd = self.driver(lambda *args: (True, 1))
    address = libc.mmap(0, 4096, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    self.addCleanup(libc.munmap, address, 4096)
    libc.mprotect(address, 4096, mmap.PROT_READ)
    with self.assertRaises(ValueError):
      fd.ioctl(fd.fd, request_number(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE), address)
    self.assertEqual(driver.contexts, {})
    self.assertEqual(driver.next_context, 1)

  def test_ioctl_struct_alignment_precedes_resource_creation(self):
    driver, fd = self.driver(lambda *args: (True, 1))
    storage = (ctypes.c_ubyte * 32)()
    with self.assertRaises(ValueError):
      fd.ioctl(fd.fd, request_number(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE), ctypes.addressof(storage) + 1)
    self.assertEqual(driver.contexts, {})

  def test_unsupported_context_and_allocation_flags_do_not_create_resources(self):
    driver, fd = self.driver(lambda *args: (True, 1))
    with self.assertRaises(ValueError): self.ioctl(fd, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE, flags=kgsl.KGSL_CONTEXT_IFH_NOP)
    self.assertEqual(driver.contexts, {})
    with self.assertRaises(ValueError):
      self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, size=4096, mmapsize=4096, flags=kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (1 << 63))
    self.assertEqual(driver.allocations, {})

  def test_allocation_alignment_request_must_be_honored_or_rejected(self):
    driver, fd = self.driver(lambda *args: (True, 1))
    flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (13 << kgsl.KGSL_MEMALIGN_SHIFT)
    with self.assertRaises(ValueError):
      self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, size=4096, mmapsize=4096, flags=flags)
    self.assertEqual(driver.allocations, {})

  def test_timestamp_cannot_wrap_or_publish_an_unrepresentable_receipt(self):
    driver, fd = self.driver(lambda *args: self.fail("overflowing receipt executed"))
    context = self.context(fd)
    driver.contexts[context].queued = 0xffffffff
    with self.assertRaises(ValueError): self.submit(driver, fd, context, (0,))
    self.assertEqual(driver.contexts[context].queued, 0xffffffff)
    self.assertEqual(len(driver.contexts[context].pending), 0)

  def test_transaction_preflight_distinguishes_read_and_write_permissions(self):
    # PM4 prevalidation must check a write target without creating a full snapshot.
    address = libc.mmap(0, 4096, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    self.addCleanup(libc.munmap, address, 4096)
    self.space.map(address, 4096)
    libc.mprotect(address, 4096, mmap.PROT_READ)
    transaction = self.space.transaction()
    transaction.validate(address, 4)
    with self.assertRaises(ValueError): transaction.validate(address, 4, write=True)

  def test_signal_before_later_wait_allows_acyclic_context_dependencies(self):
    # A merged stream may publish A, wait B, then finish; rolling the first signal
    # back at that wait deadlocks the other context even though the graph is acyclic.
    for index in range(3): self.storage[index] = 0
    def execute(words, memory, timestamp, start=0):
      for cursor in range(start, len(words)):
        action = words[cursor]
        if action == 1: memory.write(self.base, b"\x01")
        elif action == 2:
          if memory.read(self.base + 1, 1) != b"\x01": return False, cursor
        elif action == 3: memory.write(self.base + 2, b"\x01")
        elif action == 4:
          if memory.read(self.base, 1) != b"\x01": return False, cursor
        elif action == 5: memory.write(self.base + 1, b"\x01")
      return True, len(words)
    driver, fd = self.driver(execute)
    first, second = self.context(fd), self.context(fd)
    self.submit(driver, fd, first, (1, 2, 3))
    self.assertEqual(bytes(self.storage)[:3], b"\x01\x00\x00")
    self.assertEqual(driver.contexts[first].retired, 0)
    self.submit(driver, fd, second, (4, 5))
    self.assertEqual(bytes(self.storage)[:3], b"\x01\x01\x01")
    self.assertEqual((driver.contexts[first].retired, driver.contexts[second].retired), (1, 1))

  def driver(self, executor):
    driver = self.module.QCOMDriver(executor)
    fd = driver.open("/dev/kgsl-3d0", os.O_RDWR, 0o777, driver.tracked_files[0])
    self.addCleanup(fd.close, fd.fd)
    driver.memory.map(self.base, 64, self.storage)
    return driver, fd

  def ioctl(self, fd, fn, **fields):
    req = fn.args[3](**fields)
    fd.ioctl(fd.fd, request_number(fn), ctypes.addressof(req))
    return req

  def context(self, fd): return self.ioctl(fd, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE).drawctxt_id

  def submit(self, driver, fd, context, words):
    data = (ctypes.c_uint32 * len(words))(*words)
    address = ctypes.addressof(data)
    driver.memory.map(address, ctypes.sizeof(data), data)
    self.addCleanup(driver.memory.unmap, address)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=address, size=ctypes.sizeof(data), flags=kgsl.KGSL_CMDLIST_IB)
    return self.ioctl(fd, kgsl.IOCTL_KGSL_GPU_COMMAND, context_id=context, cmdlist=ctypes.addressof(obj),
                      cmdsize=ctypes.sizeof(obj), numcmds=1)

  def test_scheduler_waits_roll_back_then_resume_fifo(self):
    # Advancing blocked commands, committing partial output, or bypassing FIFO fails.
    def execute(words, memory, timestamp, start=0):
      if words[0] == 1:
        if memory.read(self.base, 1) != b"S": return False, start
        memory.write(self.base + 8, b"resumed")
      elif words[0] == 2: memory.write(self.base, b"S")
      else: memory.write(self.base + 15, b"last")
      return True, 1
    driver, fd = self.driver(execute)
    first, second = self.context(fd), self.context(fd)
    request = self.submit(driver, fd, first, (1,))
    self.assertEqual(bytes(self.storage)[8:15], bytes(range(8, 15)))
    self.submit(driver, fd, first, (3,))
    self.assertEqual(bytes(self.storage)[15:19], bytes(range(15, 19)))
    self.submit(driver, fd, second, (2,))
    self.assertEqual(bytes(self.storage)[8:19], b"resumedlast")
    retired = self.ioctl(fd, kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID, context_id=first, type=kgsl.KGSL_TIMESTAMP_RETIRED)
    self.assertGreaterEqual(retired.timestamp, request.timestamp)

  def test_driver_allocation_free_leaves_cpu_unmap_to_runtime(self):
    # KGSL free must remove GPU access without double-unmapping host storage.
    driver, fd = self.driver(lambda *args: (True, 1))
    allocation = self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, size=4096, mmapsize=4096, flags=kgsl.KGSL_MEMFLAGS_USE_CPU_MAP)
    addr = fd.mmap(0, 4096, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, fd.fd, allocation.id * 4096)
    self.addCleanup(libc.munmap, addr, 4096)
    tx = driver.memory.transaction()
    tx.write(addr, b"mapped")
    tx.commit()
    self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_FREE, id=allocation.id)
    self.assertEqual(self.module.read_host(addr, 6), b"mapped")
    with self.assertRaises(ValueError): driver.memory.transaction().read(addr, 1)

  def test_close_releases_unfreed_owned_mapping(self):
    # A descriptor that closes without a GPUOBJ_FREE must not leak its mmap.
    driver, fd = self.driver(lambda *args: (True, 1))
    allocation = self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, size=4096, mmapsize=4096, flags=kgsl.KGSL_MEMFLAGS_USE_CPU_MAP)
    address = fd.mmap(0, 4096, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, fd.fd, allocation.id * 4096)
    fd.close(fd.fd)
    with self.assertRaises(ValueError): self.module.read_host(address, 1)

  def test_borrowed_mapping_is_owned_by_descriptor_but_not_freed_on_close(self):
    # Closing a descriptor removes its guest mapping but must retain caller bytes.
    driver, fd = self.driver(lambda *args: (True, 1))
    driver.memory.unmap(self.base)
    req = self.ioctl(fd, kgsl.IOCTL_KGSL_MAP_USER_MEM, hostptr=self.base, len=64, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    other = driver.open("/dev/kgsl-3d0", os.O_RDWR, 0o777, driver.tracked_files[0])
    self.addCleanup(other.close, other.fd)
    with self.assertRaises(ValueError): self.ioctl(other, kgsl.IOCTL_KGSL_SHAREDMEM_FREE, gpuaddr=req.gpuaddr)
    fd.close(fd.fd)
    self.assertEqual(bytes(self.storage)[:3], b"\x00\x01\x02")
    with self.assertRaises(ValueError): driver.memory.transaction().read(self.base, 1)

  def test_reentrant_scheduler_does_not_repeat_inflight_command(self):
    # Host memory callbacks may ask for progress while a command already runs.
    def execute(words, memory, timestamp, start=0):
      driver.drain()
      old = memory.read(self.base, 1)[0]
      memory.write(self.base, bytes([old + 1]))
      return True, 1
    driver, fd = self.driver(execute)
    self.submit(driver, fd, self.context(fd), (0,))
    self.assertEqual(self.storage[0], 1)

  def test_submit_shape_is_validated_before_execution(self):
    # Empty lists and foreign context IDs cannot become successful submissions.
    driver, fd = self.driver(lambda *args: self.fail("invalid request executed"))
    context = self.context(fd)
    with self.assertRaises(ValueError): self.ioctl(fd, kgsl.IOCTL_KGSL_GPU_COMMAND, context_id=context)
    with self.assertRaises(ValueError): self.submit(driver, fd, context + 100, (0,))

  def test_native_callback_latches_failure_and_preserves_original_cause(self):
    # ctypes must receive an integer; Python must subsequently observe the error.
    def execute(words, memory, timestamp, start=0):
      memory.write(self.base, b"bad")
      raise ValueError("unsupported test command")
    driver, fd = self.driver(execute)
    context = self.context(fd)
    data = (ctypes.c_uint32 * 1)(0)
    address = ctypes.addressof(data)
    driver.memory.map(address, 4, data)
    obj = kgsl.struct_kgsl_command_object(gpuaddr=address, size=4, flags=kgsl.KGSL_CMDLIST_IB)
    req = kgsl.struct_kgsl_gpu_command(context_id=context, cmdlist=ctypes.addressof(obj), cmdsize=ctypes.sizeof(obj), numcmds=1)
    callback = self.module.make_ioctl_callback({fd.fd: fd}, libc.dll.ioctl)
    self.assertEqual(callback(fd.fd, request_number(kgsl.IOCTL_KGSL_GPU_COMMAND), ctypes.addressof(req)), -1)
    self.assertEqual(bytes(self.storage)[:3], b"\x00\x01\x02")
    with self.assertRaisesRegex(RuntimeError, "unsupported test command") as raised: driver.check_error()
    self.assertIsInstance(raised.exception.__cause__, ValueError)
    self.assertEqual(callback(fd.fd, request_number(kgsl.IOCTL_KGSL_GPU_COMMAND), ctypes.addressof(req)), -1)
    with self.assertRaises(RuntimeError) as again: driver.check_error()
    self.assertIs(again.exception.__cause__, raised.exception.__cause__)

  def test_native_callback_forwards_real_descriptor(self):
    # A real pipe remains serviced by the OS, outside the QCOM tracked descriptors.
    import termios
    reader, writer = os.pipe()
    self.addCleanup(os.close, reader)
    self.addCleanup(os.close, writer)
    os.write(writer, b"hello")
    length = ctypes.c_int()
    callback = self.module.make_ioctl_callback({}, libc.dll.ioctl)
    self.assertEqual(callback(reader, termios.FIONREAD, ctypes.addressof(length)), 0)
    self.assertEqual(length.value, 5)

  def test_nv_and_qcom_descriptors_do_not_replace_each_other(self):
    # All mock drivers share tracked_fds, so backend-local ranges must differ.
    from test.mockgpu.nv.nvdriver import NVDriver
    driver, fd = self.driver(lambda *args: (True, 1))
    nv = NVDriver(gpus=0)
    nvfd = nv.open("/dev/nvidiactl", os.O_RDWR, 0o777, nv.tracked_files[0])
    descriptors = {fd.fd:fd, nvfd.fd:nvfd}
    self.assertEqual(len(descriptors), 2)
    self.assertIs(descriptors[fd.fd], fd)

  def test_hcq2_native_submission_reports_callback_failure_before_next_fence(self):
    # A callback failure must raise from ordinary Tensor.realize, without waiting
    # for a later copy-out or hanging in the next replay's native fence loop.
    code = '''from tinygrad import Device, Tensor
Device["QCOM"]
from test.mockgpu.mockgpu import drivers
from tinygrad.engine.realize import compile_linear, link_linear, run_linear
def reject(*args): raise ValueError("fixture command rejected")
drivers[0].execute = reject
output = Tensor.ones(1).contiguous()
linear, variables = Tensor.linear_with_vars(output)
linked = link_linear(compile_linear(linear))
for attempt in range(2):
  try:
    run_linear(linked, variables, jit=True)
  except RuntimeError as error:
    assert "fixture command rejected" in str(error), str(error)
    print("native callback failure surfaced", flush=True)
  else: raise AssertionError("failed submit returned successfully")
'''
    with tempfile.TemporaryDirectory() as cache:
      renderer = DEV.target("QCOM").renderer or "IR3"
      result = subprocess.run([sys.executable, "-c", code], cwd=pathlib.Path(__file__).parents[2],
                              env={**os.environ, "DEV":f"MOCK+QCOM:{renderer}", "PARALLEL":"0", "XDG_CACHE_HOME":cache},
                              capture_output=True, text=True, timeout=60)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(result.stdout.count("native callback failure surfaced"), 2)


if __name__ == "__main__": unittest.main()
