import ctypes, mmap, os, types, unittest
from unittest.mock import patch
from test.mockgpu.qcom import qcomdriver
from tinygrad.runtime.autogen import kgsl, libc


class QCOMDriverFoldingTests(unittest.TestCase):
  def test_darwin_copy_uses_source_destination_with_full_length_receipt(self):
    # Both directions have the same copy contract, including a short-copy veto.
    storage = (ctypes.c_ubyte * 4)(10, 20, 30, 40)
    address, calls = ctypes.addressof(storage), []
    incoming, outgoing = bytearray(4), bytearray(b"ABCD")
    def copy(task, source, size, destination, receipt):
      calls.append((source.value, size.value, destination.value))
      if source.value == address: incoming[:] = bytes(storage)
      else: storage[:] = outgoing
      receipt._obj.value = size.value
      return 0
    api = types.SimpleNamespace(mach_vm_read_overwrite=copy)
    task = types.SimpleNamespace(value=1)
    with patch.object(qcomdriver.sys, "platform", "darwin"), patch.object(qcomdriver.libc, "dll", api), \
         patch.object(qcomdriver.ctypes, "c_uint", wraps=ctypes.c_uint) as uint:
      uint.in_dll.return_value = task
      qcomdriver.host_transfer(address, incoming, False)
      self.assertEqual(incoming, bytes(storage))
      self.assertEqual(calls[-1][0], address)
      self.assertEqual(calls[-1][1], 4)
      qcomdriver.host_transfer(address, outgoing, True)
      self.assertEqual(bytes(storage), b"ABCD")
      self.assertEqual(calls[-1], (ctypes.addressof((ctypes.c_ubyte * 4).from_buffer(outgoing)), 4, address))
      def short_copy(task, source, size, destination, receipt):
        receipt._obj.value = size.value - 1
        return 0
      api.mach_vm_read_overwrite = short_copy
      for write in (False, True):
        with self.assertRaises(ValueError): qcomdriver.host_transfer(address, bytearray(4), write)

  def test_linux_copy_vectors_select_direction_and_reject_short_copies(self):
    storage = (ctypes.c_ubyte * 4)(10, 20, 30, 40)
    address, calls = ctypes.addressof(storage), []
    incoming, outgoing = bytearray(4), bytearray(b"ABCD")
    def copy(pid, local, local_count, remote, remote_count, flags):
      destination, source = local._obj, remote._obj
      calls.append((source.base, source.size, destination.base, destination.size))
      if source.base == address: incoming[:] = bytes(storage)
      else: storage[:] = outgoing
      return source.size
    api = types.SimpleNamespace(process_vm_readv=copy)
    with patch.object(qcomdriver.sys, "platform", "linux"), patch.object(qcomdriver.libc, "dll", api):
      qcomdriver.host_transfer(address, incoming, False)
      self.assertEqual(incoming, bytes(storage))
      self.assertEqual(calls[-1][0], address)
      self.assertEqual(calls[-1][1], 4)
      qcomdriver.host_transfer(address, outgoing, True)
      self.assertEqual(bytes(storage), b"ABCD")
      self.assertEqual(calls[-1], (ctypes.addressof((ctypes.c_ubyte * 4).from_buffer(outgoing)), 4, address, 4))
      def short_copy(*args): return 3
      api.process_vm_readv = short_copy
      for write in (False, True):
        with self.assertRaises(ValueError): qcomdriver.host_transfer(address, bytearray(4), write)

  def test_native_copy_rejects_inaccessible_source_and_destination(self):
    # Kernel carriers must report faults without Python dereferencing the pointer.
    for write in (False, True):
      with self.assertRaises(ValueError): qcomdriver.host_transfer(1, bytearray(4), write)

  def driver(self):
    driver = qcomdriver.QCOMDriver(lambda *args: (True, 1))
    fd = driver.open("/dev/kgsl-3d0", os.O_RDWR, 0o777, driver.tracked_files[0])
    self.addCleanup(fd.close, fd.fd)
    return driver, fd

  def ioctl(self, fd, fn, **fields):
    req = fn.args[3](**fields)
    fd.ioctl(fd.fd, qcomdriver.ioctl_number(fn), ctypes.addressof(req))
    return req

  def allocate(self, fd):
    req = self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_ALLOC, size=4096, mmapsize=4096, flags=kgsl.KGSL_MEMFLAGS_USE_CPU_MAP)
    address = fd.mmap(0, 4096, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, fd.fd, req.id * 4096)
    return req.id, address

  def test_resource_mapping_identity_preserves_transaction_tombstones(self):
    driver, fd = self.driver()
    key, address = self.allocate(fd)
    self.assertIs(driver.allocations[key], driver.memory.mappings[address])
    old = driver.memory.transaction()
    old.write(address, b"stale")
    self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_FREE, id=key)
    self.addCleanup(libc.munmap, address, 4096)
    self.ioctl(fd, kgsl.IOCTL_KGSL_MAP_USER_MEM, hostptr=address, len=4096, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    self.assertIs(driver.borrowed[address], driver.memory.mappings[address])
    with self.assertRaises(ValueError): old.commit()
    self.assertEqual(qcomdriver.read_host(address, 5), bytes(5))

  def test_close_keeps_foreign_owned_storage_and_caller_borrowed_storage(self):
    driver, fd = self.driver()
    other = driver.open("/dev/kgsl-3d0", os.O_RDWR, 0o777, driver.tracked_files[0])
    self.addCleanup(other.close, other.fd)
    _, first_address = self.allocate(fd)
    _, other_address = self.allocate(other)
    storage = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
    borrowed = ctypes.addressof(storage)
    self.ioctl(fd, kgsl.IOCTL_KGSL_MAP_USER_MEM, hostptr=borrowed, len=4, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    fd.close(fd.fd)
    with self.assertRaises(ValueError): qcomdriver.read_host(first_address, 1)
    with self.assertRaises(ValueError): driver.memory.transaction().read(borrowed, 1)
    self.assertEqual(qcomdriver.read_host(borrowed, 4), bytes(storage))
    self.assertEqual(driver.memory.transaction().read(other_address, 4), bytes(4))

  def test_free_invalid_fields_and_pending_work_preserve_resources(self):
    driver, fd = self.driver()
    key, address = self.allocate(fd)
    for field in ("flags", "priv", "type", "len"):
      with self.subTest(field=field), self.assertRaises(ValueError):
        self.ioctl(fd, kgsl.IOCTL_KGSL_GPUOBJ_FREE, id=key, **{field:1})
      self.assertEqual(driver.memory.transaction().read(address, 4), bytes(4))
    storage = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
    borrowed = ctypes.addressof(storage)
    self.ioctl(fd, kgsl.IOCTL_KGSL_MAP_USER_MEM, hostptr=borrowed, len=4, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    context = self.ioctl(fd, kgsl.IOCTL_KGSL_DRAWCTXT_CREATE).drawctxt_id
    driver.contexts[context].pending.append(((0,), 1, 0))
    for fn, fields in ((kgsl.IOCTL_KGSL_GPUOBJ_FREE, {"id":key}), (kgsl.IOCTL_KGSL_SHAREDMEM_FREE, {"gpuaddr":borrowed})):
      with self.assertRaises(ValueError): self.ioctl(fd, fn, **fields)
    self.assertEqual(driver.memory.transaction().read(borrowed, 4), bytes(storage))
    self.assertEqual(driver.memory.transaction().read(address, 4), bytes(4))
    driver.contexts[context].pending.clear()

  def test_borrowed_mapping_requires_full_range_liveness_even_without_writes(self):
    driver, fd = self.driver()
    size = 2 * mmap.PAGESIZE
    address = libc.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
    self.addCleanup(libc.munmap, address, size)
    self.ioctl(fd, kgsl.IOCTL_KGSL_MAP_USER_MEM, hostptr=address, len=size, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    transaction = driver.memory.transaction()
    transaction.read(address, 4)
    self.assertEqual(libc.mprotect(address + mmap.PAGESIZE, mmap.PAGESIZE, mmap.PROT_READ), 0)
    with self.assertRaises(ValueError): transaction.commit()


if __name__ == "__main__": unittest.main()
