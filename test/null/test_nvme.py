import gc, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import Context, mv_address
from tinygrad.runtime.ops_nvme import NVMEAllocator
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface, VirtMapping
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta
from tinygrad.uop.ops import UOp

class MockDevice:
  def __init__(self, device):
    self.device, self.peer_group, self.nvme = device, "test", None
    self.iface = PCIIfaceBase.__new__(PCIIfaceBase)
    self.iface.dev, self.iface.pci_dev = self, SimpleNamespace(peer_group=self.peer_group)
  def synchronize(self): pass

class MockGPUAllocator(Allocator):
  def __init__(self, dev):
    super().__init__(dev)
    self.mapped, self.unmapped = set(), []
  def _alloc(self, size, options):
    return BufferStorage(0x10000, PCIAllocationMeta(VirtMapping(0x10000, size, [(0x10000, size)], AddrSpace.SYS), False))
  def _map(self, buf):
    self.mapped.add(addr:=buf._buf)
    return BufferStorage(addr)
  def _unmap(self, storage):
    self.mapped.remove(storage.buf)
    self.unmapped.append(storage.buf)
  def _offset(self, buf, size, offset): return buf + offset

class TestNVME(unittest.TestCase):
  def test_prp_mapping_survives_uop_cycle(self): self._test_prp_mapping_lifetime(wrap_uop=True)
  def test_prp_mapping_survives_buffer_cycle(self): self._test_prp_mapping_lifetime(wrap_uop=False)

  def _test_prp_mapping_lifetime(self, wrap_uop):
    gc.collect()
    gpu, nvme = MockDevice("NULL"), MockDevice("NVME")
    gpu.allocator, nvme.allocator = MockGPUAllocator(gpu), NVMEAllocator(nvme)
    memory = bytearray(4096)
    table = MMIOInterface(mv_address(memoryview(memory)), len(memory))
    with Context(LRU=1), patch.object(type(Device), '__getitem__', lambda _, key: {"NULL": gpu, "NVME": nvme}[key]), \
         patch('tinygrad.runtime.ops_nvme.prp_table', return_value=(table, [0x20000])):
      buf = Buffer("NULL", 4096, allocator=gpu.allocator, preallocate=True)
      addr = buf.get_buf("NVME")
      self.assertIn(addr, gpu.allocator.mapped)
      holder = [UOp.from_buffer(buf, dtypes.uint8) if wrap_uop else buf]
      holder.append(holder)
      del buf, holder
      gc.collect()
      reused = None
      try:
        self.assertEqual(len(gpu.allocator.cache[(4096, BufferSpec())]), 1)
        self.assertIn(addr, gpu.allocator.mapped)
        reused = Buffer("NULL", 4096, allocator=gpu.allocator, preallocate=True)
        self.assertEqual(reused.get_buf("NVME"), addr)
        self.assertIn(addr, gpu.allocator.mapped)
      finally:
        if reused is not None: reused.deallocate()
        gpu.allocator.free_cache()
      self.assertEqual(gpu.allocator.unmapped, [addr])
      self.assertFalse(gpu.allocator.mapped)
      self.assertFalse(nvme.allocator.prp_tables)

if __name__ == '__main__': unittest.main()
