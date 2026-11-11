import gc, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad.device import Allocator, Buffer, BufferStorage, Device
from tinygrad.helpers import Context
from tinygrad.runtime.ops_nvme import NVMEAllocator
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.system import PCIIfaceBase

class TestNVME(unittest.TestCase):
  def test_prp_mapping_survives_buffer_cycle(self):
    gc.collect()
    gpu = Mock(device="NULL", iface=Mock(spec=PCIIfaceBase, peer_group="test"))
    nvme = Mock(device="NVME", peer_group="test")
    gpu.allocator, nvme.allocator = Allocator(gpu), NVMEAllocator(nvme)
    mapping = SimpleNamespace(paddrs=[(0x1000, 4096)], aspace=AddrSpace.SYS)
    gpu.allocator._alloc = lambda size, options: BufferStorage(0x1000, SimpleNamespace(mapping=mapping))
    # A recording mock for _map would retain the table Buffer and hide the GC bug.
    gpu.allocator._map = lambda buf: BufferStorage(id(buf))
    released = []
    gpu.allocator._unmap = lambda storage: released.append(storage.buf)
    nvme.iface.buffer.side_effect = lambda *_: Buffer("NVME", 4096, preallocate=True)
    with Context(LRU=1), patch.object(type(Device), '__getitem__', lambda _, key: {"NULL": gpu, "NVME": nvme}[key]), \
         patch('tinygrad.runtime.ops_nvme.prp_table', return_value=(None, [])):
      buf = Buffer("NULL", 4096, allocator=gpu.allocator, preallocate=True)
      addr = buf.get_buf("NVME")
      holder = [buf]
      holder.append(holder)
      del buf, holder
      reused = None
      try:
        gc.collect()
        self.assertEqual(released, [])
        reused = Buffer("NULL", 4096, allocator=gpu.allocator, preallocate=True)
        self.assertEqual(reused.get_buf("NVME"), addr)
        reused.deallocate()
        reused = None
        self.assertEqual(released, [])
      finally:
        if reused is not None: reused.deallocate()
        gpu.allocator.free_cache()
      self.assertEqual(released, [addr])

if __name__ == '__main__': unittest.main()
