import struct, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad.runtime.ops_mt import MTDevice, MT_SPILL_WORKGROUP_KERNEL8
from tinygrad.runtime.support.mt.generated import MTGeneratedLaunch
from tinygrad.runtime.support.mt.launch import MTLaunch
from tinygrad.runtime.support.mt.mmu import MTMMU4


def image(nbytes, private=76):
  return SimpleNamespace(text=bytes(nbytes),metadata=SimpleNamespace(
    kernel={'.args':[],'.private_memory_size':private,'.local_mem_used_in_cluster':0,'.temp_reg_count':254,
            '.shared_reg_count':103,'.shared_memory_size':20,'.has_barrier':False},immediates=[]))

def generated(nbytes, private=76):
  launch = MTGeneratedLaunch.__new__(MTGeneratedLaunch)
  launch.backing_offsets = {0xda00406000:0,0x81e1200000:0x1000,0x81e1400000:0x2000,0xe000000000:0x3000}
  launch.write_map = {}
  launch.text_va = 0xe000000000
  launch.backing_pa = 0x40000000
  launch.mmu = MTMMU4(0)
  launch.generation = 1
  return launch

class TestMTLaunchLayout(unittest.TestCase):
  def test_spill_text_regions(self):
    small = generated(0x40bc)
    small.set_generated(image(0x40bc))
    self.assertEqual(small.text_offset,0x8000)
    large = generated(0x8930)
    large.set_generated(image(0x8930))
    self.assertEqual(large.text_offset,0x30000)

  def test_multithread_spill_kernel8(self):
    dev = MTDevice.__new__(MTDevice)
    dev.spill_buffer = None
    launch = generated(0)
    launch.kernel_lib = b'kernel'
    launch.core_count = 8
    launch.mmu_dirty = False
    launch.map_buffer = Mock()
    launch.set_spill = Mock()
    launch.set_grid = Mock()
    launch._write_backing = Mock()
    MTLaunch._loaded = launch
    dev.current_launch = launch
    dev.allocator = Mock()
    dev.allocator.alloc.return_value = SimpleNamespace(buf=0x8001000000,meta=SimpleNamespace(pa=0x80000000,size=0x1000))
    obj = SimpleNamespace(lib=b'kernel')
    with patch('tinygrad.runtime.ops_mt.MTKernelImage.from_elf',return_value=image(0)):
      dev.activate_launch(launch,obj,(5,1,1),(4,1,1))
    launch._write_backing.assert_any_call(launch.backing_offsets[0x81e1400000]+0x20,struct.pack('<I',MT_SPILL_WORKGROUP_KERNEL8))

if __name__ == '__main__': unittest.main()
