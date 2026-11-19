import struct, unittest
from tinygrad.runtime.support.mt.pds import compute_pds, PDS_UNUSED

class TestMTPDS(unittest.TestCase):
  def words(self, data): return list(struct.unpack(f'<{len(data)//4}I',data))
  def test_empty(self):
    p = compute_pds(0x81e0e00000)
    self.assertEqual(self.words(p.code),[0xc2c00002,0xd3000000,0xf4000002,0xd3000000])
    self.assertEqual(struct.unpack('<Q',p.data)[0],0x81e0e00000)
    self.assertEqual(p.temps_used,3)
  def test_ids(self):
    p = compute_pds(0x81e0e00000,local_regs=(0,1,PDS_UNUSED),workgroup_regs=(0,1,2))
    code = self.words(p.code)
    self.assertEqual(code[0],0xc2c00004)
    self.assertEqual(code[1:3],[0xf0000001 | (2<<16) | (64<<8),0xf0000001 | (3<<16) | (65<<8)])
    self.assertEqual(code[3],0xd3000000)
    self.assertEqual(code[4],0xf0000001 | (4<<16) | (64<<8))
    self.assertEqual(code[-2:],[0xf4000002,0xd3000000])
    data = self.words(p.data)
    self.assertEqual(data[2],(0<<13)|2|(1<<28))
    self.assertEqual(data[3],(2<<13)|1|(1<<28)|(1<<31))
    self.assertEqual(data[4],(0<<13)|2|(1<<31))
  def test_barrier_zero_constant(self):
    p = compute_pds(0x81e0e00000,barrier_coefficient=4)
    data = self.words(p.data)
    self.assertEqual(data[2:4],[0,0])
    self.assertEqual(data[4],(4<<13)|2|(1<<28)|(1<<31))
    self.assertEqual(self.words(p.code)[1],0xf0000001 | (4<<16) | (1<<8))

if __name__ == '__main__': unittest.main()
