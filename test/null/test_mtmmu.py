import unittest
from tinygrad.runtime.support.mt.mmu import MTMMU4, MT_PTE_READ_ONLY, MT_PTE_READ_WRITE

class TestMTMMU4(unittest.TestCase):
  def test_map_and_share_tables(self):
    m = MTMMU4(0x30000000)
    va, pa = 0xe1c0000000, 0x20000000
    m.map(va, pa, MT_PTE_READ_WRITE)
    m.map(va + 0x1000, pa + 0x1000, MT_PTE_READ_WRITE)
    pci = (va >> 30) & 0x3ff
    pdi = (va >> 21) & 0x1ff
    self.assertEqual(int.from_bytes(m.root[pci*4:pci*4+4], 'little'), ((m.pd[pci] >> 12) << 4) | 1)
    pd_off = m.pd[pci] - m.root_pa + pdi * 8
    self.assertEqual(int.from_bytes(m.root[pd_off:pd_off+8], 'little'), m.pt[pci,pdi] | 1)
    for i in range(2):
      off = m.pt[pci,pdi] - m.root_pa + i * 8
      self.assertEqual(int.from_bytes(m.root[off:off+8], 'little'), pa + i * 0x1000 | MT_PTE_READ_WRITE)
    self.assertEqual(len(m.pd), 1)
    self.assertEqual(len(m.pt), 1)
    self.assertEqual(m.pages[va + 0x1000], pa + 0x1000)

  def test_read_only_and_range(self):
    m = MTMMU4(0x30000000)
    m.map_range(0x8000000000, 0x40000000, 0x3000)
    for i in range(3):
      va = 0x8000000000 + i * 0x1000
      pci, pdi = (va >> 30) & 0x3ff, (va >> 21) & 0x1ff
      off = m.pt[pci,pdi] - m.root_pa + i * 8
      self.assertEqual(int.from_bytes(m.root[off:off+8], 'little'), 0x40000000 + i * 0x1000 | MT_PTE_READ_ONLY)

  def test_idempotent_mapping(self):
    m = MTMMU4(0x30000000)
    va, pa = 0xe1c0000000, 0x20000000
    m.map(va, pa, MT_PTE_READ_WRITE)
    dirty = m.dirty_pages.copy()
    m.map(va, pa, MT_PTE_READ_WRITE)
    self.assertEqual(m.dirty_pages, dirty)
    m.map(va, pa, MT_PTE_READ_ONLY)
    self.assertEqual(m.dirty_pages, dirty)
    self.assertEqual(m.page_flags[va], MT_PTE_READ_ONLY)
    pci, pdi, pti = (va>>30)&0x3ff, (va>>21)&0x1ff, (va>>12)&0x1ff
    off = m.pt[pci,pdi] - m.root_pa + pti * 8
    self.assertEqual(int.from_bytes(m.root[off:off+8], 'little'), pa | MT_PTE_READ_ONLY)

  def test_idempotent_range_invalidation(self):
    m = MTMMU4(0x30000000)
    va, pa = 0xe1c0000000, 0x20000000
    m.map_range(va, pa, 0x3000, MT_PTE_READ_WRITE)
    dirty = m.dirty_pages.copy()
    m.map_range(va, pa, 0x3000, MT_PTE_READ_WRITE)
    self.assertEqual(m.dirty_pages, dirty)
    m.map(va + 0x1000, pa + 0x10000, MT_PTE_READ_WRITE)
    self.assertEqual(m.pages[va + 0x1000], pa + 0x10000)
    m.map_range(va, pa, 0x3000, MT_PTE_READ_WRITE)
    self.assertEqual(m.pages[va + 0x1000], pa + 0x1000)
    self.assertEqual(m.page_flags[va + 0x1000], MT_PTE_READ_WRITE)

  def test_invalid_and_exhaustion(self):
    m = MTMMU4(0x30000000, size=0x3000)
    with self.assertRaises(ValueError): m.map(1, 0)
    with self.assertRaises(ValueError): m.map(0, 1)
    with self.assertRaises(ValueError): m.map(0, 0, 0)
    m.map(0, 0)
    with self.assertRaises(MemoryError): m.map(1 << 30, 0)

if __name__ == '__main__': unittest.main()
