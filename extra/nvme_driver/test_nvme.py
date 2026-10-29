#!/usr/bin/env python3
# NVME=0000:c1:00.0 [GPU=AMD:3] python3 extra/nvme_driver/test_nvme.py: write/read round trips through host memory and into gpu vram
import os, unittest
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.nvme import NVMeDev, PRPList, OP_READ, OP_WRITE, PAGE

def vram_pages(buf:Buffer) -> list[int]:
  iface = Device[buf.device].iface
  bar, bar_size = iface.pci_dev.bar_info(iface.vram_bar)
  # a drive writing past the bar hits nothing: a completion timeout takes the machine down
  paddrs = buf.get_storage().meta.mapping.paddrs
  assert all(p + sz <= bar_size for p, sz in paddrs), f"{buf.device}: vram past the {bar_size:#x} bar"
  return [bar + p + o for p, sz in paddrs for o in range(0, sz, PAGE)]

class TestNVMe(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = NVMeDev(PCIDevice("nvme", os.environ.get("NVME", "0000:c1:00.0")))
    # the tests write only the last 64 MiB of the drive: the data of the drive stays
    cls.base = (cls.dev.size - (64 << 20)) >> cls.dev.lba_shift
  @classmethod
  def tearDownClass(cls): cls.dev.fini()

  def rw(self, opcode, lba, pages, nbytes):
    self.dev.io(opcode, lba, nbytes, *PRPList(self.dev, pages)(0, nbytes))
    self.dev.io_q.kick()
    self.dev.io_q.wait()

  def test_prp_offsets(self):
    # max_xfer transfers starting at every page of a buffer: whatever the start, the list must describe exactly its pages
    n, sz = 1200 * PAGE, self.dev.max_xfer
    src, src_addrs = self.dev.pci_dev.alloc_sysmem(n)
    dst, dst_addrs = self.dev.pci_dev.alloc_sysmem(n)
    src.mv[:] = (pat:=os.urandom(n))
    sp, dp = PRPList(self.dev, src_addrs), PRPList(self.dev, dst_addrs)
    for start in range(0, n - sz + 1, PAGE):
      for op, prps in ((OP_WRITE, sp), (OP_READ, dp)):  # commands in a queue aren't ordered: the write completes first
        self.dev.io(op, self.base, sz, *prps(start, sz))
        self.dev.io_q.kick()
        self.dev.io_q.wait()
      self.assertEqual(bytes(dst.mv[start:start + sz]), pat[start:start + sz], f"start page {start // PAGE}")

  def test_identify(self):
    self.assertGreater(self.dev.size, 0)
    self.assertGreaterEqual(self.dev.max_xfer, 2 * PAGE)

  def test_host_roundtrip(self):
    for nbytes in (PAGE, 2 * PAGE, 3 * PAGE, self.dev.max_xfer):
      src, src_addrs = self.dev.pci_dev.alloc_sysmem(nbytes)
      dst, dst_addrs = self.dev.pci_dev.alloc_sysmem(nbytes)
      src.mv[:] = (pat:=os.urandom(nbytes))
      self.rw(OP_WRITE, self.base, src_addrs, nbytes)
      self.rw(OP_READ, self.base, dst_addrs, nbytes)
      self.assertEqual(bytes(dst.mv), pat, f"{nbytes} bytes")

  def test_vram_roundtrip(self):
    n = self.dev.max_xfer
    src, src_addrs = self.dev.pci_dev.alloc_sysmem(4 * n)
    src.mv[:] = (pat:=os.urandom(4 * n))
    for i in range(4): self.rw(OP_WRITE, self.base + ((i * n) >> self.dev.lba_shift), src_addrs[i * n // PAGE:(i + 1) * n // PAGE], n)
    buf = Buffer(os.environ.get("GPU", "AMD:3"), 4 * n).allocate()
    pages = vram_pages(buf)
    for i in range(4): self.rw(OP_READ, self.base + ((i * n) >> self.dev.lba_shift), pages[i * n // PAGE:(i + 1) * n // PAGE], n)
    self.assertEqual(bytes(buf.as_memoryview()), pat)

if __name__ == "__main__":
  unittest.main()
