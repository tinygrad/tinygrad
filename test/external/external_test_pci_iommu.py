# Verifies IOMMU containment of misbehaving device DMA: pages revoked from the vfio container but still mapped in the GPU's
# page tables must fault in the IOMMU (IO_PAGE_FAULT) instead of reaching host memory. Requires an active IOMMU (VFIO type1v2).
# Run with: DEV=PCI:0+AMD python3 test/external/external_test_pci_iommu.py
import subprocess, unittest
from tinygrad import Device
from tinygrad.device import BufferSpec
from tinygrad.runtime.support.system import PCIAllocationMeta
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.hcq import HCQBuffer

class TestPCIIOMMU(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev, cls.pci_dev, cls.mm = (d:=Device[Device.DEFAULT]), d.iface.pci_dev, d.iface.dev_impl.mm
    if not cls.pci_dev.iommu: raise unittest.SkipTest("requires an active IOMMU")

  def bad_buf(self, paddr:int) -> HCQBuffer:
    bp = self.mm.map_range(va:=self.mm.alloc_vaddr(0x1000), 0x1000, [(paddr, 0x1000)], aspace=AddrSpace.SYS, snooped=True, uncached=True)
    return HCQBuffer(va, 0x1000, meta=PCIAllocationMeta(bp, has_cpu_mapping=False), owner=self.dev)

  def test_wild_dma_is_contained(self):
    N, pages = 64, []
    for i in range(N):
      view, paddrs = self.pci_dev.alloc_sysmem(0x1000) # legit sysmem page: pinned in the vfio container
      view[:0x1000] = (b"SENTINEL" + i.to_bytes(2, 'little')) + bytes(0x1000 - 10)
      self.pci_dev.dma_unmap(paddrs) # revoke it: from now on any device DMA to it must fault in the IOMMU
      pages.append((view, self.bad_buf(paddrs[0])))

    src = self.dev.allocator._alloc(0x1000, BufferSpec())

    # storm the IOMMU with wild writes (a valid GART entry pointing at a revoked page == misbehaving GPU)
    q = self.dev.hw_copy_queue_t()
    for _, bad in pages: q.copy(bad, src, 0x1000)
    q.signal(self.dev.timeline_signal, tlv:=self.dev.next_timeline()).submit(self.dev)
    self.dev.timeline_signal.wait(tlv, timeout=10000)

    # and a wild read for good measure
    self.dev.hw_copy_queue_t().copy(src, pages[0][1], 0x1000).signal(self.dev.timeline_signal, tlv:=self.dev.next_timeline()).submit(self.dev)
    self.dev.timeline_signal.wait(tlv, timeout=10000)

    # none of the wild DMA may have reached host memory, and there must be no hardware error (MCE)
    for i, (view, _) in enumerate(pages): self.assertEqual(bytes(view[:10]), b"SENTINEL" + i.to_bytes(2, 'little'))
    hw_errs = subprocess.run("journalctl -k --no-pager --since '-60s' | grep -ci 'Hardware Error' || true",
                             shell=True, capture_output=True, text=True).stdout.strip()
    self.assertIn(hw_errs, ("", "0"), f"unexpected hardware errors in the kernel log: {hw_errs}")

  def test_device_survives_faults(self):
    view, paddrs = self.pci_dev.alloc_sysmem(0x1000)
    view[:0x1000] = b"IOMMU-OK!" + bytes(0x1000 - 9)
    self.pci_dev.dma_unmap(paddrs)

    src = self.dev.allocator._alloc(0x1000, BufferSpec())
    self.dev.hw_copy_queue_t().copy(self.bad_buf(paddrs[0]), src, 0x1000) \
                              .signal(self.dev.timeline_signal, tlv:=self.dev.next_timeline()).submit(self.dev)
    self.dev.timeline_signal.wait(tlv, timeout=10000)
    self.assertEqual(bytes(view[:9]), b"IOMMU-OK!")

    # device is still usable after the fault
    self.dev.allocator._copyout(mv:=memoryview(bytearray(4)), src)
    self.assertEqual(len(mv), 4)

if __name__ == "__main__": unittest.main()
