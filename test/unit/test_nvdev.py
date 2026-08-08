import ctypes, types, unittest
from unittest.mock import MagicMock, patch

from tinygrad.runtime.autogen import nv_regs
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import AddrSpace, TLSFAllocator
from tinygrad.runtime.support.nv.ip import NV_FLCN
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager, NVPageTableEntry, NVReg


class StaticDecoder:
  def __init__(self, fields): self.fields = fields
  def decode(self, value): return self.fields


class TestMemoryManagerAllocation(unittest.TestCase):
  def test_contiguous_valloc_can_skip_zeroing(self):
    mm = object.__new__(NVMemoryManager)
    mm.cpu_visible_pa_allocator = object()
    mm.alloc_vaddr, mm.palloc_cpu_visible, mm.map_range = MagicMock(return_value=0x100000), MagicMock(return_value=0x200000), MagicMock()
    mapping = mm.map_range.return_value

    self.assertIs(mm.valloc_cpu_visible(0x1800, zero=False), mapping)

    mm.palloc_cpu_visible.assert_called_once_with(0x2000, zero=False)
    mm.map_range.assert_called_once_with(0x100000, 0x2000, [(0x200000, 0x2000)], aspace=AddrSpace.PHYS, uncached=False)

class TestNVBootMemory(unittest.TestCase):
  @staticmethod
  def make_dev(boot_mem_in_vram:bool):
    dev = object.__new__(NVDev)
    dev.large_bar = False
    dev.mm = types.SimpleNamespace(palloc=MagicMock(return_value=0x123000), palloc_cpu_visible=MagicMock(return_value=0x123000))
    dev.vram = MagicMock()
    dev.vram.view.return_value = MagicMock()
    dev.pci_dev = MagicMock(boot_mem_in_vram=boot_mem_in_vram)
    dev.pci_dev.bar_info.return_value = (0x800000000, 0x600000000)
    return dev

  def test_usb_vram_boot_memory_reports_gpu_offsets(self):
    dev = self.make_dev(True)

    _, paddr, paddrs = dev._alloc_boot_mem(0x1800)

    self.assertEqual(paddr, 0x123000)
    self.assertEqual(paddrs, [0x123000, 0x124000])
    dev.mm.palloc_cpu_visible.assert_called_once_with(0x2000, boot=False)
    dev.mm.palloc.assert_not_called()
    dev.pci_dev.bar_info.assert_not_called()

  def test_normal_vram_boot_memory_reports_bar_addresses(self):
    dev = self.make_dev(False)

    _, paddr, paddrs = dev._alloc_boot_mem(0x1800, sysmem=False)

    self.assertEqual(paddr, 0x123000)
    self.assertEqual(paddrs, [0x800123000, 0x800124000])
    dev.mm.palloc.assert_called_once_with(0x2000, boot=False)
    dev.mm.palloc_cpu_visible.assert_not_called()
    dev.pci_dev.bar_info.assert_called_once_with(1)


class TestNVStaleState(unittest.TestCase):
  def test_usb_wpr2_defers_to_flr_after_selecting_falcon(self):
    dev = object.__new__(NVDev)
    dev.devfmt, dev.include = "usb:0", MagicMock()
    wpr, boot, details = MagicMock(), MagicMock(), MagicMock()
    wpr.read.return_value, boot.read.return_value = 1, 0
    details.read_bitfields.return_value = {'architecture':0x17, 'implementation':2}
    dev.reg = MagicMock(side_effect=lambda name: {
      "NV_PFB_PRI_MMU_WPR2_ADDR_HI":wpr, "NV_PMC_BOOT_0":boot, "NV_PMC_BOOT_42":details}[name])
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True, gsp_flr_recovery=True, reset=MagicMock(),
      read_config=MagicMock(return_value=0), write_config_flush=MagicMock())

    with patch.object(NVDev, "_recover_stale_wpr", autospec=True) as recover: dev._early_ip_init()

    recover.assert_called_once_with(dev)
    self.assertIsInstance(dev.flcn, NV_FLCN)
    dev.pci_dev.reset.assert_not_called()

  def test_usb_wpr2_refuses_unproven_hot_reset(self):
    dev = object.__new__(NVDev)
    dev.devfmt, dev.include = "usb:0", MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.return_value = 1
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True, gsp_flr_recovery=False, reset=MagicMock())

    with self.assertRaisesRegex(RuntimeError, "physically power-cycle"): dev._early_ip_init()

    dev.pci_dev.reset.assert_not_called()


class TestNVStartupTeardown(unittest.TestCase):
  @staticmethod
  def make_teardown_dev(wpr_reads):
    dev = object.__new__(NVDev)
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.side_effect = wpr_reads
    dev.gsp = types.SimpleNamespace(rpc_rm_free=MagicMock(), fini_hw=MagicMock())
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.shutdown_fwsec, dev.flcn.shutdown_booter = MagicMock(), MagicMock()
    return dev

  def test_full_teardown_has_explicit_order_and_is_hardware_idempotent(self):
    events, dev = [], self.make_teardown_dev([1, 0, 0])
    dev.gsp.rpc_rm_free.side_effect = lambda obj, client: events.append(("free", obj, client))
    dev.gsp.fini_hw.side_effect = lambda: events.append(("gsp_suspend",))
    dev.flcn.shutdown_fwsec.side_effect = lambda: events.append(("fwsec",))
    dev.flcn.shutdown_booter.side_effect = lambda: events.append(("booter_unload",))

    dev.fini(0xc1000000)
    dev.fini(0xc1000000)

    self.assertEqual(events, [("free", 0xc1000000, 0xc1000000), ("gsp_suspend",), ("fwsec",), ("booter_unload",)])

  def test_failed_gsp_suspend_still_runs_secure_teardown(self):
    dev = self.make_teardown_dev([1, 0])
    dev.gsp.fini_hw.side_effect = RuntimeError("not suspended")

    with self.assertRaisesRegex(RuntimeError, "not suspended"): dev.fini(0xc1000000)

    dev.flcn.shutdown_fwsec.assert_called_once_with()
    dev.flcn.shutdown_booter.assert_called_once_with()

  def test_secure_teardown_collects_failures_and_attempts_booter(self):
    dev = self.make_teardown_dev([1, 0])
    suspend_error, fwsec_error = RuntimeError("no suspend"), RuntimeError("fwsec failed")
    dev.gsp.fini_hw.side_effect, dev.flcn.shutdown_fwsec.side_effect = suspend_error, fwsec_error

    with self.assertRaises(ExceptionGroup) as raised: dev.fini(0xc1000000)

    self.assertEqual(raised.exception.exceptions, (suspend_error, fwsec_error))
    dev.flcn.shutdown_booter.assert_called_once_with()

  def test_final_wpr_read_failure_is_reported(self):
    dev = self.make_teardown_dev([1, RuntimeError("WPR read failed")])

    with self.assertRaisesRegex(RuntimeError, "WPR read failed"): dev.fini(0xc1000000)

  def test_full_initialize_prepares_shutdown_before_wpr(self):
    events = []
    flcn, gsp = object.__new__(NV_FLCN), types.SimpleNamespace()
    flcn.init_sw = lambda: events.append("flcn_sw")
    flcn.prep_fini = lambda: events.append("prep")
    flcn.init_wpr = lambda: events.append("wpr")
    flcn.boot_gsp = lambda: events.append("boot")
    gsp.init_sw, gsp.init_hw = lambda: events.append("gsp_sw"), lambda: events.append("rm")
    pci_dev = types.SimpleNamespace(pcibus="usb:0", map_bar=MagicMock(return_value="mmio"), gsp_full_teardown=True)

    def early_init(dev):
      events.append("pci")
      dev.flcn, dev.gsp = flcn, gsp

    with patch.object(NVDev, "_early_ip_init", early_init), \
         patch.object(NVDev, "_early_mmu_init", lambda dev: events.append("mmu")):
      NVDev(pci_dev)

    self.assertEqual(events, ["pci", "mmu", "flcn_sw", "gsp_sw", "prep", "wpr", "boot", "rm"])

  @staticmethod
  def make_flr_recovery_dev():
    dev = object.__new__(NVDev)
    dev.devfmt = "usb:0"
    dev.pci_dev = types.SimpleNamespace(function_level_reset=MagicMock(), map_bar=MagicMock(return_value="remapped-mmio"))
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.wait_for_reset = MagicMock()
    return dev

  def test_stale_wpr_recovery_uses_flr_then_starts_from_reset_state(self):
    dev = self.make_flr_recovery_dev()

    dev._recover_stale_wpr()

    dev.pci_dev.function_level_reset.assert_called_once_with()
    dev.pci_dev.map_bar.assert_called_once_with(0, fmt='I')
    self.assertEqual(dev.mmio, "remapped-mmio")
    dev.flcn.wait_for_reset.assert_called_once_with()

  def test_stale_wpr_recovery_flr_failure_does_not_continue(self):
    dev = self.make_flr_recovery_dev()
    dev.pci_dev.function_level_reset.side_effect = RuntimeError("FLR failed")

    with self.assertRaisesRegex(RuntimeError, "FLR failed"): dev._recover_stale_wpr()

    dev.pci_dev.function_level_reset.assert_called_once_with()
    dev.pci_dev.map_bar.assert_not_called()
    dev.flcn.wait_for_reset.assert_not_called()


class TestNVCPUVisibleMemory(unittest.TestCase):
  def test_cpu_visible_and_normal_vram_use_disjoint_allocators(self):
    storage = (ctypes.c_ubyte * (32 << 20))()
    dev = types.SimpleNamespace(
      is_booting=True, smi_dev=False, devfmt="mock", mmu_ver=2,
      vram=MMIOInterface(ctypes.addressof(storage), len(storage)),
      NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE=MagicMock())
    dev.pte_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PTE'][2])
    dev.pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PDE'][2])
    dev.dual_pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_DUAL_PDE'][2])
    NVMemoryManager.va_allocator = TLSFAllocator(1 << 44, base=0x1000000000)
    dev.mm = NVMemoryManager(dev, len(storage), boot_size=2 << 20, pt_t=NVPageTableEntry, va_bits=48,
                             va_shifts=[12, 21, 29, 38, 47], va_base=0,
                             palloc_ranges=[(2 << 20, 2 << 20), (4 << 10, 4 << 10)], reserve_ptable=True,
                             cpu_visible_limit=8 << 20)
    dev.is_booting = False

    low = dev.mm.palloc_cpu_visible(2 << 20, zero=False)
    normal = dev.mm.palloc(2 << 20, zero=False)
    self.assertEqual(low, 3 << 20)
    self.assertEqual(normal, 8 << 20)

    dev.mm.pfree(low)
    dev.mm.pfree(normal)
    self.assertEqual(dev.mm.palloc_cpu_visible(2 << 20, zero=False), low)
    self.assertEqual(dev.mm.palloc(2 << 20, zero=False), normal)

    with self.assertRaisesRegex(MemoryError, "Can't allocate"):
      dev.mm.palloc_cpu_visible(4 << 20, zero=False)

class TestNVPageTableEntry(unittest.TestCase):
  @staticmethod
  def make_dual_entry(is_page:bool):
    entry = object.__new__(NVPageTableEntry)
    entry.lv, entry.entries = 3, [0] * 512
    entry.entries[24] = int(is_page)
    entry.nvdev = types.SimpleNamespace(
      mm=types.SimpleNamespace(level_cnt=5), mmu_ver=2,
      pte_t=StaticDecoder({'address_sys': 0xE600}),
      dual_pde_t=StaticDecoder({'address_small_sys': 0x214}),
      pde_t=StaticDecoder({}))
    return entry

  def test_dual_level_huge_pte_uses_pte_address(self):
    self.assertEqual(self.make_dual_entry(is_page=True).address(12), 0xE600000)

  def test_dual_level_table_uses_small_table_address(self):
    self.assertEqual(self.make_dual_entry(is_page=False).address(12), 0x214000)

  def test_adjacent_huge_mapping_preserves_existing_path(self):
    storage = (ctypes.c_ubyte * (16 << 20))()
    dev = types.SimpleNamespace(
      is_booting=True, smi_dev=False, devfmt="mock", mmu_ver=2,
      vram=MMIOInterface(ctypes.addressof(storage), len(storage)),
      NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE=MagicMock())
    dev.pte_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PTE'][2])
    dev.pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PDE'][2])
    dev.dual_pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_DUAL_PDE'][2])
    NVMemoryManager.va_allocator = TLSFAllocator(1 << 44, base=0x1000000000)
    dev.mm = NVMemoryManager(dev, len(storage), boot_size=2 << 20, pt_t=NVPageTableEntry, va_bits=48,
                             va_shifts=[12, 21, 29, 38, 47], va_base=0,
                             palloc_ranges=[(2 << 20, 2 << 20), (4 << 10, 4 << 10)], reserve_ptable=True)
    dev.is_booting = False

    first = dev.mm.valloc(2 << 20, contiguous=True)
    dev.mm.valloc(2 << 20, contiguous=True, uncached=True)

    pt = dev.mm.root_page_table
    while not pt.is_page(idx:=(first.va_addr // dev.mm.pte_covers[pt.lv]) % dev.mm.pte_cnt[pt.lv]):
      self.assertTrue(pt.valid(idx))
      pt = dev.mm.pt_t(dev, pt.address(idx), lv=pt.lv + 1)
    self.assertEqual(pt.address(idx), first.paddrs[0][0])


if __name__ == "__main__": unittest.main()
