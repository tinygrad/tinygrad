import struct, types, unittest
from unittest.mock import MagicMock, call, patch

from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_GSP
from tinygrad.runtime.support.system import USBPCIDevice
from tinygrad.runtime.support.usb import CustomASM24Controller

class TestNVGSPSRAMBoot(unittest.TestCase):
  @staticmethod
  def booter_blob():
    blob = bytearray(0xc0)
    struct.pack_into("<6I", blob, 0x00, 0x10de, 1, len(blob), 0x18, 0x80, 0x40)
    struct.pack_into("<9I", blob, 0x18, 0x6c, 4, 0x64, 0x68, 0, 0, 0x70, 0x3c, 0x24)
    struct.pack_into("<5I", blob, 0x3c, 0, 0, 0x10, 0x20, 1)
    struct.pack_into("<4I", blob, 0x50, 0x30, 0x40, 0, 0)
    struct.pack_into("<II", blob, 0x64, 4, 0)
    blob[0x6c:0x70] = b"SIG!"
    struct.pack_into("<I", blob, 0x70, 1)
    blob[0x80:] = bytes(range(0x40))
    return bytes(blob)

  def test_prepares_signed_load_and_unload_booters(self):
    flcn = object.__new__(NV_FLCN)
    flcn.nvdev = types.SimpleNamespace(fw_name="ga102", pci_dev=types.SimpleNamespace(gsp_full_teardown=True), _alloc_boot_mem=MagicMock(
      side_effect=[(None, 0x100000, []), (None, 0x200000, []), (None, 0x300000, [])]))

    with patch("tinygrad.runtime.support.nv.ip.fetch_fw", side_effect=[self.booter_blob(), self.booter_blob()]) as fetch_fw:
      flcn.prep_booter()

    self.assertEqual([x.args[1] for x in fetch_fw.call_args_list], ["booter_load-570.144.bin", "booter_unload-570.144.bin"])
    self.assertEqual(flcn.booter_image_paddr, 0x100000)
    self.assertFalse(hasattr(flcn, "booter_unload_image_paddr"))
    self.assertEqual(flcn.nvdev._alloc_boot_mem.call_count, 1)
    self.assertEqual((flcn.booter_data_off, flcn.booter_data_sz, flcn.booter_code_off, flcn.booter_code_sz), (0x10, 0x20, 0x30, 0x40))
    self.assertEqual(bytes(flcn.nvdev._alloc_boot_mem.call_args.kwargs["data"])[4:8], b"SIG!")

    flcn._sb_image = b"SB"
    flcn.prep_fini()

    self.assertEqual((flcn.sb_image_paddr, flcn.booter_unload_image_paddr), (0x200000, 0x300000))
    self.assertEqual(bytes(flcn.nvdev._alloc_boot_mem.call_args.kwargs["data"])[4:8], b"SIG!")
    flcn.prep_fini()
    self.assertEqual(flcn.nvdev._alloc_boot_mem.call_count, 3)

  def test_native_pci_only_prepares_booter_load(self):
    flcn = object.__new__(NV_FLCN)
    flcn.nvdev = types.SimpleNamespace(fw_name="ga102", pci_dev=types.SimpleNamespace(gsp_full_teardown=False),
                                       _alloc_boot_mem=MagicMock(return_value=(None, 0x100000, [])))

    with patch("tinygrad.runtime.support.nv.ip.fetch_fw", return_value=self.booter_blob()) as fetch_fw: flcn.prep_booter()

    fetch_fw.assert_called_once()
    self.assertEqual(fetch_fw.call_args.args[1], "booter_load-570.144.bin")
    self.assertFalse(hasattr(flcn, "booter_unload_image_paddr"))

  def test_gfw_boot_waits_for_gsp_falcon_halt_before_scratch_completion(self):
    flcn = object.__new__(NV_FLCN)
    events, cpuctl = [], iter((0, NV_FLCN.FALCON_CPUCTL_HALTED))
    plm, progress = MagicMock(), MagicMock()
    plm.read_bitfields.side_effect = lambda: events.append("plm") or {'read_protection_level0': 1}
    progress.__getitem__.return_value.read.side_effect = lambda: events.append("progress") or 0xff
    flcn.nvdev = types.SimpleNamespace(
      rreg=lambda addr: events.append(("cpuctl", addr)) or next(cpuctl),
      NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK=plm,
      NV_PGC6_AON_SECURE_SCRATCH_GROUP_05=progress)

    flcn.wait_for_reset()

    self.assertEqual(events, [("cpuctl", NV_FLCN.GSP_FALCON_CPUCTL), ("cpuctl", NV_FLCN.GSP_FALCON_CPUCTL), "plm", "progress"])

  @staticmethod
  def init_wpr_flcn(frts_error=0, wpr2_hi=1, wpr2_lo=0x12345):
    flcn = object.__new__(NV_FLCN)
    flcn.falcon, flcn.frts_image_paddr, flcn.frts_offset = 0x110000, 0x100000, 0x12345000
    flcn.desc_v3 = types.SimpleNamespace(IMEMLoadSize=1, IMEMPhysBase=2, IMEMVirtBase=3, DMEMPhysBase=4, DMEMLoadSize=5,
                                         PKCDataOffset=6, EngineIdMask=7, UcodeId=8)
    scratch, hi, lo = MagicMock(), MagicMock(), MagicMock()
    scratch.__getitem__.return_value.read.return_value = frts_error << 16
    hi.read_bitfields.return_value, lo.read_bitfields.return_value = {'val': wpr2_hi}, {'val': wpr2_lo}
    flcn.nvdev = types.SimpleNamespace(pci_dev=types.SimpleNamespace(gsp_sram_boot=False), NV_PBUS_VBIOS_SCRATCH=scratch,
      NV_PFB_PRI_MMU_WPR2_ADDR_HI=hi, NV_PFB_PRI_MMU_WPR2_ADDR_LO=lo, NV_PFB_PRI_MMU_WPR2_ADDR_LO_ALIGNMENT=12)
    flcn.reset, flcn.execute_hs = MagicMock(), MagicMock()
    return flcn

  def test_init_wpr_validates_fwsec_result_and_wpr2_bounds(self):
    flcn = self.init_wpr_flcn()

    flcn.init_wpr()

    flcn.reset.assert_called_once_with(flcn.falcon)
    flcn.execute_hs.assert_called_once()

  def test_init_wpr_rejects_fwsec_error_and_invalid_wpr2_bounds(self):
    cases = (({"frts_error": 0xb}, "FWSEC FRTS failed with error 0xb"),
             ({"wpr2_hi": 0}, "without initializing WPR2"),
             ({"wpr2_lo": 0x12344}, "initialized WPR2 at 0x12344, expected 0x12345"))
    for kwargs, error in cases:
      with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error): self.init_wpr_flcn(**kwargs).init_wpr()

  def test_sram_boot_does_not_stage_duplicate_vram_sources(self):
    gsp = object.__new__(NV_GSP)
    vram_size = 24 << 30
    gsp.nvdev = types.SimpleNamespace(
      chip_name="GA102", fw_name="ga102", vram_size=vram_size, fmc_boot=False,
      flcn=types.SimpleNamespace(frts_offset=vram_size - (2 << 20)),
      pci_dev=types.SimpleNamespace(gsp_sram_boot=True), _alloc_boot_mem=MagicMock())
    sections = [types.SimpleNamespace(name=".fwimage", content=b"I" * 0x2000),
                types.SimpleNamespace(name=".fwsignature_ga10x", content=b"S" * 0x1000)]
    fw_header = types.SimpleNamespace(data_offset=4, data_size=4, header_offset=0)
    booter_desc = types.SimpleNamespace(monitorCodeOffset=1, monitorDataOffset=2, manifestOffset=3)

    with patch("tinygrad.runtime.support.nv.ip.fetch_fw", side_effect=[b"elf", bytes(4) + b"BOOT"]), \
         patch("tinygrad.runtime.support.nv.ip.elf_loader", return_value=(None, sections, None)), \
         patch("tinygrad.runtime.support.nv.ip.nv.struct_nvfw_bin_hdr") as header_t, \
         patch("tinygrad.runtime.support.nv.ip.nv.RM_RISCV_UCODE_DESC") as desc_t:
      header_t.from_buffer_copy.return_value, desc_t.from_buffer_copy.return_value = fw_header, booter_desc
      gsp.init_wpr_meta()

    gsp.nvdev._alloc_boot_mem.assert_not_called()
    self.assertEqual(gsp.wpr_meta_sysmem, 0x200000)
    meta = nv.GspFwWprMeta.from_buffer_copy(gsp._boot_sram)
    self.assertEqual((meta.sysmemAddrOfSignature, meta.sysmemAddrOfBootloader, meta.sysmemAddrOfRadix3Elf),
                     (0x201000, 0x202000, 0x208000))

  def test_sram_wpr_uses_cyclic_84_page_image_ring(self):
    gsp = object.__new__(NV_GSP)
    gsp.gsp_image = b''.join(bytes((page,)) * 0x1000 for page in range(86))
    gsp.gsp_signature, gsp.booter_image = b'S' * 0x1000, b'B' * 0x5800

    image = gsp._build_sram_wpr(nv.GspFwWprMeta())

    self.assertEqual(len(image), 0x80000)
    meta = nv.GspFwWprMeta.from_buffer_copy(image)
    self.assertEqual((meta.sysmemAddrOfSignature, meta.sysmemAddrOfBootloader, meta.sysmemAddrOfRadix3Elf),
                     (0x201000, 0x202000, 0x208000))
    self.assertEqual(image[0x1000:0x2000], gsp.gsp_signature)
    self.assertEqual(image[0x2000:0x7800], gsp.booter_image)
    image_ptes = struct.unpack_from('<86Q', image, 0xA000)
    self.assertEqual(image_ptes[:2], (0x22C000, 0x22D000))
    self.assertEqual(image_ptes[83:86], (0x27F000, 0x22C000, 0x22D000))
    self.assertEqual(image[0x2C000:0x2D000], bytes(0x1000))
    self.assertEqual(image[0x2D000:0x2E000], bytes((1,)) * 0x1000)

  def test_stream_schedule_rotates_across_all_ring_slots(self):
    ring_size = CustomASM24Controller.GSP_RING_PAGES * 0x1000
    batch_size = CustomASM24Controller.GSP_STREAM_BATCH_PAGES * 0x1000
    chunks = list(CustomASM24Controller.gsp_stream_chunks(bytes(ring_size + 4 * batch_size - 17)))

    self.assertEqual([slot for _, slot, _ in chunks], [11, 18, 25, 11])
    self.assertEqual([len(payload) for _, _, payload in chunks], [batch_size] * 4)
    self.assertAlmostEqual(chunks[0][0], 0.003)
    self.assertAlmostEqual(chunks[3][0], 0.0044)
    self.assertEqual(chunks[-1][2][-17:], bytes(17))

  def test_transport_restores_queues_and_argument_windows(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb, dev._wait_until = MagicMock(), MagicMock()
    dev.gsp_queues = types.SimpleNamespace(_root=types.SimpleNamespace(_mirror=bytearray(b'queue')))
    dev._gsp_args = {0x100:b'rm', 0x200:b'libos'}

    dev.stream_gsp_boot(b'image', 4.0)

    dev.usb.stream_gsp_image.assert_called_once_with(b'image', 4.0)
    self.assertEqual(dev._wait_until.call_args_list[0].args, (4.270,))
    self.assertEqual(dev._wait_until.call_args_list[1].args, (4.380,))
    self.assertEqual(dev.usb.scsi_write.call_args_list[0].args, (b'queue',))
    self.assertEqual(dev.usb.scsi_write.call_args_list[1].args, (b'queue',))
    self.assertEqual(dev.usb.write.call_args_list[0].args, (0xB900, b'rm'))
    self.assertEqual(dev.usb.write.call_args_list[1].args, (0xBA00, b'libos'))

  @patch('tinygrad.runtime.support.nv.ip.time.perf_counter', return_value=12.5)
  def test_sec2_launch_sets_local_contexts_then_streams(self, _):
    flcn = object.__new__(NV_FLCN)
    contexts = [MagicMock() for _ in range(8)]
    context_regs = MagicMock()
    context_regs.__getitem__.side_effect = contexts.__getitem__
    context_reg = MagicMock()
    context_reg.with_base.return_value = context_regs
    cpuctl = MagicMock()
    cpuctl.with_base.return_value.read_bitfields.return_value = {'alias_en': 0}
    stream, invalidate = MagicMock(), MagicMock()
    flcn.nvdev = types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(gsp_sram_boot=True, stream_gsp_boot=stream),
      gsp=types.SimpleNamespace(gsp_image=b'image', invalidate_rpc_memory=invalidate),
      NV_PFALCON_FBIF_TRANSCFG=context_reg, NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL=7,
      NV_PFALCON_FALCON_CPUCTL=cpuctl, NV_PFALCON_FALCON_CPUCTL_ALIAS=0x130)
    flcn.sec2 = 0x840000

    flcn.start_cpu(flcn.sec2, stream_gsp=True)

    for context in contexts: context.update.assert_called_once_with(target=0, mem_type=7)
    cpuctl.with_base.return_value.write.assert_called_once_with(startcpu=1)
    stream.assert_called_once_with(b'image', 12.5)
    invalidate.assert_called_once_with()

  def test_sec2_unload_does_not_restream_gsp(self):
    flcn = object.__new__(NV_FLCN)
    cpuctl, stream = MagicMock(), MagicMock()
    cpuctl.with_base.return_value.read_bitfields.return_value = {'alias_en': 0}
    flcn.nvdev = types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(gsp_sram_boot=True, stream_gsp_boot=stream),
      gsp=types.SimpleNamespace(gsp_image=b'image'), NV_PFALCON_FALCON_CPUCTL=cpuctl, NV_PFALCON_FALCON_CPUCTL_ALIAS=0x130)
    flcn.sec2 = 0x840000

    flcn.start_cpu(flcn.sec2)

    cpuctl.with_base.return_value.write.assert_called_once_with(startcpu=1)
    stream.assert_not_called()

  def test_falcon_reset_holds_gsp_and_sec2_in_reset_before_release(self):
    for base, engine_name in ((0x110000, "gsp"), (0x840000, "sec2")):
      with self.subTest(engine=engine_name):
        events = []
        flcn = object.__new__(NV_FLCN)
        flcn.falcon, flcn.sec2 = 0x110000, 0x840000
        hwcfg2_reg = MagicMock()
        gsp_engine, sec2_engine = MagicMock(), MagicMock()
        gsp_engine.write.side_effect = lambda **kwargs: events.append(("gsp", kwargs['reset']))
        sec2_engine.write.side_effect = lambda **kwargs: events.append(("sec2", kwargs['reset']))
        bcr = MagicMock()
        bcr_reg = MagicMock()
        bcr_reg.with_base.return_value = bcr
        flcn.nvdev = types.SimpleNamespace(
          NV_PFALCON_FALCON_HWCFG2=hwcfg2_reg, NV_PGSP_FALCON_ENGINE=gsp_engine, NV_PSEC_FALCON_ENGINE=sec2_engine,
          NV_PRISCV_RISCV_BCR_CTRL=bcr_reg)

        with patch("tinygrad.runtime.support.nv.ip.time.sleep", side_effect=lambda delay: events.append(("sleep", delay))), \
             patch("tinygrad.runtime.support.nv.ip.wait_cond", side_effect=lambda *args, **kwargs: events.append("scrub")):
          flcn.reset(base, riscv=True)

        self.assertEqual(events, [(engine_name, 1), ("sleep", 0.1), (engine_name, 0), "scrub"])
        bcr_reg.with_base.assert_called_once_with(base)
        bcr.write.assert_called_once_with(core_select=1, valid=0, brfetch=1)

  def test_falcon_reset_skips_core_switch_for_non_riscv_falcon(self):
    flcn = object.__new__(NV_FLCN)
    flcn.falcon, flcn.sec2 = 0x110000, 0x840000
    hwcfg2 = MagicMock()
    hwcfg2.read_bitfields.return_value = {'riscv': 0}
    hwcfg2_reg = MagicMock()
    hwcfg2_reg.with_base.return_value = hwcfg2
    rm = MagicMock()
    rm_reg = MagicMock()
    rm_reg.with_base.return_value = rm
    flcn.nvdev = types.SimpleNamespace(
      chip_id=0x172000a1, NV_PFALCON_FALCON_HWCFG2=hwcfg2_reg, NV_PGSP_FALCON_ENGINE=MagicMock(),
      NV_PSEC_FALCON_ENGINE=MagicMock(), NV_PRISCV_RISCV_BCR_CTRL=MagicMock(), NV_PFALCON_FALCON_RM=rm_reg)

    with patch("tinygrad.runtime.support.nv.ip.time.sleep"), \
         patch("tinygrad.runtime.support.nv.ip.wait_cond"):
      flcn.reset(flcn.sec2)

    rm_reg.with_base.assert_not_called()
    rm.write.assert_not_called()

  def test_rearms_sec2_for_cpu_sequencer_reuse(self):
    flcn = object.__new__(NV_FLCN)
    context = MagicMock()
    contexts = MagicMock()
    contexts.__getitem__.return_value = context
    context_reg = MagicMock()
    context_reg.with_base.return_value = contexts
    flcn.reset, flcn.disable_ctx_req = MagicMock(), MagicMock()
    flcn.sec2 = 0x840000
    flcn.nvdev = types.SimpleNamespace(
      pci_dev=types.SimpleNamespace(gsp_sram_boot=True), NV_PFALCON_FBIF_TRANSCFG=context_reg,
      NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM=5, NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL=7)

    flcn.rearm_sec2_queue()

    flcn.reset.assert_called_once_with(0x840000)
    flcn.disable_ctx_req.assert_called_once_with(0x840000)
    context.update.assert_called_once_with(target=5, mem_type=7)

  def test_gsp_fini_requests_unload_and_waits_for_suspend(self):
    gsp = object.__new__(NV_GSP)
    gsp.rpc_unloading_guest_driver = MagicMock()
    mailbox = MagicMock()
    mailbox.with_base.return_value.read.side_effect = [0, 1 << 31]
    gsp.nvdev = types.SimpleNamespace(pci_dev=types.SimpleNamespace(gsp_full_teardown=True), flcn=types.SimpleNamespace(falcon=0x110000),
                                      NV_PFALCON_FALCON_MAILBOX0=mailbox)

    gsp.fini_hw()

    gsp.rpc_unloading_guest_driver.assert_called_once_with()
    self.assertEqual(mailbox.with_base.call_args_list, [call(0x110000), call(0x110000)])

  def test_native_gsp_fini_keeps_existing_fast_unload_behavior(self):
    gsp = object.__new__(NV_GSP)
    gsp.rpc_unloading_guest_driver = MagicMock()
    mailbox = MagicMock()
    gsp.nvdev = types.SimpleNamespace(pci_dev=types.SimpleNamespace(gsp_full_teardown=False),
                                      flcn=types.SimpleNamespace(falcon=0x110000), NV_PFALCON_FALCON_MAILBOX0=mailbox)

    gsp.fini_hw()

    gsp.rpc_unloading_guest_driver.assert_called_once_with()
    mailbox.with_base.assert_not_called()

  def test_flcn_shutdown_executes_fwsec_despite_complete_status_registers(self):
    flcn = object.__new__(NV_FLCN)
    flcn.falcon, flcn.sec2 = 0x110000, 0x840000
    flcn.sb_image_paddr, flcn.booter_unload_image_paddr = 0x100000, 0x200000
    flcn.booter_unload_code_off, flcn.booter_unload_code_sz = 0x30, 0x40
    flcn.booter_unload_data_off, flcn.booter_unload_data_sz = 0x10, 0x20
    flcn.desc_v3 = types.SimpleNamespace(IMEMLoadSize=0x300, IMEMPhysBase=0x100, IMEMVirtBase=0x200,
                                         DMEMPhysBase=0x400, DMEMLoadSize=0x500, PKCDataOffset=0x600, EngineIdMask=7, UcodeId=8)
    flcn.sb_code_off, flcn.sb_data_off = 0, 0x300
    flcn.sb_imem_pa, flcn.sb_imem_va, flcn.sb_imem_sz = 0x100, 0x200, 0x300
    flcn.sb_dmem_pa, flcn.sb_dmem_va, flcn.sb_dmem_sz = 0x400, 0, 0x500
    flcn.sb_pkc_off, flcn.sb_engid, flcn.sb_ucodeid = 0x600, 7, 8
    wpr, scratch, plm, progress = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    wpr.read.side_effect = [1, 1, 0]
    scratch.__getitem__.return_value.read.return_value = 0
    plm.read_bitfields.return_value = {'read_protection_level0': 1}
    progress.__getitem__.return_value.read_bitfields.return_value = {'0_gfw_boot_progress': 0xff}
    pci_dev = types.SimpleNamespace(gsp_full_teardown=True, reset=MagicMock())
    flcn.nvdev = types.SimpleNamespace(pci_dev=pci_dev,
      NV_PFB_PRI_MMU_WPR2_ADDR_HI=wpr, NV_PBUS_VBIOS_SCRATCH=scratch,
      NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK=plm, NV_PGC6_AON_SECURE_SCRATCH_GROUP_05=progress,
      NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT_PROGRESS_COMPLETED=0xff)
    flcn.reset, flcn.execute_hs = MagicMock(), MagicMock(side_effect=[None, (0, 0xff)])

    flcn.shutdown_fwsec()
    flcn.shutdown_booter()

    self.assertEqual(flcn.reset.call_args_list, [call(flcn.falcon), call(flcn.sec2)])
    self.assertEqual(flcn.execute_hs.call_args_list[0].args[:2], (flcn.falcon, flcn.sb_image_paddr))
    self.assertEqual(flcn.execute_hs.call_args_list[1].args[:2], (flcn.sec2, flcn.booter_unload_image_paddr))
    self.assertEqual(flcn.execute_hs.call_args_list[1].kwargs["mailbox"], (0xff << 32) | 0xff)
    pci_dev.reset.assert_not_called()

  def test_flcn_fini_requires_wpr2_to_be_cleared(self):
    flcn = object.__new__(NV_FLCN)
    flcn.falcon, flcn.sec2 = 0x110000, 0x840000
    flcn.sb_image_paddr = flcn.booter_unload_image_paddr = 0
    flcn.booter_unload_code_off = flcn.booter_unload_code_sz = 0
    flcn.booter_unload_data_off = flcn.booter_unload_data_sz = 0
    flcn.desc_v3 = types.SimpleNamespace(IMEMLoadSize=0, IMEMPhysBase=0, IMEMVirtBase=0, DMEMPhysBase=0, DMEMLoadSize=0,
                                         PKCDataOffset=0, EngineIdMask=0, UcodeId=0)
    wpr = MagicMock()
    wpr.read.side_effect = [1, 1]
    flcn.nvdev = types.SimpleNamespace(NV_PFB_PRI_MMU_WPR2_ADDR_HI=wpr)
    flcn.reset, flcn.execute_hs = MagicMock(), MagicMock(return_value=(0, 0))

    with self.assertRaisesRegex(RuntimeError, "WPR2 is still active"): flcn.shutdown_booter()

  def test_flcn_shutdown_refuses_missing_firmware_before_reset(self):
    flcn = object.__new__(NV_FLCN)
    flcn.falcon, flcn.sec2 = 0x110000, 0x840000
    wpr = MagicMock()
    wpr.read.return_value = 1
    flcn.nvdev = types.SimpleNamespace(NV_PFB_PRI_MMU_WPR2_ADDR_HI=wpr)
    flcn.reset = MagicMock()

    with self.assertRaisesRegex(RuntimeError, "FWSEC shutdown metadata is unavailable"): flcn.shutdown_fwsec()
    with self.assertRaisesRegex(RuntimeError, "Booter Unload metadata is unavailable"): flcn.shutdown_booter()

    flcn.reset.assert_not_called()

  def test_rm_free_sends_root_object_and_checks_status(self):
    gsp = object.__new__(NV_GSP)
    gsp.priv_root, gsp.cmd_q, gsp.stat_q = 0xc1e00004, MagicMock(), MagicMock()
    response = nv.rpc_free_v(params=nv.NVOS00_PARAMETERS_v03_00(status=0))
    gsp.stat_q.wait_resp.return_value = bytes(response)

    gsp.rpc_rm_free(0xc1000000, client=0xc1000000)

    func, payload = gsp.cmd_q.send_rpc.call_args.args
    request = nv.rpc_free_v.from_buffer_copy(payload)
    self.assertEqual(func, nv.NV_VGPU_MSG_FUNCTION_FREE)
    self.assertEqual((request.params.hRoot, request.params.hObjectParent, request.params.hObjectOld), (0xc1000000, 0, 0xc1000000))
    gsp.stat_q.wait_resp.assert_called_once_with(nv.NV_VGPU_MSG_FUNCTION_FREE)

  def test_rm_free_preserves_rm_status(self):
    gsp = object.__new__(NV_GSP)
    gsp.priv_root, gsp.cmd_q, gsp.stat_q = 0xc1e00004, MagicMock(), MagicMock()
    gsp.stat_q.wait_resp.return_value = bytes(nv.rpc_free_v(params=nv.NVOS00_PARAMETERS_v03_00(status=0x31)))

    with self.assertRaisesRegex(RuntimeError, "0x31"): gsp.rpc_rm_free(0xc1000000, client=0xc1000000)


if __name__ == '__main__': unittest.main()
