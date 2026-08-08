import contextlib, unittest
from unittest.mock import MagicMock, patch

from tinygrad.device import BufferSpec
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.ops_nv import NVAllocator, USBIface
from tinygrad.runtime.support.system import System, PCIIfaceBase, USBPCIDevice
from tinygrad.runtime.support.usb import ASM24GSPQueueInterface, CustomASM24Controller, USBMMIOInterface


class TestCustomASM24Controller(unittest.TestCase):
  def test_checks_pcie_link_after_power_on(self):
    usb = MagicMock()
    with patch.object(CustomASM24Controller, "read", MagicMock(side_effect=[b"\x59", b"\x78"])) as read, \
         patch.object(CustomASM24Controller, "set_pcie_power") as set_pcie_power:
      CustomASM24Controller(usb)

    set_pcie_power.assert_called_once_with(True)
    self.assertEqual(read.call_count, 2)

  def test_memory_tlp_format_tracks_address_width(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    with patch.object(controller, "pcie_request", return_value=0):
      controller.pcie_mem_read(0x10000000, 4)
      controller.pcie_request.assert_called_once_with(0x00, 0x10000000)
      controller.pcie_request.reset_mock()
      controller.pcie_mem_read(0x800000000, 4)
      controller.pcie_request.assert_called_once_with(0x20, 0x800000000)

    controller.usb.bulk_read.return_value = memoryview(bytes(8))
    controller.pcie_mem_read(0x10000000, 8)
    self.assertEqual(controller.usb.control_write.call_args.args[:3], (0xF0, 0x0F00, 2))
    controller.pcie_mem_write(0x10000000, bytes(4))
    self.assertEqual(controller.usb.control_write.call_args.args[:3], (0xF0, 0x0F40, 1))
    controller.usb.bulk_write.assert_called_once_with(bytes(4))
    controller.usb.bulk_write.reset_mock()
    controller.pcie_mem_write(0x800000000, bytes(8))
    self.assertEqual(controller.usb.control_write.call_args.args[:3], (0xF0, 0x0F60, 1))
    controller.usb.bulk_write.assert_called_once_with(bytes(8))

  def test_mmio_slice_preserves_element_format(self):
    controller = MagicMock()
    controller.pcie_mem_read.return_value = memoryview(bytes.fromhex("0100000002000000"))

    mmio = USBMMIOInterface(controller, 0x10000000, 8, fmt='I')
    self.assertEqual(mmio[:2], [1, 2])

  def test_unaligned_mmio_uses_aligned_read_modify_write(self):
    controller = MagicMock()
    controller.pcie_mem_read.return_value = memoryview(bytes.fromhex("0011223344556677"))
    mmio = USBMMIOInterface(controller, 0x10000001, 5, fmt='B')

    self.assertEqual(bytes(mmio[:5]), bytes.fromhex("1122334455"))
    controller.pcie_mem_read.assert_called_with(0x10000000, 8)

    mmio[1:4] = b"abc"
    controller.pcie_mem_write.assert_called_once_with(0x10000000, bytes.fromhex("0011616263556677"))

  def test_mmio_write_cannot_cross_the_mapped_window(self):
    controller = MagicMock()
    mmio = USBMMIOInterface(controller, 0x800000000, 0x100, fmt='B')

    with self.assertRaisesRegex(AssertionError, "USB MMIO write size mismatch"):
      mmio[0x100:0x104] = bytes(4)

    controller.pcie_mem_write.assert_not_called()

  def test_sram_mmio_write_uses_configured_start_slot(self):
    controller = MagicMock()
    mmio = USBMMIOInterface(controller, 0xF000, 0x74000, fmt='B', pcimem=False, sram_start_slot=1)

    mmio.view(size=7)[:] = b"payload"

    controller.scsi_write.assert_called_once_with(b"payload", start_slot=1)
    controller.pcie_mem_write.assert_not_called()

  def test_scsi_write_arms_each_bulk_transfer(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    payload = bytes(0x54000)

    controller.scsi_write(payload, start_slot=11)
    controller.scsi_write(payload, start_slot=11)

    self.assertEqual(controller.usb.control_write.call_args_list,
                     [unittest.mock.call(0xF2, value=0x2A0, index=0x150B)] * 2)
    self.assertEqual(controller.usb.bulk_write.call_args_list, [unittest.mock.call(payload)] * 2)

  def test_sram_stream_arms_once_and_bulk_writes_each_scheduled_chunk(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    ring_size = controller.GSP_RING_PAGES * 0x1000
    batch_size = controller.GSP_STREAM_BATCH_PAGES * 0x1000
    image = bytes(ring_size + 4 * batch_size - 17)

    with patch("tinygrad.runtime.support.usb.time.perf_counter", return_value=1.0):
      controller.stream_gsp_image(image, 0.0)

    controller.usb.control_write.assert_called_once_with(0xF5, value=4, index=0x070B)
    self.assertEqual(controller.usb.bulk_write.call_count, 4)
    self.assertEqual([len(call.args[0]) for call in controller.usb.bulk_write.call_args_list], [batch_size] * 4)

  def test_sram_read_uses_f6_for_each_overlapping_sector(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    controller.usb.bulk_read.return_value = memoryview(bytes([3]) * 512 + bytes([4]) * 512)

    self.assertEqual(controller.sram_read(3 * 0x4000 + 0x610, 0x250), bytes([3]) * 0x1F0 + bytes([4]) * 0x60)
    controller.usb.control_write.assert_called_once_with(0xF6, value=3, index=0x0203)
    controller.usb.bulk_read.assert_called_once_with(1024)

    controller.usb.control_write.reset_mock()
    controller.usb.bulk_read.side_effect = None
    controller.usb.bulk_read.return_value = memoryview(bytes(0x1000))
    controller.scsi_read(0x1000)
    controller.usb.control_write.assert_not_called()

  def test_sram_read_rejects_a_short_sector(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    controller.usb.bulk_read.return_value = memoryview(bytes(511))

    with self.assertRaisesRegex(RuntimeError, "short read"):
      controller.sram_read(0, 1)

  def test_large_pcie_transfers_are_chunked(self):
    controller = object.__new__(CustomASM24Controller)
    controller.usb = MagicMock()
    size = controller.PCIE_BULK_CHUNK_SIZE + 8

    with patch.object(controller, "_f0_out") as f0_out:
      controller.pcie_mem_write(0x800000000, bytes(size))
      self.assertEqual([(c.args[0], c.args[2], c.args[3]) for c in f0_out.call_args_list],
                       [(0x60, 0x800000000, 0x40000), (0x60, 0x800100000, 2)])
      self.assertEqual([len(c.args[0]) for c in controller.usb.bulk_write.call_args_list], [1 << 20, 8])

      f0_out.reset_mock()
      controller.usb.bulk_read.side_effect = lambda nbytes, timeout: memoryview(bytes(nbytes))
      self.assertEqual(len(controller.pcie_mem_read(0x10000000, size)), size)
      self.assertEqual([(c.args[0], c.args[2], c.args[3]) for c in f0_out.call_args_list],
                       [(0x00, 0x10000000, 0x40000), (0x00, 0x10100000, 2)])


class FakePCIeController:
  def __init__(self): self.writes = []

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    if value is not None:
      self.writes.append((byte_addr, bus, value, size))
      return None
    if byte_addr == pci.PCI_VENDOR_ID and size == 4:
      return {0: 0x24631B21, 1: 0x1B631B21, 2: 0x220410DE}[bus]
    if byte_addr == pci.PCI_HEADER_TYPE and size == 1:
      return pci.PCI_HEADER_TYPE_BRIDGE if bus < 2 else pci.PCI_HEADER_TYPE_NORMAL
    raise AssertionError(f"unexpected config read: {byte_addr=:#x}, {bus=}, {size=}")


class TestUSBPCIeDiscovery(unittest.TestCase):
  def test_nvidia_discovers_legacy_and_current_firmware_vids(self):
    legacy, current = (object(), "usb:legacy"), (object(), "usb:current")
    with patch("tinygrad.runtime.ops_nv.USB3.list_devices", side_effect=[[legacy], [current]]) as list_devices, \
         patch("tinygrad.runtime.ops_nv.hcq_filter_visible_devices", side_effect=lambda devices, _: devices), \
         patch("tinygrad.runtime.ops_nv.USBPCIDevice") as pci_device, patch("tinygrad.runtime.ops_nv.NVDev"), \
         patch.object(USBIface, "_init_nvd"):
      iface = USBIface(MagicMock(), 1)

    self.assertEqual(list_devices.call_args_list, [unittest.mock.call(0xADD1, 0x0001), unittest.mock.call(0x3801, 0x0001)])
    pci_device.assert_called_once_with("NV", *current)
    self.assertEqual(iface.count, 2)

  @patch.object(USBPCIDevice, "_setup_pcie")
  @patch.object(USBPCIDevice, "supports_flr", return_value=True)
  @patch("tinygrad.runtime.support.system.CustomASM24Controller")
  @patch("tinygrad.runtime.support.system.USB3")
  @patch.object(System, "flock_acquire", return_value=1)
  def test_nvidia_device_allows_full_fixed_window_boot_drain(self, flock_acquire, usb3, controller, supports_flr, setup_pcie):
    dev = USBPCIDevice("NV", MagicMock(), "custom v0.1")

    self.assertEqual(dev.gsp_rpc_timeout_ms, 120000)
    self.assertTrue(dev.gsp_full_teardown)
    self.assertTrue(dev.gsp_flr_recovery)
    self.assertFalse(hasattr(dev, "reset_after_gsp_teardown"))
    controller.assert_called_once_with(usb3.return_value)
    supports_flr.assert_called_once_with()

  def test_stops_before_writing_endpoint_bus_registers(self):
    controller = FakePCIeController()

    self.assertEqual(System.pci_find_usb_endpoint(controller), 2)
    self.assertEqual(controller.writes, [
      (pci.PCI_PRIMARY_BUS, 0, 0x00FF0100, 4),
      (pci.PCI_PRIMARY_BUS, 1, 0x00FF0201, 4),
    ])

  def test_bar_setup_programs_each_bridge_primary_bus(self):
    controller = MagicMock()
    controller.pcie_cfg_req.side_effect = lambda byte_addr, **kw: \
      (0 if byte_addr == 0x100 else pci.PCI_BASE_ADDRESS_SPACE_IO) if "value" not in kw else None

    System.pci_setup_usb_bars(controller, gpu_bus=2, mem_base=0x10000000, pref_mem_base=0x800000000)

    bus_writes = [c for c in controller.pcie_cfg_req.call_args_list if c.args[0] == pci.PCI_PRIMARY_BUS and "value" in c.kwargs]
    self.assertEqual([(c.kwargs["bus"], c.kwargs["value"]) for c in bus_writes], [(0, 0x00020100), (1, 0x00020201)])
    endpoint_commands = [c.kwargs["value"] for c in controller.pcie_cfg_req.call_args_list
                         if c.args[0] == pci.PCI_COMMAND and c.kwargs.get("bus") == 2 and "value" in c.kwargs]
    self.assertEqual(endpoint_commands, [0, pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER])

  def test_stages_nvidia_gsp_arguments_in_fixed_xdata_window(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb = MagicMock()

    self.assertEqual(dev.stage_gsp_args(b"RM", 0x100), 0x828100)
    self.assertEqual(dev.stage_gsp_args(b"OS", 0x200), 0x828200)

    self.assertEqual(dev.usb.write.call_args_list[0].args, (0xB900, b"RM" + bytes(0xFE)))
    self.assertEqual(dev.usb.write.call_args_list[1].args, (0xBA00, b"OS" + bytes(0xFE)))

  def test_map_bar_rejects_an_adjacent_bar_address(self):
    dev = object.__new__(USBPCIDevice)
    dev.usb, dev._bar_info = MagicMock(), {1: (0x800000000, 0x10000000)}

    with self.assertRaisesRegex(ValueError, "exceeds its 0x10000000-byte aperture"):
      dev.map_bar(1, off=0x10000000, size=4)

  @patch("tinygrad.runtime.support.system.time.sleep")
  def test_stages_gsp_boot_after_configuring_known_good_link_settings(self, sleep):
    dev = object.__new__(USBPCIDevice)
    dev.gpu_bus, dev.usb = 2, MagicMock()
    dev.usb.pcie_cfg_req.return_value = 0

    dev.stage_gsp_boot(b"boot")

    sleep.assert_called_once_with(0.1)
    writes = [(call.args[0], call.kwargs.get("bus"), call.kwargs.get("value"))
              for call in dev.usb.pcie_cfg_req.call_args_list if "value" in call.kwargs]
    self.assertIn((0x80 + 0x30, 1, 1), writes)
    self.assertIn((0x78 + 0x30, 2, 1), writes)
    self.assertIn((0x78 + 0x08, 2, 0), writes)
    self.assertFalse(any(offset in (0x104, 0x110) for offset, _, _ in writes))
    dev.usb.scsi_write.assert_called_once_with(b"boot")

  @patch("tinygrad.runtime.support.system.time.sleep")
  def test_reset_uses_upstream_bridge_without_power_cycling_usb(self, sleep):
    dev = object.__new__(USBPCIDevice)
    dev.gpu_bus, dev.usb = 2, MagicMock()
    dev.read_config, dev.write_config_flush, dev._setup_pcie = MagicMock(return_value=0x7), MagicMock(), MagicMock()
    dev.usb.pcie_cfg_req.return_value = pci.PCI_BRIDGE_CTL_VGA

    dev.reset()

    dev.write_config_flush.assert_called_once_with(pci.PCI_COMMAND, 0x3, 2)
    self.assertEqual(dev.usb.pcie_cfg_req.call_args_list, [
      unittest.mock.call(pci.PCI_BRIDGE_CONTROL, bus=1, size=2),
      unittest.mock.call(pci.PCI_BRIDGE_CONTROL, bus=1, value=pci.PCI_BRIDGE_CTL_VGA | pci.PCI_BRIDGE_CTL_BUS_RESET, size=2),
      unittest.mock.call(pci.PCI_BRIDGE_CONTROL, bus=1, value=pci.PCI_BRIDGE_CTL_VGA, size=2),
    ])
    self.assertEqual(sleep.call_args_list, [unittest.mock.call(0.1), unittest.mock.call(1.0)])
    dev._setup_pcie.assert_called_once_with()

  def test_discovers_flr_through_standard_capability_list(self):
    dev = object.__new__(USBPCIDevice)
    config = {
      (pci.PCI_STATUS, 2): pci.PCI_STATUS_CAP_LIST,
      (pci.PCI_CAPABILITY_LIST, 1): 0x60,
      (0x60, 2): (0x78 << 8) | 0x01,
      (0x78, 2): pci.PCI_CAP_ID_EXP,
      (0x78 + pci.PCI_EXP_DEVCAP, 4): pci.PCI_EXP_DEVCAP_FLR,
    }
    dev.read_config = MagicMock(side_effect=lambda offset, size: config[(offset, size)])

    self.assertEqual(dev._pci_capability(pci.PCI_CAP_ID_EXP), 0x78)
    self.assertTrue(dev.supports_flr())

  def test_rejects_cyclic_pci_capability_list(self):
    dev = object.__new__(USBPCIDevice)
    config = {
      (pci.PCI_STATUS, 2): pci.PCI_STATUS_CAP_LIST,
      (pci.PCI_CAPABILITY_LIST, 1): 0x60,
      (0x60, 2): (0x60 << 8) | 0x01,
    }
    dev.read_config = MagicMock(side_effect=lambda offset, size: config[(offset, size)])

    with self.assertRaisesRegex(RuntimeError, "Malformed PCI capability list at 0x60"):
      dev._pci_capability(pci.PCI_CAP_ID_EXP)

  def test_flr_requires_endpoint_support(self):
    dev = object.__new__(USBPCIDevice)
    dev._pci_capability, dev.read_config = MagicMock(return_value=None), MagicMock()
    dev.write_config, dev.write_config_flush = MagicMock(), MagicMock()

    with self.assertRaisesRegex(RuntimeError, "does not support Function Level Reset"):
      dev.function_level_reset()

    dev.read_config.assert_not_called()
    dev.write_config.assert_not_called()
    dev.write_config_flush.assert_not_called()

  @patch("tinygrad.runtime.support.system.time.sleep")
  def test_flr_quiesces_endpoint_and_rebuilds_pcie_state(self, sleep):
    dev = object.__new__(USBPCIDevice)
    cap, identity, events = 0x78, 0x220410DE, []
    identities, transaction_status = iter((identity, 0xffffffff, identity, identity)), iter((pci.PCI_EXP_DEVSTA_TRPND, 0))

    def read_config(offset, size):
      events.append(("read", offset, size))
      if offset == cap + pci.PCI_EXP_DEVCAP: return pci.PCI_EXP_DEVCAP_FLR
      if offset == pci.PCI_VENDOR_ID: return next(identities)
      if offset == pci.PCI_COMMAND: return pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER
      if offset == cap + pci.PCI_EXP_DEVSTA: return next(transaction_status)
      if offset == cap + pci.PCI_EXP_DEVCTL: return 0x123
      raise AssertionError(f"unexpected config read at {offset:#x} size {size}")

    dev._pci_capability = MagicMock(return_value=cap)
    dev.read_config = MagicMock(side_effect=read_config)
    dev.write_config_flush = MagicMock(side_effect=lambda *args: events.append(("flush", *args)))
    dev.write_config = MagicMock(side_effect=lambda *args: events.append(("write", *args)))
    dev._setup_pcie = MagicMock(side_effect=lambda: events.append(("setup",)))

    dev.function_level_reset()

    dev.write_config_flush.assert_called_once_with(
      pci.PCI_COMMAND, pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY, 2)
    dev.write_config.assert_called_once_with(
      cap + pci.PCI_EXP_DEVCTL, 0x123 | pci.PCI_EXP_DEVCTL_BCR_FLR, 2)
    self.assertEqual(sleep.call_args_list, [unittest.mock.call(0.001), unittest.mock.call(0.1), unittest.mock.call(0.01)])
    dev._setup_pcie.assert_called_once_with()
    self.assertLess(events.index(("write", cap + pci.PCI_EXP_DEVCTL, 0x123 | pci.PCI_EXP_DEVCTL_BCR_FLR, 2)),
                    events.index(("setup",)))

  @patch("tinygrad.runtime.support.system.time.monotonic", side_effect=(0.0, 1.0))
  def test_flr_does_not_trigger_with_pending_transactions(self, monotonic):
    dev = object.__new__(USBPCIDevice)
    cap, identity = 0x78, 0x220410DE
    config = {
      (cap + pci.PCI_EXP_DEVCAP, 4): pci.PCI_EXP_DEVCAP_FLR,
      (pci.PCI_VENDOR_ID, 4): identity,
      (pci.PCI_COMMAND, 2): pci.PCI_COMMAND_MASTER,
      (cap + pci.PCI_EXP_DEVSTA, 2): pci.PCI_EXP_DEVSTA_TRPND,
    }
    dev._pci_capability = MagicMock(return_value=cap)
    dev.read_config = MagicMock(side_effect=lambda offset, size: config[(offset, size)])
    dev.write_config, dev.write_config_flush, dev._setup_pcie = MagicMock(), MagicMock(), MagicMock()

    with self.assertRaisesRegex(TimeoutError, "waiting for PCIe transactions before FLR"):
      dev.function_level_reset(timeout=0.5)

    dev.write_config_flush.assert_called_once_with(pci.PCI_COMMAND, 0, 2)
    dev.write_config.assert_not_called()
    dev._setup_pcie.assert_not_called()


class TestUSBIfaceAllocation(unittest.TestCase):
  def test_host_staging_buffer_gets_bar_view_over_contiguous_vram(self):
    iface = object.__new__(USBIface)
    iface.pci_dev, iface.vram_bar = MagicMock(), 1
    ret = MagicMock()
    ret.meta.mapping.paddrs, ret.meta.mapping.size, ret.meta.has_cpu_mapping = [(0x123000, 0x200000)], 0x200000, False
    bar_view = iface.pci_dev.map_bar.return_value

    with patch.object(PCIIfaceBase, "alloc", autospec=True, return_value=ret) as alloc:
      self.assertIs(iface.alloc(0x200000, host=True), ret)

    alloc.assert_called_once_with(iface, 0x200000, host=False, uncached=False, cpu_access=False,
                                  contiguous=True, force_devmem=True, cpu_visible=True)
    iface.pci_dev.map_bar.assert_called_once_with(1, off=0x123000, size=0x200000)
    self.assertIs(ret.view, bar_view)
    self.assertFalse(ret.meta.has_cpu_mapping)

  def test_normal_gpu_buffer_does_not_consume_cpu_visible_vram(self):
    iface = object.__new__(USBIface)
    ret = MagicMock()
    ret.meta.has_cpu_mapping = False

    with patch.object(PCIIfaceBase, "alloc", autospec=True, return_value=ret) as alloc:
      self.assertIs(iface.alloc(0x200000), ret)

    alloc.assert_called_once_with(iface, 0x200000, host=False, uncached=False, cpu_access=False,
                                  contiguous=False, force_devmem=True, cpu_visible=False)


class TestNVAllocatorAllocation(unittest.TestCase):
  def setUp(self):
    self.allocator = object.__new__(NVAllocator)
    self.allocator.dev = MagicMock()
    self.allocator.dev.iface = object.__new__(USBIface)
    self.allocator.dev.iface.alloc = MagicMock()

  def test_usb_host_buffer_retains_default_clear(self):
    self.allocator._alloc(0x200000, BufferSpec(host=True))
    self.allocator.dev.iface.alloc.assert_called_once_with(0x200000, cpu_access=False, host=True)

  def test_usb_cpu_access_buffer_retains_default_clear(self):
    self.allocator._alloc(0x1000, BufferSpec(cpu_access=True))
    self.allocator.dev.iface.alloc.assert_called_once_with(0x1000, cpu_access=True, host=False)

  def test_usb_host_cpu_access_buffer_retains_default_clear(self):
    self.allocator._alloc(0x1000, BufferSpec(host=True, cpu_access=True))
    self.allocator.dev.iface.alloc.assert_called_once_with(0x1000, cpu_access=True, host=True)

  def test_non_usb_host_buffer_retains_default_clear(self):
    self.allocator.dev.iface = MagicMock()
    self.allocator._alloc(0x200000, BufferSpec(host=True))
    self.allocator.dev.iface.alloc.assert_called_once_with(0x200000, cpu_access=False, host=True)

  def test_usb_copyout_chunks_through_sram_and_restores_completion(self):
    arena_size, prefix_size = ASM24GSPQueueInterface.TRANSFER_SIZE, ASM24GSPQueueInterface.TRANSFER_START_SLOT * 0x4000
    stage, source = MagicMock(), MagicMock()
    stage.size = arena_size
    self.allocator.b = [stage]
    self.allocator.dev.device, self.allocator.dev.hw_copy_queue_t = "NV", MagicMock()
    self.allocator.dev.timeline_value = 7
    events = []
    def next_timeline():
      events.append("next")
      value = self.allocator.dev.timeline_value
      self.allocator.dev.timeline_value += 1
      return value
    self.allocator.dev.next_timeline.side_effect = next_timeline

    usb, completion, completion_view = MagicMock(), MagicMock(), MagicMock()
    self.allocator.dev.iface.pci_dev, self.allocator.dev.iface.cq_buf = MagicMock(), MagicMock()
    self.allocator.dev.iface.pci_dev.usb = usb
    self.allocator.dev.iface.cq_buf.offset.return_value = completion
    completion.va_addr, completion.cpu_view.return_value = 0x12345000, completion_view
    completion_view.__getitem__.return_value = b"\x04\x00\x00\x00"

    expected = bytes((i * 17 + 3) & 0xff for i in range(arena_size + 17))
    responses = []
    for chunk in (expected[:arena_size], expected[arena_size:]):
      raw = bytes(prefix_size) + chunk
      responses.append(memoryview(raw + bytes((-len(raw)) % 512)))
    usb.usb.bulk_read.side_effect = responses

    queues = [MagicMock(), MagicMock()]
    for queue in queues:
      queue.wait.return_value = queue
      queue.copy.return_value = queue
      queue.write.return_value = queue
      queue.signal.return_value = queue
      queue.bind.side_effect = lambda *_: events.append("bind")
      queue.submit.side_effect = lambda *_: events.append("submit")
    result = memoryview(bytearray(len(expected)))
    with patch("tinygrad.runtime.ops_nv.NVCopyQueue", side_effect=queues), \
         patch("tinygrad.runtime.ops_nv.hcq_profile", return_value=contextlib.nullcontext()):
      self.allocator._copyout(result, source)

    self.assertEqual(bytes(result), expected)
    self.assertEqual(usb.scsi_read_arm.call_args_list,
                     [unittest.mock.call(prefix_size + arena_size, start_slot=0), unittest.mock.call(prefix_size + 17, start_slot=0)])
    for queue in queues: queue.write.assert_called_once_with(completion, 0)
    self.assertEqual(completion_view.__setitem__.call_count, 2)
    self.assertEqual(events, ["bind", "next", "submit"] * 2)
    usb.pcie_mem_read.assert_not_called()


class FakeQueueController:
  def __init__(self):
    self.sram, self.writes, self.reads = bytearray(ASM24GSPQueueInterface.SRAM_SIZE), [], []

  def sram_read(self, offset, size):
    self.reads.append((offset, size))
    return bytes(self.sram[offset:offset+size])
  def scsi_write(self, data, start_slot=0):
    offset = start_slot * ASM24GSPQueueInterface.SLOT_SIZE
    self.sram[offset:offset+len(data)] = data
    self.writes.append((start_slot, bytes(data)))


class TestASM24GSPQueueInterface(unittest.TestCase):
  def setUp(self):
    self.controller = FakeQueueController()
    self.queue = ASM24GSPQueueInterface(self.controller)

  def test_nvidia_compact_page_map_routes_each_queue_region(self):
    queue = self.queue
    self.assertEqual(queue.paddrs(), list(ASM24GSPQueueInterface.PAGE_PADDRS))
    self.assertEqual(queue.PAGE_PADDRS[:6], (0x201000, 0x27F000, 0x27B000, 0x27C000, 0x27D000, 0x27E000))
    self.assertEqual(queue.PAGE_PADDRS[6:], (0x200000, 0x204000, 0x208000, 0x20C000, 0x210000))
    self.assertEqual(len(set(queue.PAGE_PADDRS[6:])), 5)
    self.assertEqual((queue.TRANSFER_PADDR, queue.TRANSFER_SIZE), (0x214000, 0x64000))

    queue[0:4] = b"PTES"
    queue.view(0x1000)[0:4] = b"CMDH"
    queue.view(0x2000)[0:4] = b"CMD0"

    self.assertEqual([(write[0], len(write[1])) for write in self.controller.writes], [(0, 0x4000), (31, 0x4000), (30, 0x4000)])
    self.assertEqual([self.controller.writes[0][1][0x1000:0x1004], self.controller.writes[1][1][0x3000:0x3004],
                      self.controller.writes[2][1][0x3000:0x3004]],
                     [b"PTES", b"CMDH", b"CMD0"])

    for logical_offset, paddr, value in ((0x6000, 0x200000, b"STAH"), (0x7000, 0x204000, b"STA0"),
                                         (0x8000, 0x208000, b"STA1"), (0x9000, 0x20C000, b"STA2"),
                                         (0xA000, 0x210000, b"STA3")):
      physical_offset = paddr - queue.SRAM_PADDR
      self.controller.sram[physical_offset:physical_offset+4] = value
      self.assertEqual(queue.view(logical_offset)[0:4], value)
    self.assertEqual(self.controller.reads, [(slot * 0x4000, 4) for slot in range(5)])

if __name__ == "__main__": unittest.main()
