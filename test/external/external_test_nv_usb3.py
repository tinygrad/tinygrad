import ctypes, unittest
from unittest.mock import patch
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.runtime.autogen import libusb, pci, nv_570 as nv_gpu
from tinygrad.runtime.ops_nv import NVCopyQueue, USBIface
from tinygrad.runtime.support.usb import ASM24GSPQueueInterface, USB3

USB_IDS = {(0xADD1, 0x0001), (0x3801, 0x0001)}
RTX_3090_PCI_ID = 0x220410DE
CHRAM_FAULT_BITS = (1 << 4) | (1 << 5) | (1 << 12)


class TestNVUSB3(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not (USB3.list_devices(0xADD1, 0x0001) + USB3.list_devices(0x3801, 0x0001)):
      raise unittest.SkipTest("no ASM2464 USB device found")
    cls.dev = Device["NV"]
    if not isinstance(cls.dev.iface, USBIface): raise unittest.SkipTest("run with DEV=USB+NV:NAK")
    cls.iface = cls.dev.iface

  @classmethod
  def tearDownClass(cls):
    cls.dev.finalize()

  def _assert_aer_clean(self):
    for bus in range(self.iface.pci_dev.gpu_bus + 1):
      uncorrectable = self.iface.pci_dev.usb.pcie_cfg_req(0x104, bus=bus, size=4)
      correctable = self.iface.pci_dev.usb.pcie_cfg_req(0x110, bus=bus, size=4)
      self.assertEqual((uncorrectable, correctable), (0, 0), f"PCIe AER fault on bus {bus}")

  def test_device_identity(self):
    raw_usb = self.iface.pci_dev.usb.usb
    usb_dev = libusb.libusb_get_device(raw_usb.handle)
    descriptor = libusb.struct_libusb_device_descriptor()
    self.assertEqual(libusb.libusb_get_device_descriptor(usb_dev, ctypes.byref(descriptor)), 0)
    self.assertIn((int(descriptor.idVendor), int(descriptor.idProduct)), USB_IDS)
    self.assertEqual(libusb.libusb_get_device_speed(usb_dev), libusb.LIBUSB_SPEED_SUPER)
    self.assertEqual(self.iface.pci_dev.read_config(pci.PCI_VENDOR_ID, 4), RTX_3090_PCI_ID)

  def test_compute_and_exact_transfer(self):
    usb = self.iface.pci_dev.usb
    with patch.object(usb, "pcie_mem_write", wraps=usb.pcie_mem_write) as slow_write, \
         patch.object(usb, "pcie_mem_read", wraps=usb.pcie_mem_read) as slow_read:
      self.assertEqual((Tensor([1., 2., 3., 4.], device="NV") * 3 + 1).tolist(), [4., 7., 10., 13.])
      source = np.arange((4 << 20) // 4, dtype=np.uint32) ^ np.uint32(0xA5A55A5A)
      result = Tensor(source, device="NV").contiguous().realize().numpy()
    np.testing.assert_array_equal(result, source)
    self.assertFalse(any(len(call.args[1]) >= ASM24GSPQueueInterface.TRANSFER_SIZE for call in slow_write.call_args_list))
    self.assertFalse(any(call.args[1] >= ASM24GSPQueueInterface.TRANSFER_SIZE for call in slow_read.call_args_list))
    self._assert_aer_clean()

  def test_compact_sram_transfer_arena(self):
    layout, usb = ASM24GSPQueueInterface, self.iface.pci_dev.usb
    prefix_size, read_size = layout.TRANSFER_PADDR - layout.SRAM_PADDR, layout.TRANSFER_PADDR - layout.SRAM_PADDR + layout.TRANSFER_SIZE

    saved_completion = usb.read(0xB80C, 4)
    try:
      sram, completion = self.iface.copy_bufs[0], self.iface.cq_buf
      self.assertEqual((sram.meta.mapping.paddrs, sram.size), ([(layout.TRANSFER_PADDR, layout.TRANSFER_SIZE)], layout.TRANSFER_SIZE))
      pattern = bytes((i * 47 + 23) & 0xff for i in range(layout.TRANSFER_SIZE))
      source = self.iface.alloc(layout.TRANSFER_SIZE, cpu_access=True, zero=False)
      source.cpu_view().view(size=layout.TRANSFER_SIZE, fmt='B')[:] = pattern

      signal_value = self.dev.timeline_value
      queue = NVCopyQueue().wait(self.dev.timeline_signal, signal_value - 1).copy(sram, source, layout.TRANSFER_SIZE) \
                           .write(completion.offset(12), 0).signal(self.dev.timeline_signal, signal_value)
      queue.bind(self.dev)

      usb.scsi_read_arm(read_size, start_slot=0)
      self.assertEqual(self.dev.next_timeline(), signal_value)
      queue.submit(self.dev)
      enclosing = bytes(usb.usb.bulk_read(read_size, timeout=1000))
      self.dev.timeline_signal.wait(signal_value, timeout=1000)
      self.assertEqual(enclosing[0x1000:0x2000], bytes(self.iface.pci_dev.gsp_queues._root._mirror[0x1000:0x2000]))
      self.assertEqual(enclosing[prefix_size:], pattern)
      self._assert_aer_clean()
    finally: usb.write(0xB80C, saved_completion)

  def test_channel_and_aer_health(self):
    self.dev.synchronize()
    params = nv_gpu.NV2080_CTRL_GPU_GET_ENGINE_RUNLIST_PRI_BASE_PARAMS()
    params.engineList[0] = nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS
    runlist = self.iface.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GPU_GET_ENGINE_RUNLIST_PRI_BASE, params)
    runlist_id, runlist_pri_base = int(runlist.runlistId[0]), int(runlist.runlistPriBase[0])
    base_index, chram = 0, None
    while chram is None:
      table = self.iface.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_FIFO_GET_DEVICE_INFO_TABLE,
        nv_gpu.NV2080_CTRL_FIFO_GET_DEVICE_INFO_TABLE_PARAMS(baseIndex=base_index))
      entry = next((x for x in table.entries[:table.numEntries]
                    if x.engineData[3] == runlist_id and x.engineData[11] == runlist_pri_base), None)
      if entry is not None: chram = self.iface.pci_dev.map_bar(0, fmt='I', off=int(entry.engineData[14]), size=0x2000)
      elif not table.bMore or not table.numEntries: self.fail("graphics CHRAM table entry unavailable")
      else: base_index += table.numEntries
    for name in ("compute", "dma"):
      fifo = getattr(self.dev, f"{name}_gpfifo")
      state = chram[fifo.token]
      self.assertEqual(state & CHRAM_FAULT_BITS, 0, f"{name} CHRAM fault bits set: {state:#x}")
    self._assert_aer_clean()


if __name__ == "__main__": unittest.main()
