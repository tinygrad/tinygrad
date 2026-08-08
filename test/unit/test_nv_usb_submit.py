import types, unittest
from unittest.mock import MagicMock, patch

from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.ops_nv import GPFifo, NVCommandQueue, NVCopyQueue, PCIIface, nv_flags


class RecordingMMIO:
  def __init__(self, events, name): self.events, self.name = events, name
  def __setitem__(self, index, value):
    self.events.append((self.name, index, value))


class TestNVUSBSubmission(unittest.TestCase):
  def test_copy_queue_write_uses_one_word_semaphore_release(self):
    queue, target = NVCopyQueue(), types.SimpleNamespace(va_addr=0x123456789ABC)

    self.assertIs(queue.write(target, 0xA5A55A5A), queue)
    self.assertEqual(queue._q, [
      (2 << 28) | (3 << 16) | (4 << 13) | (nv_gpu.NVC6B5_SET_SEMAPHORE_A >> 2),
      0x1234, 0x56789ABC, 0xA5A55A5A,
      (2 << 28) | (1 << 16) | (4 << 13) | (nv_gpu.NVC6B5_LAUNCH_DMA >> 2),
      nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_one_word_semaphore"),
    ])

  def test_submission_without_pci_transport(self):
    events = []
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(), gpu_mmio=RecordingMMIO(events, "doorbell"))
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x1234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertEqual([event[0] for event in events], ["ring", "gpput", "doorbell"])
    self.assertEqual(events[-1], ("doorbell", 0x90 // 4, gpfifo.token))
    queue.binded_device = None

  def test_device_fini_passes_root_to_gsp_teardown(self):
    iface = object.__new__(PCIIface)
    iface.root, iface.dev_impl = 0xC1000000, MagicMock()

    iface.device_fini()

    iface.dev_impl.fini.assert_called_once_with(iface.root)

if __name__ == "__main__": unittest.main()
