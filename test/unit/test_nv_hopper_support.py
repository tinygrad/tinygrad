import unittest

from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.ops_nv import _select_nv_classes_from_classlist
from tinygrad.runtime.support.nv.ip import _select_gsp_classes_from_chip


class TestNVHopperSupport(unittest.TestCase):
  def test_classlist_selection_supports_hopper(self):
    classes = {
      nv_gpu.HOPPER_USERMODE_A,
      nv_gpu.HOPPER_CHANNEL_GPFIFO_A,
      nv_gpu.HOPPER_COMPUTE_A,
      nv_gpu.HOPPER_DMA_COPY_A,
    }

    usermode_class, gpfifo_class, compute_class, dma_class = _select_nv_classes_from_classlist(classes, nv_gpu)

    self.assertEqual(usermode_class, nv_gpu.HOPPER_USERMODE_A)
    self.assertEqual(gpfifo_class, nv_gpu.HOPPER_CHANNEL_GPFIFO_A)
    self.assertEqual(compute_class, nv_gpu.HOPPER_COMPUTE_A)
    self.assertEqual(dma_class, nv_gpu.HOPPER_DMA_COPY_A)

  def test_gsp_chip_prefix_selection_supports_hopper(self):
    gpfifo_class, compute_class, dma_class = _select_gsp_classes_from_chip("GH200", nv_gpu)

    self.assertEqual(gpfifo_class, nv_gpu.HOPPER_CHANNEL_GPFIFO_A)
    self.assertEqual(compute_class, nv_gpu.HOPPER_COMPUTE_A)
    self.assertEqual(dma_class, nv_gpu.HOPPER_DMA_COPY_A)


if __name__ == "__main__":
  unittest.main()
