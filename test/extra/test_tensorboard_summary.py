import unittest, torch, numpy
import torch.utils.tensorboard.summary as torch_summary
import extra.tensorboard.summary as tiny_summary
from tinygrad.tensor import Tensor

class TestTensorboardSummary(unittest.TestCase):

  def base_test_scalar(self, value):
    key = "test_scalar_" + str(value)
    self.assertEqual(torch_summary.scalar(key, value), tiny_summary.scalar(key, value))
  def test_scalar(self):
    self.base_test_scalar(0)
    self.base_test_scalar(42.0)
    self.base_test_scalar(-69)

  def base_test_histogram(self, shape, bins, max_bins):
    key, values = "test_histogram_" + str(shape) + "_" + str(bins), numpy.random.randn(*shape)
    self.assertEqual(torch_summary.histogram(key, values, bins, max_bins),
                     tiny_summary.histogram(key, values, bins, max_bins))
  def test_histogram(self):
    self.base_test_histogram((1,), bins=1, max_bins=None)
    self.base_test_histogram((4,), bins=2, max_bins=None)
    self.base_test_histogram((4,), bins=2, max_bins=5)
    self.base_test_histogram((2,), bins=1, max_bins=3)
    self.base_test_histogram((6,), bins=9, max_bins=2)
    self.base_test_histogram((6,), bins='auto', max_bins=3)

  def base_test_image(self, shape, rescale=1, dataformats="NCHW"):
    key = "test_image_" + dataformats + "_" + str(rescale)
    self.assertEqual(torch_summary.image(key, torch.ones(shape), rescale, dataformats),
                     tiny_summary.image(key, Tensor.ones(shape), rescale, dataformats))
  def test_image(self):
    self.base_test_image((69, 3, 2, 1), 1, dataformats="NCHW")
    self.base_test_image((69, 3, 2, 1), 2, dataformats="NCHW")
    self.base_test_image((2, 1, 3), 1, dataformats="HWC")
    self.base_test_image((2, 1, 3), 2, dataformats="HWC")
    self.base_test_image((1, 1, 1), 2, dataformats="CHW")
    self.base_test_image((2, 1, 1), 2, dataformats="CHW")
    self.base_test_image((2, 1), 1, dataformats="HW")
    self.base_test_image((2, 1), 2, dataformats="HW")

  def base_test_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete):
    self.assertEqual(torch_summary.hparams(hparam_dict, metric_dict, hparam_domain_discrete),
                     tiny_summary.hparams(hparam_dict, metric_dict, hparam_domain_discrete))
  def test_hparams(self):
    self.base_test_hparams({'lr': 0.1, 'bsize': 1}, {'hparam/accuracy': 10, 'hparam/loss': 10}, None)
    self.base_test_hparams({'lr': 0.1, 'bsize': 1}, {'hparam/accuracy': 10, 'hparam/loss': 10}, {'lr': [0.1, 0.2], 'bsize': [1, 2]})
    self.base_test_hparams({'opt': 'SGD'}, {}, {'opt': ['SGD', 'Adam']})

  def base_test_text(self, tag, text_string):
    self.assertEqual(torch_summary.text(tag, text_string),
                     tiny_summary.text(tag, text_string))
  def test_text(self):
    self.base_test_text("test_text", "hello world")

if __name__ == '__main__': unittest.main()
