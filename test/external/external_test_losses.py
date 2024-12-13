from tinygrad import Tensor
from test.external.mlperf_unet3d.dice import DiceCELoss
from test.external.mlperf_retinanet.focal_loss import sigmoid_focal_loss as ref_sigmoid_focal_loss
from examples.mlperf.losses import dice_ce_loss, sigmoid_focal_loss

import numpy as np
import torch
import unittest

class TestLoss(unittest.TestCase):
  def setUp(self):
    np.random.seed(1337)

  def assert_loss(self, pred, tgt, tinygrad_metrics, ref_metrics, rtol=1e-07, atol=0, **kwargs):
    tinygrad_metrics_res = tinygrad_metrics(Tensor(pred), Tensor(tgt), **kwargs)
    ref_metrics_res = ref_metrics(torch.from_numpy(pred), torch.from_numpy(tgt), **kwargs)
    np.testing.assert_allclose(tinygrad_metrics_res.numpy(), ref_metrics_res.numpy(), rtol=rtol, atol=atol)

class TestDiceCELoss(TestLoss):
  def setUp(self):
    np.random.seed(1337)

  def test_loss(self):
    pred, label = np.random.rand(1, 3, 128, 128, 128).astype(np.float32), np.ones((1, 1, 128, 128, 128)).astype(np.uint8)
    tinygrad_metrics_res, ref_metrics_res = dice_ce_loss, DiceCELoss(True, True, "NCDHW", False)
    self.assert_loss(pred, label, tinygrad_metrics_res, ref_metrics_res, atol=1e-4)

class TestSigmoidFocalLoss(TestLoss):
  def _generate_samples(self):
    def _apply_logit(p): return np.log(p / (1 - p))
    return _apply_logit(np.random.rand(5,2).astype(np.float32)), np.random.randint(0, 2, (5, 2)).astype(np.float32)

  def test_loss_equal_to_ce(self):
    for reduction in ["mean", "sum", "none"]:
      pred, tgt = self._generate_samples()
      self.assert_loss(pred, tgt, sigmoid_focal_loss, ref_sigmoid_focal_loss, rtol=1e-4, alpha=-1, gamma=0, reduction=reduction)

  def test_loss_correct_ratio(self):
    for reduction in ["mean", "sum", "none"]:
      pred, tgt = self._generate_samples()
      self.assert_loss(pred, tgt, sigmoid_focal_loss, ref_sigmoid_focal_loss, rtol=1e-4, alpha=0.58, gamma=2, reduction=reduction)


if __name__ == '__main__':
  unittest.main()