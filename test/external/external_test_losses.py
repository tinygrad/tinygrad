from tinygrad import Tensor
from test.external.mlperf_unet3d.dice import DiceCELoss
from examples.mlperf.losses import dice_ce_loss

import nibabel as nib
import numpy as np
import pathlib
import torch
import unittest

class ExternalTestLosses(unittest.TestCase):
  def _test_losses(self, tinygrad_metrics, orig_metrics, pred, label):
    tinygrad_metrics_res = tinygrad_metrics(Tensor(pred), Tensor(label)).numpy()
    orig_metrics_res = orig_metrics(torch.from_numpy(pred), torch.from_numpy(label)).numpy()
    np.testing.assert_allclose(tinygrad_metrics_res, orig_metrics_res, atol=1e-4)

  def test_dice_ce(self):
    seg = nib.load(pathlib.Path(__file__).parent / "mlperf_unet3d" / "segmentation.nii.gz")
    pred, label = seg.get_fdata().astype(np.float32), seg.get_fdata().astype(np.uint8)
    pred, label = np.repeat(pred[None, None], 3, axis=1), label[None]
    self._test_losses(dice_ce_loss, DiceCELoss(True, True, "NCDHW", False), pred, label)

if __name__ == '__main__':
  unittest.main()