from tinygrad import Tensor
from test.external.mlperf_unet3d.dice import DiceScore
from examples.mlperf.metrics import dice_score

import nibabel as nib
import numpy as np
import pathlib
import torch
import unittest

class ExternalTestMetrics(unittest.TestCase):
  def _test_metrics(self, tinygrad_metrics, orig_metrics, pred, label):
    tinygrad_metrics_res = tinygrad_metrics(Tensor(pred), Tensor(label)).squeeze().numpy()
    orig_metrics_res = orig_metrics(torch.from_numpy(pred), torch.from_numpy(label)).numpy()
    np.testing.assert_equal(tinygrad_metrics_res, orig_metrics_res)

  def test_dice(self):
    seg = nib.load(pathlib.Path(__file__).parent / "mlperf_unet3d" / "segmentation.nii.gz")
    pred, label = seg.get_fdata().astype(np.float32), seg.get_fdata().astype(np.uint8)
    pred, label = np.repeat(pred[None, None], 3, axis=1), label[None]
    self._test_metrics(dice_score, DiceScore(), pred, label)

if __name__ == '__main__':
  unittest.main()