from examples.mlperf.losses import dice_ce_loss
from test.external.mlperf_unet3d.dice_loss import DiceCELoss

from tinygrad import Tensor

import numpy as np
import unittest
import torch

def tinygrad_dice_ce_loss(pred, target):
  return dice_ce_loss(pred, target)

def torch_dice_ce_loss(to_onehot_y, use_softmax, layout, include_background):
  return DiceCELoss(to_onehot_y, use_softmax, layout, include_background)

class ExternalTestLosses(unittest.TestCase):
  def test_dice_loss(self):
    torch_dice_loss = torch_dice_ce_loss(True, True, "NCDHW", False)
    pred = Tensor.randn((1, 3, 128, 128, 128))
    tgt = Tensor.arange(0, 1).reshape(1, -1, 1, 1, 1).expand(1, -1, 128, 128, 128)

    tinygrad_loss = tinygrad_dice_ce_loss(pred, tgt)
    torch_loss = torch_dice_loss(torch.from_numpy(pred.numpy()), torch.from_numpy(tgt.numpy()))

    np.testing.assert_allclose(tinygrad_loss.numpy(), torch_loss.numpy())

if __name__ == "__main__":
  unittest.main()
