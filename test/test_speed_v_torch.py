import unittest
import torch
import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d

CNT = 5
class TestSpeedVTorch(unittest.TestCase):
  def test_conv2d(self):
    torch.manual_seed(0)
    for bs in [32]:
      for in_chans in [1,16,64]:
        for out_chans in [64]:
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
          img_size = 64 if device == 'cuda' else 32
          src = torch.rand(bs, in_chans, img_size, img_size)
          dat = src.clone().to(device)
          src_conv = torch.nn.Conv2d(in_chans, out_chans, 3)
          conv = src_conv.to(device)
          with torch.no_grad():
            val_torch = conv(dat).cpu().numpy().sum()
            ets_torch = []
            for _ in range(CNT):
              dat += 1
              st = time.monotonic()
              val_torch = conv(dat).cpu().numpy().sum()
              et_torch = (time.monotonic() - st) * 1000
              ets_torch.append(et_torch)

          Tensor.no_grad = False
          dat = Tensor(src.numpy())
          conv = Conv2d(in_chans, out_chans, 3)
          conv.weight = Tensor(src_conv.weight.detach().cpu().numpy())
          conv.bias = Tensor(src_conv.bias.detach().cpu().numpy())
          val_tinygrad = conv(dat).numpy().sum()
          ets_tinygrad = []
          for _ in range(CNT):
            dat += 1
            dat.realize()
            st = time.monotonic()
            val_tinygrad = conv(dat).numpy().sum()
            et_tinygrad = (time.monotonic() - st) * 1000
            ets_tinygrad.append(et_tinygrad)

          et_torch = np.median(ets_torch)
          et_tinygrad = np.median(ets_tinygrad)
          print(f"bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} {et_torch:7.2f} ms in torch({device}), {et_tinygrad:7.2f} ms in tinygrad, {et_tinygrad/et_torch:7.2f}x slower", val_torch, val_tinygrad)
          relative_error = abs((val_tinygrad-val_torch)/val_torch)
          assert relative_error < 0.01

if __name__ == '__main__':
  unittest.main()
