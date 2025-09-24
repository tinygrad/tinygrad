import unittest, math
from tinygrad import Tensor, dtypes
from examples.mlperf.lr_schedulers import LambdaLR, LambdaLinearScheduler
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

class TestLRScheduler(unittest.TestCase):
  def test_linear_lr_warmup(self):
    BS, BASE_LR = 304, 2.5e-7
    lr = BS * BASE_LR
    # Use a dummy Tensor parameter for optimizer because the lr_scheduler only needs the optimizer's device and lr, the params aren't touched.
    optimizer = AdamW(get_parameters([Tensor([1.])]))
    lambda_lr_callback = LambdaLinearScheduler(1000, 1.0, 1.0, 1e-06, 10000000000000).schedule
    lr_scheduler = LambdaLR(optimizer, Tensor(lr, dtype=dtypes.float, device=optimizer.device), lambda_lr_callback)
    lrs = {}

    # with above settings, optimizer.lr should warm up to lr over 1000 steps linearly
    for i in range(1200):
      lr_scheduler.step()
      if i in {0, 499, 998, 999, 1000, 1199}:
        lrs[i] = optimizer.lr.item()

    assert math.isclose(lr, lrs[999], rel_tol=0.0, abs_tol=1e-11)
    assert lrs[999] == lrs[1000] == lrs[1199]
    assert math.isclose(lrs[999] / lrs[0], 1000, rel_tol=0.0, abs_tol=1)
    assert math.isclose(lrs[999] / lrs[499], 2, rel_tol=0.0, abs_tol=1e-5)

if __name__=="__main__":
  unittest.main()