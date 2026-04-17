import math, unittest

from tinygrad import Tensor
from tinygrad.nn.optim import AdamW

from examples.mlperf.lr_schedulers import CosineAnnealingLRWithWarmup


class TestMLPerfLRSchedulers(unittest.TestCase):
  def test_cosine_scheduler_allows_zero_warmup(self):
    optim = AdamW([Tensor([1.0])], lr=0.0)
    sched = CosineAnnealingLRWithWarmup(optim, 4e-4, 0.0, 0, 8)

    lr = []
    for _ in range(8):
      lr.append(optim.lr.item())
      sched.step()

    expected = [4e-4 * (1 + math.cos((i + 1) * math.pi / 8)) / 2 for i in range(8)]
    self.assertEqual(len(lr), len(expected))
    for got, want in zip(lr, expected):
      self.assertAlmostEqual(got, want, places=10)


if __name__ == "__main__":
  unittest.main()
