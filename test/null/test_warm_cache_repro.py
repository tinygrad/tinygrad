import unittest
from tinygrad import Tensor

class TestWarmCacheRepro(unittest.TestCase):
  def test_warm_method_cache(self):
    for t in [Tensor.empty(4).add(1), Tensor.empty(8).add(1)]:
      for si in Tensor.schedule(t): si.lower()

if __name__ == "__main__":
  unittest.main()
