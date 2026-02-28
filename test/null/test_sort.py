import unittest
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters, Context
import numpy as np

class TestSortComplexity(unittest.TestCase):
  def _sort_values_ops(self, n:int) -> int:
    t = Tensor.randn(n, device="NULL").realize()
    GlobalCounters.reset()
    t.sort()[0].realize()
    return GlobalCounters.global_ops

  def _sort_indices_ops(self, n:int) -> int:
    t = Tensor.randn(n, device="NULL").realize()
    GlobalCounters.reset()
    t.sort()[1].realize()
    return GlobalCounters.global_ops

  def _sort_both_ops(self, n:int) -> int:
    t = Tensor.randn(n, device="NULL").realize()
    values, indices = t.sort()
    GlobalCounters.reset()
    Tensor.realize(values, indices)
    return GlobalCounters.global_ops

  def test_sort_values_complexity(self):
    with Context(NOOPT=1, SPLIT_REDUCEOP=0):
      ops_64 = self._sort_values_ops(64)
      ops_256 = self._sort_values_ops(256)
    self.assertLess(ops_256, int(ops_64*7.2), f"value sort growth too high with NOOPT=1 SPLIT_REDUCEOP=0: {ops_64=} {ops_256=}")
    np.testing.assert_allclose(ops_256, 85628, rtol=0.01)

  def test_sort_indices_complexity(self):
    with Context(NOOPT=1, SPLIT_REDUCEOP=0):
      ops_64 = self._sort_indices_ops(64)
      ops_256 = self._sort_indices_ops(256)
    self.assertLess(ops_256, int(ops_64*8.0), f"index sort growth too high with NOOPT=1 SPLIT_REDUCEOP=0: {ops_64=} {ops_256=}")
    np.testing.assert_allclose(ops_256, 183292, rtol=0.01)

  def test_sort_corealize_values_indices(self):
    with Context(NOOPT=1, SPLIT_REDUCEOP=0):
      indices_ops = self._sort_indices_ops(256)
      both_ops = self._sort_both_ops(256)
    self.assertLess(both_ops, int(indices_ops*1.2), f"co-realize should share sort work with NOOPT=1 SPLIT_REDUCEOP=0: {indices_ops=} {both_ops=}")

if __name__ == '__main__':
  unittest.main()
