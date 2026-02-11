import unittest, time
from tinygrad import Tensor

class TestScheduleScaling(unittest.TestCase):
  """Test that .schedule() scales linearly with graph size (no O(n^2) behavior)."""

  def _assert_linear(self, fn, n_small=200, n_large=1000):
    """Assert schedule time scales at most ~linearly: time(n_large)/time(n_small) should be close to n_large/n_small."""
    fn(n_small).schedule()  # warmup
    t_small = min(self._time_schedule(fn, n) for n in [n_small]*3)
    t_large = min(self._time_schedule(fn, n) for n in [n_large]*3)
    size_ratio = n_large / n_small  # 5.0
    time_ratio = t_large / t_small
    # O(n) -> time_ratio ~ 5, O(n^2) -> time_ratio ~ 25. threshold at 10 catches n^2 with margin.
    self.assertLess(time_ratio / size_ratio, 2.0,
      f"schedule appears superlinear: n={n_small} {t_small*1e3:.1f}ms, n={n_large} {t_large*1e3:.1f}ms "
      f"(time grew {time_ratio:.1f}x for {size_ratio:.0f}x size, per-node ratio {time_ratio/size_ratio:.2f})")

  @staticmethod
  def _time_schedule(fn, n) -> float:
    st = time.perf_counter()
    fn(n).schedule()
    return time.perf_counter() - st

  # ending_ranges accumulation via sum([], []) and nested scan in run_rangeify.
  # this creates reduce ops whose ending_ranges lists grow with graph depth, causing O(n^2) list copies.
  def test_multi_reduce_scaling(self):
    def multi_reduce(n):
      x = Tensor.empty(256, 256)
      for _ in range(n):
        s = x.sum(axis=-1, keepdim=True)
        x = x + s + s
      return x
    self._assert_linear(multi_reduce)

  # reduce+elementwise chain stresses ending_ranges propagation and post-rangeify rewrites
  def test_wide_reduce_scaling(self):
    def wide_reduce(n):
      x = Tensor.empty(256, 256)
      for _ in range(n):
        x = x + x.sum(axis=-1, keepdim=True)
      return x
    self._assert_linear(wide_reduce)

  # multi-consumer diamond pattern (fan-out/fan-in) stresses consumer_rngs merge in run_rangeify
  def test_diamond_scaling(self):
    def diamond(n):
      x = Tensor.empty(256, 256)
      for _ in range(n):
        a = x + 1
        b = x + 2
        x = a + b
      return x
    self._assert_linear(diamond)

  # elementwise chain baseline — should be trivially O(n)
  def test_chain_scaling(self):
    def chain(n):
      x = Tensor.empty(256, 256)
      for _ in range(n): x = x + 1
      return x
    self._assert_linear(chain)

  # expand ops inject into ending_ranges via the EXPAND path in run_rangeify
  def test_expand_reduce_scaling(self):
    def expand_reduce(n):
      x = Tensor.empty(256, 1)
      for _ in range(n):
        y = x.expand(256, 256)
        x = (y + y).sum(axis=-1, keepdim=True)
      return x
    self._assert_linear(expand_reduce)

if __name__ == '__main__':
  unittest.main(verbosity=2)
