import unittest
from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.amd_qwen import after_kernel, rows_scatter
from tinygrad.schedule import create_schedule
from tinygrad.schedule.prepare import prepare_rangeify
from tinygrad.schedule.rangeify import get_kernel_graph

class TestPrefillPairAssignCycle(unittest.TestCase):
  def schedule(self, *states):
    # The outer AFTER reproduces seq_prefill_pair's two tails, not a serialization of their internal cache writers.
    return create_schedule(get_kernel_graph(prepare_rangeify(states[0].uop.after(*[s.uop for s in states[1:]]))))

  def writers(self, cache, x, y):
    first = rows_scatter(cache, x, Tensor([0], dtype=dtypes.int32))
    second = rows_scatter(cache, y, Tensor([1], dtype=dtypes.int32))
    return first, second

  def test_independent_cache_writers_cycle(self):
    cache = Tensor.zeros(2, 4).contiguous().realize()
    first, second = self.writers(cache, Tensor.ones(1, 4), Tensor.full((1, 4), 2.0))
    # Runtime indices are disjoint, but opaque custom-kernel calls give the scheduler no disjointness proof.
    with self.assertRaisesRegex(RuntimeError, "cycle detected in assign graph"):
      self.schedule(first, second)

  def test_value_dependency_breaks_cycle(self):
    cache = Tensor.zeros(2, 4).contiguous().realize()
    first = rows_scatter(cache, Tensor.ones(1, 4), Tensor([0], dtype=dtypes.int32))
    # The second call's read/write state consumes its values, not just an outer scheduling annotation.
    cache_in = after_kernel(cache, first)
    second = rows_scatter(cache_in, Tensor.full((1, 4), 2.0), Tensor([1], dtype=dtypes.int32))
    self.assertEqual(len(self.schedule(first, second).src), 7)

if __name__ == "__main__": unittest.main()
