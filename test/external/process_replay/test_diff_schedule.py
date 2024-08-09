from typing import cast
import unittest
from test.external.process_replay.diff_schedule import diff_schedule
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.engine.schedule import _graph_schedule
from tinygrad.lazy import LazyBuffer

class TestDiffSchedule(unittest.TestCase):
  def test_diff_arange(self):
    # diff a single arange kernel
    X = Tensor.randn(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = cast(LazyBuffer, X[idxs].lazydata)
    with Context(FUSE_ARANGE=0): ref_graph, ref_in_degree = _graph_schedule([xt], set())
    with Context(FUSE_ARANGE=1): compare_graph, compare_in_degree = _graph_schedule([xt], set())
    # 1 arange LazyBuffer folds, 1 arange child's kernel changes
    changed = diff_schedule([(ref_graph, ref_in_degree), (compare_graph, compare_in_degree)])
    self.assertEqual(changed, 2)

    # no diff
    a = cast(LazyBuffer, (Tensor([1])+Tensor([2])).lazydata)
    with Context(FUSE_ARANGE=0): ref_graph, ref_in_degree = _graph_schedule([a], set())
    with Context(FUSE_ARANGE=1): compare_graph, compare_in_degree = _graph_schedule([a], set())
    changed = diff_schedule([(ref_graph, ref_in_degree), (compare_graph, compare_in_degree)])
    self.assertEqual(changed, 0)

if __name__ == '__main__':
  unittest.main()
