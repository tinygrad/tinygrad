import unittest
import numpy as np
import tinygrad.codegen.simplify as simp
from tinygrad import Tensor, Device
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, graph_rewrite, tracked_ctxs

def count_check_merge(n:int) -> int:
  # compile an n-axis reduce kernel with match stats on, counting check_merge rewrites (each is a full graph walk)
  linear = Tensor.empty(*([2]*n)).sum().schedule_linear()
  asts = [u.src[0] for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK]
  assert len(asts) == 1
  with Context(TRACK_MATCH_STATS=2):
    tracked_ctxs.clear()
    full_rewrite_to_sink(asts[0], Device[Device.DEFAULT].renderer)
  return sum(tr.name.startswith("check_merge") for ctx in tracked_ctxs for tr in ctx)

def merged_reduce_ranges(*shape):
  # apply just the "simplify ranges" pass (full_rewrite_to_sink eliminates the REDUCE entirely in later passes)
  linear = Tensor.empty(*shape).sum().schedule_linear()
  ast = [u.src[0] for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.SINK][0]
  out = graph_rewrite(ast, simp.pm_flatten_range+simp.pm_simplify_ranges, ctx={}, name="simplify ranges")
  reds = [u for u in out.toposort() if u.op is Ops.REDUCE]
  assert len(reds) == 1
  return reds[0].ended_ranges

class TestMergeAdjacentScaling(unittest.TestCase):
  # on b53cd35cf every ordered pair of ranges was rewritten: 70 check_merge rewrites for n=8.
  # with greedy restart per merge, each merge is accepted on its first try: n-1 rewrites.
  # (repeat compiles of related graphs in one process can do less work due to rewrite caches, so these are upper bounds)
  # measured with TRACK_MATCH_STATS=2 (the same data VIZ shows), base -> this branch:
  #   check_merge rewrites:  70 -> 7 (n=8),  310 -> 15 (n=16)
  #   full_rewrite_to_sink: 16.4ms -> 7.7ms (n=8),  172.5ms -> 17.2ms (n=20, 10x)
  def test_check_merge_rewrites(self):
    self.assertLessEqual(count_check_merge(8), 7)

  def test_check_merge_rewrites_large(self):
    self.assertLessEqual(count_check_merge(16), 15)

  # greedy merging collapses all 4 adjacent same-type ranges into one range of size 2*3*4*5
  def test_greedy_merge_outcome(self):
    ranges = merged_reduce_ranges(2,3,4,5)
    self.assertEqual(len(ranges), 1)
    self.assertEqual(ranges[0].src[0].arg, 120)

  # the merged range must index correctly (the //s1, %s1 substitution): sum of 0..119
  @unittest.skipIf(Device.DEFAULT == "NULL", "no copyout on NULL")
  def test_merge_preserves_numerics(self):
    self.assertEqual(Tensor(np.arange(120, dtype=np.int32).reshape(2,3,4,5)).sum().item(), 7140)

if __name__ == '__main__':
  unittest.main(verbosity=2)
