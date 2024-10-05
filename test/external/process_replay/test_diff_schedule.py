from typing import cast
import unittest
from test.external.process_replay.diff_schedule import CAPTURING_PROCESS_REPLAY, diff_schedule
from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.engine.schedule import _graph_schedule
from tinygrad.engine.lazy import LazyBuffer

class TestDiffSchedule(unittest.TestCase):
  def setUp(self):
    self.old_value = CAPTURING_PROCESS_REPLAY.value
    CAPTURING_PROCESS_REPLAY.value = 0
  def tearDown(self):
    CAPTURING_PROCESS_REPLAY.value = self.old_value

  def test_diff_arange(self):
    # diff a single arange kernel
    X = Tensor.randn(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = cast(LazyBuffer, X[idxs].lazydata)
    with Context(FUSE_ARANGE=0): ref_graph, ref_in_degree, _ = _graph_schedule([xt])
    with Context(FUSE_ARANGE=1): compare_graph, compare_in_degree, _ = _graph_schedule([xt])
    # 1 arange LazyBuffer folds, 1 arange child's kernel changes
    changed = diff_schedule([(ref_graph, ref_in_degree), (compare_graph, compare_in_degree)])
    self.assertEqual(changed, 1)

  def test_diff_dedups(self):
    idxs = Tensor([0, 2]).realize()
    schedules = []
    for _ in range(2):
      X = Tensor.randn(10, 10).realize()
      xt = cast(LazyBuffer, X[idxs].lazydata)
      with Context(FUSE_ARANGE=0): schedules.append(_graph_schedule([xt])[:-1])
      with Context(FUSE_ARANGE=1): schedules.append(_graph_schedule([xt])[:-1])
    changed = diff_schedule(schedules)
    self.assertEqual(changed, 1)

  def test_no_diff(self):
    a = cast(LazyBuffer, (Tensor([1])+Tensor([2])).lazydata)
    with Context(FUSE_ARANGE=0): ref_graph, ref_in_degree, _ = _graph_schedule([a])
    with Context(FUSE_ARANGE=1): compare_graph, compare_in_degree, _ = _graph_schedule([a])
    changed = diff_schedule([(ref_graph, ref_in_degree), (compare_graph, compare_in_degree)])
    self.assertEqual(changed, 0)

  def test_diff_fused_conv_bw(self):
    c1 = nn.Conv2d(3,16,3, bias=False)
    c1.weight.requires_grad = True
    img = Tensor.rand(2,3,64,64, requires_grad=True)
    c1(img).relu().mean().backward()
    assert img.grad is not None and c1.weight.grad is not None
    outs = [cast(LazyBuffer, img.grad.lazydata), cast(LazyBuffer, c1.weight.grad.lazydata)]
    with Context(FUSE_CONV_BW=0): ref_graph, ref_in_degree, _ = _graph_schedule(outs)
    with Context(FUSE_CONV_BW=1): compare_graph, compare_in_degree, _ = _graph_schedule(outs)
    changed = diff_schedule([(ref_graph, ref_in_degree), (compare_graph, compare_in_degree)])
    # 1 reduceop folds, its child reduceop changes
    self.assertEqual(changed, 1)

if __name__ == '__main__':
  unittest.main()
