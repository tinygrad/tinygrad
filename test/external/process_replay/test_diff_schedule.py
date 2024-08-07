import unittest
from test.external.process_replay.diff_schedule import diff_schedule
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.engine.schedule import SCHEDULES

class TestDiffSchedule(unittest.TestCase):
  def test_diff_arange(self):
    # diff a single arange kernel
    X = Tensor.randn(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = X[idxs]
    with Context(ARANGE_DIFF=1): xt.schedule()
    self.assertEqual(len(SCHEDULES), 2)
    changed = diff_schedule(SCHEDULES)
    self.assertEqual(changed, 1)
    SCHEDULES.clear()

    # no diff
    a = Tensor([1])+Tensor([2])
    with Context(ARANGE_DIFF=1): a.schedule()
    self.assertEqual(len(SCHEDULES), 2)
    changed = diff_schedule(SCHEDULES)
    self.assertEqual(changed, 0)
    SCHEDULES.clear()

    # no diff with two schedule creation calls
    a = Tensor([1])+Tensor([2])
    with Context(ARANGE_DIFF=1): a.schedule()
    b = Tensor([3])+Tensor([4])
    with Context(ARANGE_DIFF=1): b.schedule()
    self.assertEqual(len(SCHEDULES), 4)
    changed = diff_schedule(SCHEDULES)
    self.assertEqual(changed, 0)

if __name__ == '__main__':
  unittest.main()
