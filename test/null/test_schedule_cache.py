import unittest
from tinygrad import Tensor, Variable, Context
from tinygrad.helpers import cpu_events
from tinygrad.codegen import schedule_cache

def schedule_one():
  Tensor([1]).schedule_linear()

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_var_vals(self):
    v = Variable('pos', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    t = x + Tensor(v.bind(42))
    _, var_vals = t.linear_with_vars()
    self.assertEqual(var_vals, {'pos': 42})

  def test_view_offset_not_in_cache_key(self):
    # every use of a contiguous view is the same input, so slices at different offsets share one schedule
    x = Tensor.arange(16).float().clone().realize()
    schedule_cache.clear()
    for start in (1, 2, 3):
      v = x[start:start+4]
      (v.contiguous() + v).realize()
      self.assertEqual(sum(isinstance(k, bytes) for k in schedule_cache), 1)

  def test_disable_schedule_cache(self):
    schedule_cache.clear()

    # test write
    with Context(SCACHE=0): schedule_one()
    self.assertEqual(len(schedule_cache), 0)
    with Context(SCACHE=1):
      schedule_one()
      schedule_one()
    self.assertEqual(len(schedule_cache), 2)  # one body schedule and one final LINEAR

    # test read
    with Context(PROFILE=1):
      cpu_events.clear()
      with Context(SCACHE=0): schedule_one()
      num_events_no_cache = len(cpu_events)

      cpu_events.clear()
      with Context(SCACHE=1): schedule_one()
      num_events_cache = len(cpu_events)
    self.assertLess(num_events_cache, num_events_no_cache)

if __name__ == "__main__":
  unittest.main()
