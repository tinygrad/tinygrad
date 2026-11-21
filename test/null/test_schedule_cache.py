import unittest
from unittest.mock import patch
from tinygrad import Tensor, Variable, Context, function
from tinygrad.helpers import cpu_events
from tinygrad.schedule import schedule_cache, canonicalize_call_body

def schedule_one():
  Tensor([1]).schedule_linear()

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_var_vals(self):
    v = Variable('pos', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    t = x + Tensor(v.bind(42))
    _, var_vals = t.linear_with_vars()
    self.assertEqual(var_vals, {'pos': 42})

  def test_disable_schedule_cache(self):
    schedule_cache.clear()

    # test write
    with Context(SCACHE=0): schedule_one()
    self.assertEqual(len(schedule_cache), 0)
    with Context(SCACHE=1):
      schedule_one()
      schedule_one()
    self.assertEqual(len(schedule_cache), 1)

    # test read
    with Context(PROFILE=1):
      cpu_events.clear()
      with Context(SCACHE=0): schedule_one()
      num_events_no_cache = len(cpu_events)

      cpu_events.clear()
      with Context(SCACHE=1): schedule_one()
      num_events_cache = len(cpu_events)
    self.assertLess(num_events_cache, num_events_no_cache)

  def test_nested_calls_canonicalized_once(self):
    def chain(d):
      if d == 0: return lambda x: x + 1
      inner = chain(d - 1)
      @function(precompile=True)
      def f(x): return inner(x) * 2
      return f
    with patch("tinygrad.schedule.canonicalize_call_body", side_effect=canonicalize_call_body) as c:
      chain(16)(Tensor.ones(4).contiguous()).realize()
    self.assertLess(c.call_count, 5 * 16)

if __name__ == "__main__":
  unittest.main()
