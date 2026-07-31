import os, unittest
from unittest.mock import patch
from tinygrad import Tensor, Variable, Context, UOp
from tinygrad.callify import transform_to_call
from tinygrad.helpers import cpu_events
from tinygrad.schedule import lower_sink_to_linear, schedule_cache

def schedule_one():
  (Tensor.empty(1) + 1).schedule_linear()

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

  def test_disk_schedule_cache(self):
    function = transform_to_call(UOp.sink((Tensor.empty(1) + 1).uop))[0].src[0]
    schedule_cache.clear()
    with patch.dict(os.environ, {"DISK_SCACHE":"1"}), \
         patch("tinygrad.schedule.diskcache_get", return_value=None), \
         patch("tinygrad.schedule.diskcache_put") as cache_put:
      lower_sink_to_linear(function)
      cached = cache_put.call_args.args[2]

    schedule_cache.clear()
    with patch.dict(os.environ, {"DISK_SCACHE":"1"}), \
         patch("tinygrad.schedule.diskcache_get", return_value=cached) as cache_get, \
         patch("tinygrad.schedule.diskcache_put") as cache_put:
      self.assertIs(lower_sink_to_linear(function), cached)
      cache_get.assert_called_once()
      cache_put.assert_not_called()

if __name__ == "__main__":
  unittest.main()
