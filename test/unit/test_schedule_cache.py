import gc, unittest
from tinygrad import Tensor, Variable, Device
from tinygrad.engine.schedule import schedule_cache, complete_create_schedule_with_vars
from tinygrad.uop.ops import Ops, UOp, buffers as uop_buffers

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_reuses_cache(self):
    schedule_cache.clear()
    v = Variable('v', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    # first run with v=5
    t1 = (x + Tensor(v.bind(5))).sum()
    self.assertEqual(t1.item(), 60.0)
    cache_size_after_first = len(schedule_cache)

    # second run with v=10 should reuse cache
    t2 = (x + Tensor(v.bind(10))).sum()
    self.assertEqual(t2.item(), 110.0)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_bound_variable_var_vals(self):
    v = Variable('pos', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    t = x + Tensor(v.bind(42))
    _, var_vals = t.schedule_with_vars()
    self.assertEqual(var_vals, {'pos': 42})

  def test_simple(self):
    a = Tensor.ones(10).contiguous()
    b = Tensor.ones(10).contiguous()
    Tensor.realize(a, b)

    # warm up
    for _ in range(2):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)

    # confirm schedule cache doesn't grow
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)

@unittest.skipIf(Device.DEFAULT in ("CL", "CUDA"), "no multi device in CI for CL/CUDA")
class TestScheduleCacheMultiDevice(unittest.TestCase):
  def setUp(self):
    schedule_cache.clear()
    self.d0 = f"{Device.DEFAULT}:0"
    self.d1 = f"{Device.DEFAULT}:1"

  def test_after_in_tensor_map_cache_miss(self):
    """Test that AFTERs are correctly included in tensor_map on cache miss."""
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((self.d0, self.d1), 0)
    Y = (X + 1).sum()

    # Get the tensor_map from schedule
    big_sink = UOp.sink(Y.uop)
    becomes_map, sched, var_vals = complete_create_schedule_with_vars(big_sink)

    # Find AFTERs in becomes_map
    after_entries = {k: v for k, v in becomes_map.items() if k.op is Ops.AFTER}
    # Multi-device ops should create AFTERs that map to buffers
    self.assertGreater(len(after_entries), 0, "tensor_map should contain AFTER entries")

    # Each AFTER should map to a buffer-related UOp
    for after_uop, buf_uop in after_entries.items():
      self.assertEqual(after_uop.op, Ops.AFTER)
      # buf_uop should be a BUFFER or related type
      self.assertIn(buf_uop.op, {Ops.BUFFER, Ops.MSTACK, Ops.MSELECT})

  def test_after_cache_hit_consistency(self):
    """Test that cache hit produces the same AFTER mappings as cache miss."""
    # First run - cache miss
    X1 = Tensor.ones(256).contiguous().realize()
    X1.shard_((self.d0, self.d1), 0)
    Y1 = (X1 + 1).sum()
    big_sink1 = UOp.sink(Y1.uop)
    becomes_map1, sched1, _ = complete_create_schedule_with_vars(big_sink1)
    afters1 = [k for k in becomes_map1 if k.op is Ops.AFTER]

    cache_size_after_first = len(schedule_cache)
    self.assertGreater(cache_size_after_first, 0)

    # Second run - should hit cache
    X2 = Tensor.ones(256).contiguous().realize()
    X2.shard_((self.d0, self.d1), 0)
    Y2 = (X2 + 1).sum()
    big_sink2 = UOp.sink(Y2.uop)
    becomes_map2, sched2, _ = complete_create_schedule_with_vars(big_sink2)
    afters2 = [k for k in becomes_map2 if k.op is Ops.AFTER]

    # Cache should have been hit (no growth)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

    # Both should have AFTERs
    self.assertEqual(len(afters1), len(afters2), "Cache hit should have same number of AFTER ops")

  def test_multi_device_schedule_and_realize(self):
    """Test that multi-device schedules work correctly with cache."""
    # First run
    X1 = Tensor.ones(256).contiguous().realize()
    X1.shard_((self.d0, self.d1), 0)
    Y1 = (X1 + 1).sum()
    sched1 = Y1.schedule()
    Y1.realize()

    # Second run - cache hit
    X2 = Tensor.ones(256).contiguous().realize()
    X2.shard_((self.d0, self.d1), 0)
    Y2 = (X2 + 1).sum()
    sched2 = Y2.schedule()
    Y2.realize()

    # Both schedules should have the same structure
    self.assertEqual(len(sched1), len(sched2))

  def test_schedule_cache_does_not_keep_buffers_alive(self):
    """Test that the schedule cache doesn't keep actual Buffer objects alive."""
    # Get baseline buffer count
    gc.collect()
    baseline_buffers = len(uop_buffers)

    # Create and schedule tensors
    X = Tensor.ones(256).contiguous().realize()
    X.shard_((self.d0, self.d1), 0)
    Y = (X + 1).sum()
    Y.realize()

    # Cache should have entries
    self.assertGreater(len(schedule_cache), 0)
    gc.collect()

    # Delete all tensor references
    del X, Y
    gc.collect()

    # Buffers should be freed even though cache still has entries
    buffers_after_cleanup = len(uop_buffers)
    self.assertGreater(len(schedule_cache), 0, "cache should still have entries")
    self.assertLessEqual(buffers_after_cleanup, baseline_buffers, "buffers should be freed after tensor cleanup")

    # Verify no BUFFER UOps in cache map to actual Buffers
    for key, (pre_sched, combined_sink) in schedule_cache.items():
      for u in combined_sink.toposort():
        if u.op is Ops.BUFFER:
          self.assertIsNone(uop_buffers.get(u), "cache BUFFER UOp should not map to actual Buffer")

if __name__ == "__main__":
  unittest.main()
