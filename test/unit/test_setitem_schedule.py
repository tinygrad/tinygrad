import unittest
from tinygrad import Tensor, Device, TinyJit, Variable, dtypes, GlobalCounters
from test.helpers import assert_kernel_count

class TestSetitemInto(unittest.TestCase):
  def test_setitem_into_unrealized(self):
    GlobalCounters.reset()
    t = Tensor.arange(4, dtype=dtypes.int32).reshape(2, 2)
    assert_kernel_count(0)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [[0, 1], [5, 5]])

  def test_setitem_into_unrealized_sliced_compute(self):
    # base computation contains SHRINK from prior slicing (like QR decomposition pattern)
    GlobalCounters.reset()
    a = Tensor.arange(8, dtype=dtypes.int32).reshape(2, 4)
    w = a[0] + a[1]  # unrealized ADD with SHRINK in graph: [4, 6, 8, 10]
    assert_kernel_count(0)
    w[1] = 99
    assert_kernel_count(0)
    w.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(w.tolist(), [4, 99, 8, 10])

  def test_setitem_into_empty(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t[1].realize()
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_empty_alu(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32) + 1
    assert_kernel_count(0)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertLessEqual(GlobalCounters.global_mem, 32)
    t[1].realize()
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_tensor(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize()
    GlobalCounters.reset()
    t[1] = 5
    assert_kernel_count(0)
    t[1].realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t.realize()
    assert_kernel_count(1)
    self.assertListEqual(t.tolist(), [1, 5, 3, 4])

  def test_setitem_into_tensor_alu(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize() + 1
    GlobalCounters.reset()
    t[1] = 5
    assert_kernel_count(0)
    t[1].realize()
    assert_kernel_count(1)
    self.assertLessEqual(GlobalCounters.global_mem, 32)
    t[1].realize()
    t.realize()
    assert_kernel_count(1)
    self.assertListEqual(t.tolist(), [2, 5, 4, 5])

  def test_setitem_into_const(self):
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32, buffer=False)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [1, 5, 1, 1])

  def test_setitem_into_const_alu(self):
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32, buffer=False) + 1
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertEqual(GlobalCounters.global_mem, 0)
    self.assertListEqual(t.tolist(), [2, 5, 2, 2])

  def test_setitem_into_arange(self):
    # NOTE: arange has no real buffer, but assigning to it is fine
    GlobalCounters.reset()
    other = Tensor.arange(4, dtype=dtypes.int32)
    t = Tensor.arange(4, dtype=dtypes.int32)
    self.assertIs(other.uop, t.uop)
    t[1] = 5
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(0)
    self.assertListEqual(t.tolist(), [0, 5, 2, 3])

  def test_setitem_slice_const(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = 3
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 30*4)  # 30 elements written

  def test_setitem_slice_tensor(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    v = Tensor.zeros(30, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[20:50] = v
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 30*4*2)  # 30 read + 30 written

  def test_setitem_full(self):
    t = Tensor.zeros(100, dtype=dtypes.int32).contiguous().realize()
    GlobalCounters.reset()
    t[:] = 3
    assert_kernel_count(0)
    t.realize()
    assert_kernel_count(1)
    self.assertEqual(GlobalCounters.global_mem, 100*4)  # full buffer written

  def test_setitem_slice_assign_from_other_device(self):
    for device in ("CPU:1", Device.DEFAULT):
      if device == "CPU": continue
      with self.subTest(device=device):
        a = Tensor.ones(20, device="CPU")
        b = Tensor.arange(20).float().clone(device=device)
        Tensor.realize(a, b)
        GlobalCounters.reset()
        a[10:12].assign(b[13:15].to(a.device)).realize()
        # creation copies retain their own storage; backends without views materialize the source slice
        assert_kernel_count(2 if device.split(":")[0] in {"PYTHON", "NPY", "CL", "WEBGPU"} else 1)
        self.assertListEqual(a.tolist(), [1.0]*10 + [13.0, 14.0] + [1.0]*8)

class TestSlicedCopy(unittest.TestCase):
  def test_contiguous(self):
    for size in (1, 2, 8, 20):
      for dst_offset in (0, 20-size):
        for src_offset in (0, 20-size):
          with self.subTest(size=size, dst_offset=dst_offset, src_offset=src_offset):
            a = Tensor([-1]*20, device="CPU").realize()
            b = Tensor(list(range(20)), device="CPU:1").realize()
            GlobalCounters.reset()
            a[dst_offset:dst_offset+size].assign(b[src_offset:src_offset+size].to(a.device)).realize()
            assert_kernel_count(1)
            self.assertEqual(GlobalCounters.global_mem, size * a.dtype.itemsize)
            self.assertListEqual(a.tolist(), [-1]*dst_offset + list(range(src_offset, src_offset+size)) + [-1]*(20-dst_offset-size))

  def test_noncontiguous_destination(self):
    a = Tensor([-1]*20, device="CPU").realize()
    b = Tensor(list(range(10)), device="CPU:1").realize()
    a[::2].assign(b.to(a.device)).realize()
    self.assertListEqual(a.tolist(), [x for i in range(10) for x in (i, -1)])

  def test_noncontiguous_source(self):
    a = Tensor([-1]*20, device="CPU").realize()
    b = Tensor(list(range(20)), device="CPU:1").realize()
    a[5:15].assign(b[::2].to(a.device)).realize()
    self.assertListEqual(a.tolist(), [-1]*5 + list(range(0, 20, 2)) + [-1]*5)

  def test_reshaped_destination(self):
    a = Tensor([-1]*20, device="CPU").reshape(4, 5).realize()
    b = Tensor(list(range(20)), device="CPU:1").reshape(4, 5).realize()
    GlobalCounters.reset()
    a[1:3].assign(b[2:4].to(a.device)).realize()
    assert_kernel_count(1)
    self.assertListEqual(a.flatten().tolist(), [-1]*5 + list(range(10, 20)) + [-1]*5)

  def test_symbolic_destination(self):
    for size in (1, 3, 10):
      a = Tensor([-1]*20, device="CPU").realize()
      b = Tensor(list(range(20)), device="CPU:1").realize()
      v = Variable("size", 1, 10).bind(size)
      a[5:5+v].assign(b[:v].to(a.device)).realize()
      self.assertListEqual(a.tolist(), [-1]*5 + list(range(size)) + [-1]*(15-size))

  def test_source_write_ordering(self):
    a = Tensor([-1]*20, device="CPU").realize()
    b = Tensor(list(range(20)), device="CPU:1").realize()
    a[10:12].assign(b[13:15].to(a.device))
    b.assign(b + 100)
    Tensor.realize(a, b)
    self.assertListEqual(a.tolist(), [-1]*10 + [13, 14] + [-1]*8)
    self.assertListEqual(b.tolist(), list(range(100, 120)))

  def test_pending_destination_source_write_ordering(self):
    a = Tensor([-1]*20, device="CPU").realize()
    b = Tensor(list(range(20)), device="CPU:1").realize()
    a.assign(a + 9)
    a[3:7].assign(b[5:9].to(a.device))
    b.assign(b + 200)
    Tensor.realize(a, b)
    self.assertListEqual(a.tolist(), [8]*3 + [5, 6, 7, 8] + [8]*13)
    self.assertListEqual(b.tolist(), list(range(200, 220)))

  def test_bitcast_source_write_ordering(self):
    a = Tensor([-1.0]*20, device="CPU").realize()
    b = Tensor.arange(20).float().clone(device="CPU:1").bitcast(dtypes.int32).contiguous().realize()
    a[3:7].assign(b.bitcast(dtypes.float32)[5:9].to(a.device))
    b.assign(b + 1)
    Tensor.realize(a, b)
    self.assertListEqual(a.tolist(), [-1.0]*3 + [5.0, 6.0, 7.0, 8.0] + [-1.0]*13)

  def test_jit_new_buffers(self):
    @TinyJit
    def copy_slice(a, b):
      return a[10:12].assign(b[13:15].to(a.device)).realize()
    for i in range(5):
      a = Tensor([-1]*20, device="CPU").realize()
      b = Tensor([x+i for x in range(20)], device="CPU:1").realize()
      copy_slice(a, b)
      self.assertListEqual(a.tolist(), [-1]*10 + [13+i, 14+i] + [-1]*8)

if __name__ == '__main__':
  unittest.main()
