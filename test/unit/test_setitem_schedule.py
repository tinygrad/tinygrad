import unittest
from tinygrad import Tensor, dtypes, GlobalCounters

class TestSetitemInto(unittest.TestCase):
  def test_setitem_into_unrealized(self):
    GlobalCounters.reset()
    t = Tensor.arange(4, dtype=dtypes.int32).reshape(2, 2)
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(GlobalCounters.global_mem, 4*4+4*2)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertListEqual(t.tolist(), [[0, 1], [5, 5]])

  def test_setitem_into_unrealized_sliced_compute(self):
    # base computation contains SHRINK from prior slicing (like QR decomposition pattern)
    GlobalCounters.reset()
    a = Tensor.arange(8, dtype=dtypes.int32).reshape(2, 4)
    w = a[0] + a[1]  # unrealized ADD with SHRINK in graph: [4, 6, 8, 10]
    self.assertEqual(GlobalCounters.kernel_count, 0)
    w[1] = 99
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(GlobalCounters.global_mem, 4*4+4)
    self.assertListEqual(w.tolist(), [4, 99, 8, 10])

  def test_setitem_into_empty(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32)
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_empty_alu(self):
    GlobalCounters.reset()
    t = Tensor.empty(4, dtype=dtypes.int32) + 1
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(GlobalCounters.global_mem, 4*4*2+4)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(t[1].item(), 5)

  def test_setitem_into_tensor(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize()
    GlobalCounters.reset()
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1].realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertEqual(GlobalCounters.global_mem, 4)
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 1)
    self.assertListEqual(t.tolist(), [1, 5, 3, 4])

  def test_setitem_into_tensor_alu(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.int32).realize() + 1
    GlobalCounters.reset()
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(GlobalCounters.global_mem, 4*4*2+4)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertListEqual(t.tolist(), [2, 5, 4, 5])

  def test_setitem_into_cont(self):
    t = Tensor.ones(4, dtype=dtypes.int32)
    with self.assertRaises(RuntimeError): t[1] = 5

  def test_setitem_into_const_alu(self):
    # TODO: this is not consistent
    GlobalCounters.reset()
    t = Tensor.ones(4, dtype=dtypes.int32) + 1
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(GlobalCounters.global_mem, 4*4+4)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertListEqual(t.tolist(), [2, 5, 2, 2])

    t = Tensor.ones(4, dtype=dtypes.int32) + 1
    t.realize()
    with self.assertRaises(RuntimeError): t[1] = 5

  def test_setitem_into_arange(self):
    # NOTE: arange has no real buffer, but assigning to it is fine
    GlobalCounters.reset()
    t = Tensor.arange(4, dtype=dtypes.int32)
    self.assertEqual(GlobalCounters.kernel_count, 0)
    t[1] = 5
    self.assertEqual(GlobalCounters.kernel_count, 2)
    t[1].realize()
    t.realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertListEqual(t.tolist(), [0, 5, 2, 3])

if __name__ == '__main__':
  unittest.main()
