import unittest
import numpy as np
from tinygrad import Device, Tensor, Context, dtypes, TinyJit

@unittest.skipUnless(Device.DEFAULT == "MALI", "MALI device required")
class TestMali(unittest.TestCase):
  def test_raw_compute_repeat_and_input(self):
    x = Tensor([1., 2., 3., 4.], device="MALI")
    for _ in range(5): self.assertEqual((x * 2).tolist(), [2., 4., 6., 8.])
    self.assertEqual((x + 1).tolist(), [2., 3., 4., 5.])

  def test_tinyjit_repeat_and_input(self):
    @TinyJit
    def f(x): return (x * 2).realize()
    x = Tensor([1., 2., 3., 4.], device="MALI")
    for _ in range(3): self.assertEqual(f(x).tolist(), [2., 4., 6., 8.])
    self.assertEqual(f(x + 1).tolist(), [4., 6., 8., 10.])

  def test_multiple_descriptor_sets(self):
    xs = [Tensor.full((8,), float(i - 16), device="MALI") for i in range(34)]
    expected = np.full(8, sum(i - 16 for i in range(34)), dtype=np.float32)
    np.testing.assert_array_equal(sum(xs[1:], xs[0]).numpy(), expected)

  def test_explicit_fp16_image_repeat_and_input(self):
    with Context(IMAGE=1, FLOAT16=1):
      x = Tensor.arange(16, dtype=dtypes.float).reshape(1, 4, 2, 2).contiguous()
      w = Tensor.ones((4, 4, 1, 1)).contiguous()
      expected = np.sum(x.numpy().astype(np.float32), axis=1, keepdims=True).repeat(4, axis=1)
      first = x.conv2d(w).numpy()
      np.testing.assert_array_equal(first, expected)
      np.testing.assert_array_equal(x.conv2d(w).numpy(), first)
      changed = (x + 1).conv2d(w).numpy()
      np.testing.assert_array_equal(changed, expected + 4)
      self.assertFalse(np.array_equal(changed, first))

if __name__ == '__main__': unittest.main()
