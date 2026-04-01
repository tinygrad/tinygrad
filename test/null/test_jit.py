import unittest

from tinygrad import Tensor, TinyJit


class TestNullJit(unittest.TestCase):
  def test_jit_can_feed_null_output_back_as_input(self):
    @TinyJit
    def add(a, b): return (a + b).realize()

    a = Tensor.randn(4, 4).contiguous().realize()
    b = Tensor.randn(4, 4).contiguous().realize()

    out = add(a, b)
    out = add(out, b)
    out = add(out, b)

    self.assertIsNotNone(add.captured)
    self.assertGreaterEqual(len(add.captured.jit_cache), 1)
    self.assertEqual(out.shape, (4, 4))


if __name__ == "__main__":
  unittest.main()
