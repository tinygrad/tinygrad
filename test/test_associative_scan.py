import unittest
import numpy as np
from tinygrad import Tensor, associative_scan

class TestAssociativeScan(unittest.TestCase):
  def test_add(self):
    for size in (0, 1, 2, 5, 8):
      x = Tensor.arange(size)
      np.testing.assert_equal(associative_scan(lambda a,b: a+b, x, 0).numpy(), np.arange(size).cumsum())

  def test_axis_and_reverse(self):
    x = Tensor.arange(24).reshape(2, 3, 4)
    np.testing.assert_equal(associative_scan(lambda a,b: a+b, x, -1).numpy(), np.arange(24).reshape(2, 3, 4).cumsum(-1))
    np.testing.assert_equal(associative_scan(lambda a,b: a+b, x, 1, reverse=True).numpy(),
                            np.flip(np.flip(np.arange(24).reshape(2, 3, 4), 1).cumsum(1), 1))

  def test_noncommutative(self):
    rng = np.random.default_rng(0)
    mats = rng.normal(size=(7, 2, 2)).astype(np.float32)
    expected = [mats[0]]
    for mat in mats[1:]: expected.append(expected[-1] @ mat)
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(mats.tolist()), 0, combine_mode="generic").numpy(), expected, atol=1e-5)

    reverse_expected = [mats[-1]]
    for mat in mats[-2::-1]: reverse_expected.append(reverse_expected[-1] @ mat)
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(mats.tolist()), 0, reverse=True,
                                                combine_mode="generic").numpy(), reverse_expected[::-1], atol=1e-5)

    transposed = np.moveaxis(mats, 0, -1)
    np.testing.assert_allclose(associative_scan(lambda a,b: a@b, Tensor(transposed.tolist()), -1,
                                                combine_mode="generic").numpy(), np.moveaxis(expected, 0, -1), atol=1e-5)

  def test_pytree(self):
    a = Tensor([0.8, 0.6, 0.7, 0.5])
    b = Tensor([1.0, 2.0, 3.0, 4.0])
    def compose(x, y): return {"a":x["a"]*y["a"], "b":[y["a"]*x["b"][0]+y["b"][0]]}
    out = associative_scan(compose, {"a":a, "b":[b]}, 0)
    expected_a, expected_b = [a.numpy()[0]], [b.numpy()[0]]
    for ai,bi in zip(a.numpy()[1:], b.numpy()[1:]):
      expected_a.append(expected_a[-1]*ai)
      expected_b.append(ai*expected_b[-1]+bi)
    np.testing.assert_allclose(out["a"].numpy(), expected_a)
    np.testing.assert_allclose(out["b"][0].numpy(), expected_b)

  def test_gradient(self):
    x = Tensor.arange(6, dtype="float32")
    grad, = associative_scan(lambda a,b: a+b, x, 0).sum().gradient(x)
    np.testing.assert_equal(grad.numpy(), np.arange(6, 0, -1))

  def test_errors(self):
    with self.assertRaisesRegex(ValueError, "same scan length"):
      associative_scan(lambda a,b: (a[0]+b[0], a[1]+b[1]), (Tensor.ones(2), Tensor.ones(3)), 0)
    with self.assertRaisesRegex(ValueError, "combine_mode"):
      associative_scan(lambda a,b: a+b, Tensor.ones(2), 0, combine_mode="invalid")
    with self.assertRaisesRegex(ValueError, "metadata"):
      associative_scan(lambda a,b: a.sum(0), Tensor.ones(3, 2), 0)

if __name__ == "__main__": unittest.main()
