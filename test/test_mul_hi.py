import unittest, numpy as np
from tinygrad import Tensor, dtypes


def ref_mul_hi(x:int, y:int) -> np.int64:
  return np.int64((int(x)*int(y)) >> 64)


class TestMulHi(unittest.TestCase):
  def test_mul_hi_vs_numpy_reference(self):
    rng = np.random.default_rng(0)
    cases = [(2**40, 2**40), (2**62, 2), (-(2**40), 2**40), ((1<<63)-1, (1<<63)-1),
             (-(1<<63), -(1<<63)), (-1, -1), (12345, 67890), (0, (1<<63)-1)]
    cases += [(int(rng.integers(-(1<<62), 1<<62)), int(rng.integers(-(1<<62), 1<<62))) for _ in range(16)]
    a = Tensor([x for x,_ in cases], dtype=dtypes.int64)
    b = Tensor([y for _,y in cases], dtype=dtypes.int64)
    expected = np.array([ref_mul_hi(x, y) for x,y in cases], dtype=np.int64)
    np.testing.assert_array_equal(a.mul_hi(b).numpy(), expected)

  def test_mul_hi_broadcasts(self):
    a_vals = [[-(1<<63)], [(1<<62)]]
    b_vals = [[2, -3]]
    a = Tensor(a_vals, dtype=dtypes.int64)
    b = Tensor(b_vals, dtype=dtypes.int64)
    expected = np.array([[ref_mul_hi(x, y) for y in b_vals[0]] for x, in a_vals], dtype=np.int64)
    np.testing.assert_array_equal(a.mul_hi(b).numpy(), expected)

  def test_mul_hi_rejects_non_int64(self):
    for dt in (dtypes.float32, dtypes.int32):
      with self.assertRaisesRegex(RuntimeError, "int64"):
        Tensor([1, 2], dtype=dt).mul_hi(Tensor([3, 4], dtype=dt))


if __name__ == "__main__":
  unittest.main()
