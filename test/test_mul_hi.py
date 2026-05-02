import unittest, numpy as np
from tinygrad import Tensor, dtypes


def ref_mul_hi(x:int, y:int, dtype) -> np.integer:
  return dtype.type((int(x)*int(y)) >> (dtype.itemsize*8))


class TestMulHi(unittest.TestCase):
  def test_mul_hi_vs_numpy_reference(self):
    rng = np.random.default_rng(0)
    cases = [(2**40, 2**40), (2**62, 2), (-(2**40), 2**40), ((1<<63)-1, (1<<63)-1),
             (-(1<<63), -(1<<63)), (-1, -1), (12345, 67890), (0, (1<<63)-1)]
    cases += [(int(rng.integers(-(1<<62), 1<<62)), int(rng.integers(-(1<<62), 1<<62))) for _ in range(16)]
    a = Tensor([x for x,_ in cases], dtype=dtypes.int64)
    b = Tensor([y for _,y in cases], dtype=dtypes.int64)
    expected = np.array([ref_mul_hi(x, y, np.dtype(np.int64)) for x,y in cases], dtype=np.int64)
    np.testing.assert_array_equal(a.mul_hi(b).numpy(), expected)

  def test_mul_hi_int32_vs_numpy_reference(self):
    rng = np.random.default_rng(1)
    cases = [(2**30, 2), (-(2**30), 3), ((1<<31)-1, (1<<31)-1), (-(1<<31), -(1<<31)), (-1, -1), (12345, 67890)]
    cases += [(int(rng.integers(-(1<<31), 1<<31)), int(rng.integers(-(1<<31), 1<<31))) for _ in range(16)]
    a = Tensor([x for x,_ in cases], dtype=dtypes.int32)
    b = Tensor([y for _,y in cases], dtype=dtypes.int32)
    expected = np.array([ref_mul_hi(x, y, np.dtype(np.int32)) for x,y in cases], dtype=np.int32)
    np.testing.assert_array_equal(a.mul_hi(b).numpy(), expected)

  def test_mul_hi_broadcasts(self):
    a_vals = [[-(1<<63)], [(1<<62)]]
    b_vals = [[2, -3]]
    a = Tensor(a_vals, dtype=dtypes.int64)
    b = Tensor(b_vals, dtype=dtypes.int64)
    expected = np.array([[ref_mul_hi(x, y, np.dtype(np.int64)) for y in b_vals[0]] for x, in a_vals], dtype=np.int64)
    np.testing.assert_array_equal(a.mul_hi(b).numpy(), expected)

  def test_mul_hi_rejects_non_int32_int64(self):
    for dt in (dtypes.float32, dtypes.int16):
      with self.assertRaisesRegex(RuntimeError, "int32 or int64"):
        Tensor([1, 2], dtype=dt).mul_hi(Tensor([3, 4], dtype=dt))
    with self.assertRaisesRegex(RuntimeError, "matching"):
      Tensor([1, 2], dtype=dtypes.int32).mul_hi(Tensor([3, 4], dtype=dtypes.int64))


if __name__ == "__main__":
  unittest.main()
