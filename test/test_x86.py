import unittest
from tinygrad import Tensor, dtypes, Device
import numpy as np

Device.DEFAULT = "X86"
class TestX86(unittest.TestCase):
  def test_add(self):
    a = Tensor([1,2,3])
    b = Tensor([1,2,3])
    assert (a+b).tolist() == [2,4,6]

  def test_sub(self):
    a = Tensor([2,2,3])
    b = Tensor([1,2,3])
    assert (a-b).tolist() == [1, 0, 0]

  def test_equal(self):
    a = Tensor([2,2,3])
    b = Tensor([1,2,3])
    assert (a==b).tolist() == [False, True, True]
    # floats
    a = Tensor([2,2,3], dtype=dtypes.float32)
    b = Tensor([1,2,3], dtype=dtypes.float32)
    assert (a==b).tolist() == [False, True, True]
    # with nan
    a = Tensor([2,2,float("nan")], dtype=dtypes.float32)
    b = Tensor([1,2,float("nan")], dtype=dtypes.float32)
    assert (a==b).tolist() == [False, True, False]

  def test_greater(self):
    a = Tensor([2,2,3])
    b = Tensor([1,2,3])
    assert (a>b).tolist() == [True, False, False]
    # floats
    a = Tensor([2,2,3], dtype=dtypes.float32)
    b = Tensor([1,2,3], dtype=dtypes.float32)
    assert (a>b).tolist() == [True, False, False]
    # with nan
    a = Tensor([2,2,3], dtype=dtypes.float32)
    b = Tensor([1,2,float("nan")], dtype=dtypes.float32)
    assert (a>b).tolist() == [True, False, False]

  def test_mul(self):
    a = Tensor([2,2,3])
    b = Tensor([1,2,3])
    assert (a*b).tolist() == [2, 4, 9]

  def test_div(self):
    a = Tensor([2,2,3])
    b = Tensor([1,2,3])
    assert (a//b).tolist() == [2, 1, 1]

  def test_and(self):
    a = Tensor([2,2,3])
    b = Tensor([1,2,3])
    assert (a&b).tolist() == [0, 2, 3]

  def test_or(self):
    a = Tensor([2,2,3])
    b = Tensor([1,2,3])
    assert (a|b).tolist() == [3, 2, 3]

  def test_max(self):
    a = Tensor([1,2,3])
    assert (a.max()).tolist() == 3

  def test_min(self):
    a = Tensor([1,2,3])
    assert (a.min()).tolist() == 1

  def test_where(self):
    a = Tensor([2,2,2])
    b = Tensor([1,1,1])
    c = Tensor([False, True, False])
    assert (c.where(a,b)).tolist() == [1, 2, 1]

  def test_cast(self):
    a = Tensor([2,2,3], dtype=dtypes.int32)
    assert a.cast(dtypes.float32).tolist() == [2., 2., 3.]
    # floats
    a = Tensor([2,2,3], dtype=dtypes.float32)
    assert a.cast(dtypes.int32).tolist() == [2, 2, 3]
    # with nans
    a = Tensor([-2,2,float("nan")], dtype=dtypes.float64)
    assert a.cast(dtypes.int32).tolist() == [-2, 2, -2147483648]
    # to bool
    a = Tensor([-2,0,float("nan")], dtype=dtypes.float64)
    assert a.cast(dtypes.bool).tolist() == [True, False, True]
    # signed to unsigned
    a = Tensor([-1,0,1], dtype=dtypes.int32)
    assert a.cast(dtypes.uint64).tolist() == [18446744073709551615, 0, 1]
    # uint32 to uint64
    a = Tensor([1,2,3000000000], dtype=dtypes.uint32)
    assert a.cast(dtypes.uint64).tolist() == [1, 2, 3000000000]

  def test_bitcast(self):
    a = Tensor([-1.,1.,2.123], dtype=dtypes.float32)
    assert a.bitcast(dtypes.int32).tolist() == [-1082130432, 1065353216, 1074257723]

  def test_cumsum(self):
    assert Tensor([1,2,3]).cumsum().tolist() == [1, 3, 6]

  def test_sin(self):
    a = Tensor([1,2,3]).sin().tolist()
    b = Tensor([1,2,3], device="CLANG").sin().tolist()
    assert np.allclose(a, b, rtol=1e-6)

  def test_cos(self):
    a = Tensor([1,2,3]).cos().tolist()
    b = Tensor([1,2,3], device="CLANG").cos().tolist()
    assert np.allclose(a, b, rtol=1e-6)

  def test_tan(self):
    a = Tensor([1,2,3]).tan().tolist()
    b = Tensor([1,2,3], device="CLANG").tan().tolist()
    assert np.allclose(a, b, rtol=1e-6)

  def test_rand(self):
    a = Tensor.rand(3).tolist()

  def test_randint(self):
    a = Tensor.randint(3).tolist()
  
  def test_log(self):
    a = Tensor([1,2,3]).log().tolist()
    b = Tensor([1,2,3], device="CLANG").log().tolist()
    assert np.allclose(a, b, rtol=1e-7)
  
  def test_exp(self):
    a = Tensor([1,2,3]).exp().tolist()
    b = Tensor([1,2,3], device="CLANG").exp().tolist()
    assert np.allclose(a, b, rtol=1e-7)

  def test_softmax(self):
    a = Tensor([1,2,3]).softmax().tolist()
    b = Tensor([1,2,3], device="CLANG").softmax().tolist()
    assert np.allclose(a, b, rtol=1e-7)

  def test_sparse_categorical_crossentropy(self):
    a = Tensor([[-1, 2, -3], [1, -2, 3]], device="X86").sparse_categorical_crossentropy(Tensor([1, 2], device="X86")).tolist()
    b = Tensor([[-1, 2, -3], [1, -2, 3]], device="CLANG").sparse_categorical_crossentropy(Tensor([1, 2], device="CLANG")).tolist()
    assert np.allclose(a, b, rtol=1e-7)

  def test_logcumsumexp(self):
    # vectorize 2
    a = Tensor([1,2,3]).logcumsumexp().tolist()
    b = Tensor([1,2,3], device="CLANG").logcumsumexp().tolist()
    assert np.allclose(a, b, rtol=1e-7)
    # vectorize 4
    a = Tensor([1,2,3,4]).logcumsumexp().tolist()
    b = Tensor([1,2,3,4], device="CLANG").logcumsumexp().tolist()
    assert np.allclose(a, b, rtol=1e-7)

if __name__ == '__main__':
  unittest.main()
