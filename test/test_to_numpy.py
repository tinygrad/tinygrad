from tinygrad.tensor import Tensor, dtypes
import numpy as np
import pickle
import unittest

class TestToNumpy(unittest.TestCase):
  def test_numpy_is_numpy(self):
    output = Tensor.ones((1, 3, 4096)).realize().numpy()
    new = np.copy(output)
    print(type(new))
    serialized = pickle.dumps(new)
    out = pickle.loads(serialized)
    assert out.shape == (1,3,4096)
    assert (out==1).all()

  def test_to_numpy_mv(self):
    shape = (36, 12345)
    tens = Tensor.rand(shape, dtype=dtypes.float32)
    np_holder = np.zeros(shape, dtype=np.float32)
    tens.to_numpy_mv(np_holder.data)
    np.testing.assert_equal(tens.numpy(), np_holder)

  def test_fp16_to_numpy_mv(self):
    shape = (36, 12345)
    tens = Tensor.rand(shape, dtype=dtypes.float16)
    np_holder = np.zeros(shape, dtype=np.float16)
    tens.to_numpy_mv(np_holder.data)
    np.testing.assert_equal(tens.numpy(), np_holder)

  def test_ones_to_numpy_mv(self):
    shape = (36, 12345)
    tens = Tensor.ones(shape, dtype=dtypes.float32)
    np_holder = np.zeros(shape, dtype=np.float32)
    tens.to_numpy_mv(np_holder.data)
    np.testing.assert_equal(tens.numpy(), np_holder)

if __name__ == '__main__':
  unittest.main()