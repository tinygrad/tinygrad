import unittest
import numpy as np
from tinygrad import Tensor, GlobalCounters, dtypes

class TestGetitemOps(unittest.TestCase):
  def test_empty_tensor_setitem(self):
    for index_shape in [(0,), (2, 0)]:
      x = Tensor.arange(6).reshape(2, 3).contiguous()
      index = Tensor.empty(index_shape, dtype=dtypes.int32)
      x[index] = -1
      x[:, index] = Tensor.empty(2, *index_shape, dtype=x.dtype)
      self.assertEqual(x.tolist(), [[0, 1, 2], [3, 4, 5]])

  def test_two_tensor_indices(self):
    # linear indexing is O(idx_size), one-hot masks is O(idx_size * src_size)
    src_np = np.random.rand(10, 100, 200).astype(np.float32)
    idx1_np, idx2_np = np.random.randint(0, 100, (50, 60), dtype=np.int32), np.random.randint(0, 200, (50, 60), dtype=np.int32)
    src, idx1, idx2 = Tensor(src_np), Tensor(idx1_np), Tensor(idx2_np)
    # O(50*60) = 3K vs O(50*60*100*200) = 60M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[0, idx1_np, idx2_np], src[0, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 100_000)
    # consecutive indices not starting from dim 0: O(10*50*60) = 30K vs O(10*50*60*100*200) = 600M
    GlobalCounters.reset()
    np.testing.assert_allclose(src_np[:, idx1_np, idx2_np], src[:, idx1, idx2].numpy())
    self.assertLess(GlobalCounters.global_ops, 1_000_000)

if __name__ == '__main__':
  unittest.main()
