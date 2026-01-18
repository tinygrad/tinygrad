import unittest
from tinygrad import Tensor, GlobalCounters

class TestGetitemOps(unittest.TestCase):
  def test_two_tensor_indices(self):
    # linear indexing is O(idx_size), one-hot masks is O(idx_size * src_size)
    src = Tensor.rand(10, 100, 200).realize()
    idx1 = Tensor.randint(50, 60, low=0, high=100).realize()
    idx2 = Tensor.randint(50, 60, low=0, high=200).realize()
    # O(50*60) = 3K vs O(50*60*100*200) = 60M
    GlobalCounters.reset()
    src[0, idx1, idx2].realize()
    self.assertLess(GlobalCounters.global_ops, 50_000)
    # consecutive indices not starting from dim 0: O(10*50*60) = 30K vs O(10*50*60*100*200) = 600M
    GlobalCounters.reset()
    src[:, idx1, idx2].realize()
    self.assertLess(GlobalCounters.global_ops, 500_000)

if __name__ == '__main__':
  unittest.main()
