import unittest

from tinygrad import Tensor


class TestDeviceBehavior(unittest.TestCase):
  def test_should_pass(self):
    queue = Tensor.zeros((3, 1)).contiguous()
    buffered = Tensor.cat(*[Tensor([3]), Tensor([1])]).unsqueeze(-1)
    replace = buffered.shape[0]
    queue[replace:] = queue[:-replace]
    queue[:replace] = buffered
    buffered = Tensor.cat(*[Tensor([4])]).unsqueeze(-1)
    replace = buffered.shape[0]
    queue[replace:] = queue[:-replace]
    queue[:replace] = buffered
    assert (res := queue.tolist()) == [[4.0], [3.0], [1.0]], res
