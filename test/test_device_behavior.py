import unittest

from tinygrad import Tensor


class TestDeviceBehavior(unittest.TestCase):
  def test_should_pass(self):
    queue = Tensor.zeros((3, 1)).contiguous()
    buffered = Tensor([2, 1]).unsqueeze(-1)
    replace = buffered.shape[0]
    queue[replace:] = queue[:-replace]
    queue[:replace] = buffered
    buffered = Tensor([3]).unsqueeze(-1)
    replace = buffered.shape[0]
    queue[replace:] = queue[:-replace]
    queue[:replace] = buffered
    res = queue.tolist()
    assert res == [[3.0], [2.0], [1.0]], res
