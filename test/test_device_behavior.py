import unittest

from tinygrad import Tensor

class TestDeviceBehavior(unittest.TestCase):
  def test_should_pass(self):
    t = Tensor([[3.0], [2.0], [1.0]]).contiguous()
    t[1:] = t[:-1]
    assert t.tolist() == [[3.0], [3.0], [2.0]]
