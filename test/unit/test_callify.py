import unittest
from tinygrad import Tensor

@unittest.skip("WIP")
class TestCallify(unittest.TestCase):
  def test_callify_twice(self):
    a = Tensor.ones(32, 32)
    b = Tensor.eye(32)
    out = (a@b).callify()
    out = (out+4).callify()
    out.realize()

if __name__ == '__main__':
  unittest.main()
