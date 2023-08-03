import os, unittest
from tinygrad.helpers import CI
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor, Device

@unittest.skipUnless(Device.DEFAULT == "CLANG", "CLANG only")
class TestSymbolicOps(unittest.TestCase):
  def test_plus1(self):
    os.environ["TEST_COMPILE_ONLY"] = "1"
    vi = Variable("i", 1, 10)
    a = Tensor.rand(3, 4).reshape(vi, 4)
    s = a + 1
    if not CI:
      s.realize()

if __name__ == '__main__':
  unittest.main()