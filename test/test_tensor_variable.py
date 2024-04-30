import unittest
from tinygrad import Tensor, Variable

class TestTensorVariable(unittest.TestCase):
  def test_add_tvar(self):
    vv = Variable("a", 0, 10)
    vv.bind(1)
    ret = (Tensor(vv) + 3).item()
    assert ret == 4

if __name__ == '__main__':
  unittest.main()
