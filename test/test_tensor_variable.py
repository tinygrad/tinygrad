import unittest
from tinygrad import Tensor, Variable

class TestTensorVariable(unittest.TestCase):
  def test_add_tvar(self):
    vv = Variable("a", 0, 10)
    vv.bind(1)
    ret = (Tensor(vv) + 3).item()
    assert ret == 4

  def test_inner_tvar(self):
    vv = Variable("w", 0, 10)
    vv.bind(2)
    ret = Tensor.from_node(vv * 4).item()
    assert ret == 8

  def test_symbolic_mean(self):
    vv = Variable("a", 1, 10)
    vv.bind(2)
    t = Tensor.ones(2, 2).contiguous().reshape(2, vv)
    ret = t.mean().item()
    assert ret == 1

if __name__ == '__main__':
  unittest.main()
