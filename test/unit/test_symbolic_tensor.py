import unittest
from tinygrad import Variable
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import _broadcast_shape

class TestBroadcastShape(unittest.TestCase):
  def test_symbolic(self):
    v = Variable("v", 1, 10)
    self.assertEqual(_broadcast_shape((v,), (1,)), (v,))
    self.assertEqual(_broadcast_shape((v,), ()), (v,))
    self.assertEqual(_broadcast_shape((v,), (v,)), (v,))
    with self.assertRaises(IndexError): _broadcast_shape((v,), (5,))

  def test_symbolic_vmin_zero(self):
    # a symbolic dim that may be 0 still broadcasts against 1 to itself
    v0 = Variable("v0", 0, 10)
    self.assertEqual(_broadcast_shape((v0,), (1,)), (v0,))
    self.assertEqual(_broadcast_shape((v0,), ()), (v0,))
    self.assertEqual(_broadcast_shape((3, v0), (3, 1)), (3, v0))
    with self.assertRaises(IndexError): _broadcast_shape((v0,), (5,))

class TestSymbolicPad(unittest.TestCase):
  def test_pad(self):
    v = Variable("v", 1, 100).bind(5)
    t = Tensor.ones(100)[:v].pad(((4, 0),))
    t = t[:9]
    assert t.tolist() == [0,0,0,0,1,1,1,1,1]

if __name__ == '__main__':
  unittest.main()
