import unittest
from tinygrad import Tensor
from tinygrad.dtype import Invalid

class TestInvalidTensor(unittest.TestCase):
  def test_where_x_invalid(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid)
    ret = out.tolist()
    assert ret[0] == 1.0 and ret[1] == 2.0

  def test_where_invalid_x(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Invalid, Tensor([1.0, 2.0, 3.0, 4.0]))
    ret = out.tolist()
    assert ret[2] == 3.0 and ret[3] == 4.0

  def test_where_invalid_2d(self):
    mask = Tensor.arange(6).reshape(2, 3) < 3
    vals = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = mask.where(vals, Invalid)
    ret = out.tolist()
    assert ret[0] == [1.0, 2.0, 3.0]

  def test_where_invalid_int(self):
    mask = Tensor.arange(3) < 2
    out = mask.where(Tensor([10, 20, 30]), Invalid)
    ret = out.tolist()
    assert ret[0] == 10 and ret[1] == 20

  def test_where_invalid_add(self):
    mask = Tensor.arange(3) < 2
    mixed = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    out = mixed + Tensor([1.0, 2.0, 3.0])
    ret = out.tolist()
    assert ret[0] == 11.0 and ret[1] == 22.0

  def test_where_always_true(self):
    mask = Tensor.arange(3) < 10
    out = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    ret = out.tolist()
    assert ret == [10.0, 20.0, 30.0]

if __name__ == '__main__':
  unittest.main()
