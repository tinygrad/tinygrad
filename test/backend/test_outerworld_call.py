import unittest
from tinygrad import Tensor

class TestOuterCall(unittest.TestCase):
  def test_outer_call_assign(self):
    a = Tensor.zeros(10,10).contiguous()
    b = Tensor.ones(10,10).contiguous()
    Tensor.realize(a,b)

    pa = a.as_param(0)
    pb = b.as_param(1)
    out = Tensor.call(a, b, fxn=pa.assign(pa+pb))
    out.realize()

    print(a.numpy())
    assert (a == 1).all().item()

if __name__ == '__main__':
  unittest.main()
