import unittest
from tinygrad.tensor import Tensor

class TestMaskedShapeTracker(unittest.TestCase):
  def test_mul_masked(self):
    a = Tensor([1,1,1,1])
    b = Tensor([1,1]).pad(((0,2),))
    c = a*b
    # TODO: make this true
    #assert c.lazydata.st.views[0].mask is not None
    ret = c.data()
    assert ret.tolist() == [1.0, 1.0, 0.0, 0.0]

  def test_add_masked(self):
    a = Tensor([1,1]).pad(((0,2),))
    b = Tensor([1,1]).pad(((0,2),))
    c = a+b
    # TODO: make this true
    #assert c.lazydata.st.views[0].mask is not None
    ret = c.data()
    assert ret.tolist() == [2.0, 2.0, 0.0, 0.0]

if __name__ == '__main__':
  unittest.main()
