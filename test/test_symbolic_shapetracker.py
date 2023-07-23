import unittest
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor

class TestSymbolic(unittest.TestCase):
  def test_symbolic_st(self):
    x = Variable("x", 1, 100)
    st = ShapeTracker((x, 3))
    assert st.shape == (x, 3)
    assert st.real_strides() == (3, 1)

  def test_expr_idxs(self):
    x = Variable("x", 1, 100)
    st = ShapeTracker((x, 3))
    idxs = [Variable("x", 0, 100), Variable("y", 0, 100)]
    e1, e2 = st.expr_idxs(idxs)
    assert e1.render() == "((x*3)+y)"
    assert e2.render() == "1"
    st.permute((1, 0))
    e1, e2 = st.expr_idxs(idxs)
    assert e1.render() == "((y*3)+x)"
    assert e2.render() == "1"

  def test_cat_strides(self):
    i = Variable("i", 1, 5)
    j = Variable("j", 1, 5)
    k = Variable("k", 1, 5)
    t1 = Tensor.rand(3, 4).reshape(i, 4).cat(Tensor.rand(3, 4).reshape(j, 4), dim=0).cat(Tensor.rand(3, 4).reshape(k, 4), dim=0)
    st = t1.lazydata.st
    assert st.shape == (i+j+k, 4)
    assert st.real_strides() == (4, 1)
    t1 = Tensor.rand(3, 4).reshape(3, i).cat(Tensor.rand(3, 4).reshape(3, j), dim=1).cat(Tensor.rand(3, 4).reshape(3, k), dim=1)
    st = t1.lazydata.st
    assert st.shape == (3, i+j+k)
    assert st.real_strides() == (i+j+k, 1)