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
    i = Variable("i", 1, 5)
    j = Variable("j", 1, 5)
    k = Variable("k", 1, 5)
    t1 = Tensor.rand(3, 4).reshape(3, i).cat(Tensor.rand(3, 4).reshape(3, j), dim=1).cat(Tensor.rand(3, 4).reshape(3, k), dim=1)
    st = t1.lazydata.st
    assert st.shape == (3, i+j+k)
    assert st.real_strides() == (i+j+k, 1)

class TestSymbolicReshape(unittest.TestCase):
  def test_reshape_into_symbols_simple(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10)
      assert Tensor.rand(i, 4).reshape(vi, 4).shape == (vi, 4)
      assert vi.val == i
      vi = Variable("i", 1, 10)
      assert Tensor.rand(i, 6).reshape(vi, 2, 3).shape == (vi, 2, 3)
      assert vi.val == i

  def test_reshape_symbols_reshape_ints(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10)
      assert Tensor.rand(i, 4).reshape(vi, 4).reshape(i, 4).shape == (i, 4)
      assert Tensor.rand(i, 4).reshape(vi, 4).reshape(i*4,).shape == (i*4,)
      assert Tensor.rand(i, 6).reshape(vi, 6).reshape(i*2, 3).shape == (i*2, 3)
      with self.assertRaises(AssertionError):
        Tensor.rand(i, 6).reshape(vi, 6).reshape(1, 77).shape

  def test_reshape_reuse_var_same_value_ok(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10)
      a = Tensor.rand(i, 4).reshape(vi, 4)
      b = Tensor.rand(i, 3).reshape(vi, 3)
      assert vi.val == i

  def test_reshape_reuse_var_different_value_fail(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10)
      a = Tensor.rand(i, 4).reshape(vi, 2)
      with self.assertRaises(AssertionError):
        b = Tensor.rand(i, 3).reshape(vi, 3)

  def test_reshape_into_symbols_bad_shape(self):
    vi = Variable("i", 1, 10)
    vj = Variable("j", 1, 10)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(3, 4).reshape(vi, vj)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(4, 4).reshape(vi, vi)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(4, 6).reshape(vi, 6).reshape(vi, 4)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(100, 4).reshape(Variable("too_small", 1, 10), 4)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(3, 4).reshape(Variable("too_big", 100, 200), 4)

class TestSymbolicExpand(unittest.TestCase):
  def test_expand_into_symbols(self):
    vi = Variable("i", 1, 10)
    a = Tensor([[1], [2], [3]]).expand((3, vi))
    assert a.shape == (3, vi)
    vj = Variable("j", 1, 10)
    a = a.reshape(3, vi, 1).expand((3, vi, vj))
    assert a.shape == (3, vi, vj)

  def test_plus_expands_constant(self):
    vi = Variable("i", 1, 10)
    a = Tensor.rand(3, 4).reshape(3, vi)
    a = a + 1
    assert a.shape == (3, vi)