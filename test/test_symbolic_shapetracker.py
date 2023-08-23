import unittest
from tinygrad.shape.shapetracker import ShapeTracker, View
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
    t = Tensor.rand(3, 4).reshape(i, 4).cat(Tensor.rand(3, 4).reshape(j, 4), dim=0).cat(Tensor.rand(3, 4).reshape(k, 4), dim=0)
    st = t.lazydata.st
    assert st.shape == (i+j+k, 4)
    assert st.real_strides() == (4, 1)
    t = Tensor.rand(3, 4).reshape(3, i).cat(Tensor.rand(3, 4).reshape(3, j), dim=1).cat(Tensor.rand(3, 4).reshape(3, k), dim=1)
    st = t.lazydata.st
    assert st.shape == (3, i+j+k)
    assert st.real_strides() == (i+j+k, 1)
    t = Tensor.rand(i, 3).reshape(i, 3).cat(Tensor.rand(3, 3).reshape(i, 3), dim=0).cat(Tensor.rand(3, 3), dim=0)
    st = t.lazydata.st
    assert st.shape == (2*i+3, 3)
    assert st.real_strides() == (3, 1)

class TestSymbolicReshape(unittest.TestCase):
  def test_reshape_into_symbols_simple(self):
    vi = Variable("i", 1, 5)
    for i in range(1, 6):
      t = Tensor.rand(i, 4).reshape(vi, 4)
      assert t.shape == (vi, 4)
      assert t.lazydata.st.var_vals[vi] == i
      t = Tensor.rand(i, 6).reshape(vi, 2, 3)
      assert t.shape == (vi, 2, 3)
      assert t.lazydata.st.var_vals[vi] == i

  def test_reshape_symbols_reshape_ints(self):
    vi = Variable("i", 1, 5)
    for i in range(1, 6):
      t = Tensor.rand(i, 4).reshape(vi, 4)
      assert t.shape == (vi, 4)
      assert t.lazydata.st.var_vals == {vi: i}
      t = t.reshape(i, 4)
      assert t.shape == (i, 4)
      assert t.lazydata.st.var_vals == {}

  def test_reshape_reuse_var_same_value_ok(self):
    vi = Variable("i", 1, 5)
    for i in range(1, 6):
      a = Tensor.rand(i, 4).reshape(vi, 4)
      b = Tensor.rand(i, 3).reshape(vi, 3)
      assert a.lazydata.st.var_vals[vi] == i
      assert b.lazydata.st.var_vals[vi] == i

  def test_reshape_reuse_var_different_value_ok(self):
    vi = Variable("i", 1, 10)
    for i in range(1, 6):
      a = Tensor.rand(i, 4).reshape(vi, 2)
      b = Tensor.rand(i, 3).reshape(vi, 3)
      # a and b have different values of vi
      assert a.lazydata.st.var_vals[vi] == 2 * i
      assert b.lazydata.st.var_vals[vi] == i

  def test_reshape_into_symbols_bad_shape(self):
    vi = Variable("i", 1, 10)
    vj = Variable("j", 1, 10)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(3, 4).reshape(vi, vj) # reshape into two variables
    with self.assertRaises(AssertionError):
      t = Tensor.rand(4, 4).reshape(vi, vi) # reshape into same variable in 2 dimensions
    with self.assertRaises(AssertionError):
      t = Tensor.rand(4, 6).reshape(vi, 6).reshape(vi, 4) # conflicted implied variable values
    with self.assertRaises(AssertionError):
      t = Tensor.rand(4, 6).reshape(vi, 6).reshape(1, 77) # reshape to a different size new shape through symbolic shape
    with self.assertRaises(AssertionError):
      t = Tensor.rand(100, 4).reshape(Variable("too_small", 1, 10), 4)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(3, 4).reshape(Variable("too_big", 100, 200), 4)
    with self.assertRaises(AssertionError):
      t = Tensor.rand(3, 4).reshape(3, (vi+1)) # reshape into non-Variable Node

  def test_two_symbol_reshape(self):
    vi = Variable("i", 1, 5)
    vj = Variable("j", 1, 5)
    for i in range(1, 6):
      for j in range(1, 6):
        t1 = Tensor.rand(i, 5).reshape(vi, 5)
        t2 = Tensor.rand(5, j).reshape(5, vj)
        t = t1@t2
        assert t.shape == (vi, vj)
        t = t.reshape(1, vi*vj)
        assert t.shape == (1, vi*vj)
        t = t.reshape(vj, vi)
        assert t.shape == (vj, vi)

class TestSymbolicExpand(unittest.TestCase):
  def test_expand_into_symbols(self):
    vi = Variable("i", 1, 5)
    vj = Variable("j", 1, 5)
    a = Tensor([[1], [2], [3]]).expand((3, vi))
    assert a.shape == (3, vi)
    assert a.lazydata.st.var_vals == {}
    a = a.reshape(3, vi, 1).expand((3, vi, vj))
    assert a.shape == (3, vi, vj)
    assert a.lazydata.st.var_vals == {}

  def test_plus_expands_constant(self):
    vi = Variable("i", 1, 5)
    for i in range(1, 6):
      a = Tensor.rand(3, i).reshape(3, vi)
      a = a + 1
      assert a.shape == (3, vi)

class TestSymbolicShrink(unittest.TestCase):
  def test_shrink_symbols(self):
    vi = Variable("i", 1, 5)
    t = Tensor.rand(3, 5).shrink(((0, 2), (vi, vi+1)))
    assert t.shape == (2, 1)

class TestSymbolicShapeExpr(unittest.TestCase):
  def test_symbolic_expr_idxs(self):
    # taken from symbolic shape llama
    i = Variable("i", 1, 120)
    gidx0 = Variable("gidx0", 0, i)
    lidx1 = Variable("lidx1", 0, 7)
    idx = (gidx0, lidx1, Variable.num(1))
    shape = (i+1, 8, 4)
    strides = (1, (i*4)+4, i+1)
    view = View(shape, strides)
    st = ShapeTracker(shape, [view])
    idx, valid = st.expr_idxs(idx)
    assert idx.render() == "((lidx1*((i*4)+4))+1+gidx0+i)"

class TestShapeTrackerVarVals(unittest.TestCase):
  def test_reshape_reshape_updates_var_vals(self):
    vi = Variable("i", 1, 5)
    vj = Variable("j", 1, 5)
    t = Tensor.rand(3, 4).reshape(3, vi).reshape(4, vj)
    assert t.lazydata.st.var_vals == {vi: 4, vj: 3}

  def test_lazy_check_var_vals(self):
    vi = Variable("i", 1, 5)
    a = Tensor.rand(3, 4).reshape(3, vi)
    b = Tensor.rand(5, 6).reshape(vi, 6)
    assert a.lazydata.st.var_vals == {vi: 4}
    assert b.lazydata.st.var_vals == {vi: 5}
    c = a@b
    # shapetracker works with symbolic shape and doesn't check / propagate the underlying variable values
    assert c.shape == (3, 6)
    assert c.lazydata.st.var_vals == {}

if __name__ == '__main__':
  unittest.main()