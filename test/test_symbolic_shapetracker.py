import unittest
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters

class TestSymbolic(unittest.TestCase):
  def setUp(self): GlobalCounters.var_vals = {}
  def tearDown(self): GlobalCounters.var_vals = {}

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
  def setUp(self): GlobalCounters.var_vals = {}
  def tearDown(self): GlobalCounters.var_vals = {}

  def test_reshape_into_symbols_simple(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10)
      assert Tensor.rand(i, 4).reshape(vi, 4).shape == (vi, 4)
      assert GlobalCounters.var_vals[vi] == i
      vi = Variable("i", 1, 10)
      assert Tensor.rand(i, 6).reshape(vi, 2, 3).shape == (vi, 2, 3)
      assert GlobalCounters.var_vals[vi] == i

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
      assert GlobalCounters.var_vals[vi] == i

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

  def test_two_symbol_reshape(self):
    for i in range(1, 6):
      for j in range(1, 6):
        vi = Variable("i", 1, 5)
        vj = Variable("j", 1, 5)
        t1 = Tensor.rand(i, 5).reshape(vi, 5)
        t2 = Tensor.rand(5, j).reshape(5, vj)
        t = t1@t2
        assert t.shape == (vi, vj)
        t = t.reshape(1, vi*vj)
        assert t.shape == (1, vi*vj)
        t = t.reshape(vj, vi)
        assert t.shape == (vj, vi)

class TestSymbolicExpand(unittest.TestCase):
  def setUp(self): GlobalCounters.var_vals = {}
  def tearDown(self): GlobalCounters.var_vals = {}

  def test_expand_into_symbols(self):
    for i in range(1, 6):
      for j in range(1, 6):
        vi = Variable("i", 1, 5)
        vj = Variable("j", 1, 5)
        # this sets values for vi, vj, which is required if we want to expand into it
        Tensor.rand(i).reshape(vi)
        Tensor.rand(j).reshape(vj)

        a = Tensor([[1], [2], [3]]).expand((3, vi))
        assert a.shape == (3, vi)
        a = a.reshape(3, vi, 1).expand((3, vi, vj))
        assert a.shape == (3, vi, vj)

  def test_plus_expands_constant(self):
    for i in range(1, 6):
      vi = Variable("i", 1, 5)
      a = Tensor.rand(3, i).reshape(3, vi)
      a = a + 1
      assert a.shape == (3, vi)

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
    assert idx.render() == "(((1+i)*1)+(lidx1*((i*4)+4))+gidx0)"

class TestSymbolicSameVariables(unittest.TestCase):
  def setUp(self): GlobalCounters.var_vals = {}
  def tearDown(self): GlobalCounters.var_vals = {}

  def test_same_var_same_value_work(self):
    for i in range(1, 6):
      vi = Variable("i", 1, 5)
      a = Tensor.rand(3, i).reshape(3, vi)
      b = Tensor.rand(i, 5).reshape(vi, 5)
      c = a@b
      assert c.shape == (3, 5)

  def test_same_name_different_var_same_value_work(self):
    vi1 = Variable("i", 1, 10)
    vi2 = Variable("i", 1, 10)
    a = Tensor.rand(3, 4).reshape(3, vi1)
    b = Tensor.rand(4, 5).reshape(vi2, 5)
    c = a@b
    assert c.shape == (3, 5)

  def test_same_name_different_var_different_value_overwrite(self):
    vi1 = Variable("i", 1, 10)
    a = Tensor.rand(3, 4).reshape(3, vi1)
    assert GlobalCounters.var_vals[vi1] == 4
    vi2 = Variable("i", 1, 10)
    b = Tensor.rand(4, 5).reshape(4, vi2)
    assert GlobalCounters.var_vals[vi1] == 5
    assert GlobalCounters.var_vals[vi2] == 5

  def test_different_var_same_value_fail(self):
    a = Tensor.rand(3, 4).reshape(3, Variable("i", 1, 10))
    b = Tensor.rand(4, 5).reshape(Variable("j", 1, 10), 5)
    with self.assertRaises(AssertionError):
      c = a@b

  def test_same_var_different_value_fail(self):
    vi = Variable("i", 1, 10)
    a = Tensor.rand(3, 4).reshape(3, vi)
    with self.assertRaises(AssertionError):
      b = Tensor.rand(7, 5).reshape(vi, 5)

if __name__ == '__main__':
  unittest.main()