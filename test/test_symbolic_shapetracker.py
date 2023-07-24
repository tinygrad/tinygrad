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

class TestSymbolicReshape(unittest.TestCase):
  def test_reshape_into_symbols(self):
    for i in range(1, 10):
      vi = Variable("i", 1, 10)
      t = Tensor.rand(i, 4).reshape(vi, 4)
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(i, 4).reshape(vi, 4)
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(i, 6).reshape(vi, 2, 3)
      assert t.shape == (vi, 2, 3)
      assert t.inferred_shape == (i, 2, 3)
      assert t.lazydata.symbols == {vi: i}

  def test_reshape_symbols_reshape_ints(self):
    for i in range(1, 10):
      vi = Variable("i", 1, 10)
      t = Tensor.rand(i, 4).reshape(vi, 4).reshape(i, 4)
      assert t.shape == (i, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {}
      t = Tensor.rand(i, 4).reshape(vi, 4).reshape(i*4,)
      assert t.shape == (i*4,)
      assert t.inferred_shape == (i*4,)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(i, 6).reshape(vi, 6).reshape(i*2, 3)
      assert t.shape == (i*2, 3)
      assert t.inferred_shape == (i*2, 3)
      assert t.lazydata.symbols == {}

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

  def test_full_like(self):
    for i in range(1, 10):
      vi = Variable("i", 1, 10)
      t = Tensor.full_like(Tensor.rand(i, 4).reshape(vi, 4), 5)
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}

  def test_reshape_move_ops(self):
    for i in range(1, 10):
      vi = Variable("i", 1, 10)
      t = Tensor.rand(i, 4).reshape(vi, 4).permute((1, 0))
      assert t.shape == (4, vi)
      assert t.inferred_shape == (4, i)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(i, 4).reshape(vi, 4).reshape(vi*4,)
      assert t.shape == (vi*4,)
      assert t.inferred_shape == (i*4,)
      assert t.lazydata.symbols == {vi: i}

  def test_reshape_unary_ops(self):
    for i in range(1, 10):
      vi = Variable("i", 1, 10)
      t = Tensor.rand(i, 4).reshape(vi, 4).contiguous()
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(i, 4).reshape(vi, 4).log()
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(i, 4).reshape(vi, 4).relu()
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}

  def test_reshape_binary_ops(self):
    for i in range(1, 10):
      vi = Variable("i", 1, 10)
      t = Tensor.rand(i, 4).reshape(vi, 4) + Tensor.rand(i, 4).reshape(vi, 4)
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(i, 4).reshape(vi, 4) * Tensor.rand(i, 4).reshape(vi, 4)
      assert t.shape == (vi, 4)
      assert t.inferred_shape == (i, 4)
      assert t.lazydata.symbols == {vi: i}
      t = Tensor.rand(4, i).reshape(4, vi) @ Tensor.rand(i, 4).reshape(vi, 4)
      assert t.shape == (4, 4)
      assert t.inferred_shape == (4, 4)
      assert t.lazydata.symbols == {}

  def test_reshape_reduce_ops(self):
    for i in range(1, 10):
      vi = Variable("i", 1, 10)
      t = Tensor.rand(i, 4).reshape(vi, 4).sum(axis=0)
      assert t.shape == (4,)
      assert t.inferred_shape == (4,)
      assert t.lazydata.symbols == {}
      t = Tensor.rand(i, 4).reshape(vi, 4).sum(axis=1)
      assert t.shape == (vi,)
      assert t.inferred_shape == (i,)
      assert t.lazydata.symbols == {vi: i}

  def test_reshape_reduce_ops_2d(self):
    for i in range(1, 10):
      for j in range(1, 10):
        vi = Variable("i", 1, 10)
        vj = Variable("j", 1, 10)
        a = Tensor.rand(i, 3).reshape(vi, 3) @ Tensor.rand(3, j).reshape(3, vj)
        assert a.shape == (vi, vj)
        assert a.inferred_shape == (i, j)
        assert a.lazydata.symbols == {vi: i, vj: j}
        r = a.sum(axis=0)
        assert r.shape == (vj,)
        assert r.inferred_shape == (j,)
        assert r.lazydata.symbols == {vj: j}
        r = a.sum(axis=1)
        assert r.shape == (vi,)
        assert r.inferred_shape == (i,)
        assert r.lazydata.symbols == {vi: i}
        r = a.sum()
        assert r.shape == ()
        assert r.inferred_shape == ()
        assert r.lazydata.symbols == {}
