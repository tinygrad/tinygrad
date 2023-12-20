import unittest
from tinygrad.helpers import prod
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, sym_infer

def st_equal(st1, st2) -> bool:
  if st1.shape != st2.shape: return False
  if st1 == st2: return True
  idx = Variable("idx", 0, prod(st1.shape)-1)
  st1_idx, st1_valid = st1.expr_node(idx)
  st2_idx, st2_valid = st2.expr_node(idx)
  for i in range(idx.min, idx.max):
    st1_off = sym_infer(st1_idx, {idx: i})
    st2_off = sym_infer(st2_idx, {idx: i})
    st1_v = sym_infer(st1_valid, {idx: i})
    st2_v = sym_infer(st2_valid, {idx: i})
    if st1_v != st2_v or (st1_off != st2_off and st1_v): return False
  return True

class TestShapeTrackerBasics(unittest.TestCase):
  def test_pad_shrink_removes_mask(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.pad(((0,2), (0,2)))
    a = a.shrink(((0,10), (0,10)))
    assert len(a.views) == 1 and a.views[-1].mask is None

  def test_pad_shrink_leaves_mask(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.pad(((0,2), (0,2)))
    a = a.shrink(((0,10), (0,11)))
    assert len(a.views) == 1 and a.views[-1].mask is not None

  def test_reshape_makes_same(self):
    a = ShapeTracker.from_shape((2, 5))
    x = a.pad( ((2, 0), (0, 0)) )
    x = x.reshape( (2, 2, 5) )
    x1 = x.reshape( (4, 5) )
    x1 = x1.reshape( (2, 2, 5) )
    assert x == x1.simplify()

class TestShapeTrackerAdd(unittest.TestCase):
  def test_simple_add_reshape(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.reshape((100,))
    b = ShapeTracker.from_shape((100,))
    assert a+b == b

  def test_simple_add_permute(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.permute((1,0))
    b = ShapeTracker.from_shape((10, 10))
    b = b.permute((1,0))
    assert a+b == ShapeTracker.from_shape((10, 10))

class TestShapeTrackerInvert(unittest.TestCase):
  def test_invert_reshape(self):
    a = ShapeTracker.from_shape((10, 10))
    x = a.reshape((5, 20))
    ap = ShapeTracker.from_shape(x.shape) + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_invert_permute(self):
    a = ShapeTracker.from_shape((5, 20))
    x = a.permute((1,0))
    ap = x + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_invert_permute_3(self):
    a = ShapeTracker.from_shape((8, 4, 5))
    x = a.permute((1,2,0))
    ap = x + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_invert_real1(self):
    a = ShapeTracker.from_shape((3, 6, 10))
    x = a.reshape( (3, 3, 2, 10) )
    x = x.permute( (2, 1, 3, 0) )
    ap = x + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_cant_invert_expand(self):
    a = ShapeTracker.from_shape((10, 1))
    x = a.expand((10,10))
    assert x.invert(a.shape) is None

  def test_cant_invert_shrink(self):
    a = ShapeTracker.from_shape((10, 10))
    x = a.shrink(((0,10),(2,8)))
    assert x.invert(a.shape) is None

  def test_can_invert_flip(self):
    a = ShapeTracker.from_shape((20, 10))
    x = a.stride((-1,1))
    ap = x + x.invert(a.shape)
    assert st_equal(ap, a)

  def test_can_invert_flip_permute(self):
    a = ShapeTracker.from_shape((20, 10))
    x = a.permute((1,0))
    x = x.stride((-1,1))
    ap = x + x.invert(a.shape)
    assert st_equal(ap, a)

  def test_cant_invert_stride(self):
    a = ShapeTracker.from_shape((10, 10))
    x = a.stride((2,2))
    assert x.invert(a.shape) is None

  def test_invert_failure(self):
    a = ShapeTracker.from_shape((2, 5))
    x = a.pad( ((2, 0), (0, 0)) )
    x = x.reshape( (2, 2, 5) )
    x = x.reshape( (4, 5) )
    ap = x + x.invert(a.shape)
    assert st_equal(ap, a)

if __name__ == '__main__':
  unittest.main()

