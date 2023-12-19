import unittest
from tinygrad.shape.shapetracker import ShapeTracker

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

  @unittest.skip("reshape is broken")
  def test_reshape_makes_same(self):
    a = ShapeTracker.from_shape((2, 5))
    x = a.pad( ((2, 0), (0, 0)) )
    x = x.reshape( (2, 2, 5) )
    x1 = x.reshape( (4, 5) )
    x1 = x1.reshape( (2, 2, 5) )
    assert x == x1

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
    ap = ShapeTracker.from_shape(x.shape) + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_invert_permute_3(self):
    a = ShapeTracker.from_shape((8, 4, 5))
    x = a.permute((1,2,0))
    ap = ShapeTracker.from_shape(x.shape) + x.invert(a.shape)
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
    a = ShapeTracker.from_shape((10, 10))
    x = a.stride((-1,1))
    ap = ShapeTracker.from_shape(x.shape) + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

  def test_cant_invert_stride(self):
    a = ShapeTracker.from_shape((10, 10))
    x = a.stride((2,2))
    assert x.invert(a.shape) is None

  @unittest.skip("reshape is broken")
  def test_invert_failure(self):
    a = ShapeTracker.from_shape((2, 5))
    x = a.pad( ((2, 0), (0, 0)) )
    x = x.reshape( (2, 2, 5) )
    x = x.reshape( (4, 5) )
    ap = ShapeTracker.from_shape(x.shape) + x.invert(a.shape)
    assert ap == a, f"{ap} != {a}"

if __name__ == '__main__':
  unittest.main()

