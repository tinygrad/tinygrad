import unittest
from tinygrad.shape.shapetracker import ShapeTracker

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
  def test_invert_permute(self):
    a = ShapeTracker.from_shape((10, 10))
    a = a.permute((1,0))
    assert a+a.invert() == ShapeTracker.from_shape((10, 10))

  def test_invert_permute_3(self):
    a = ShapeTracker.from_shape((10, 10, 10))
    a = a.permute((0,2,1))
    assert a+a.invert() == ShapeTracker.from_shape((10, 10, 10))

  def test_cant_permute_expand(self):
    a = ShapeTracker.from_shape((10, 1)).expand((10,10))
    assert a.invert() is None

if __name__ == '__main__':
  unittest.main()

