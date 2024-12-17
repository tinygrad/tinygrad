#!/usr/bin/env python
import unittest
from tinygrad.shape.view import View, merge_dims
# from tinygrad.shape.shapetracker import ShapeTracker

class TestView(unittest.TestCase):
  def test_canonicalize_empty_mask(self):
    v = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    assert v.mask is None
    v = View.create(shape=(4,3,2), strides=(1,4,10), mask=((0,4),(0,3),(0,2)))
    assert v.mask is None

  def test_minify_zero_strided_dims(self):
    target = View.create(shape=(2,2), strides=(30,2), offset=7, mask=None)
    v = View.create(shape=(2,1,2), strides=(30,0,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,2,2), strides=(0,30,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(2,2,1), strides=(30,2,0), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(2,1,1,2), strides=(30,0,0,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,1,2,2), strides=(0,0,30,2), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(2,2,1,1), strides=(30,2,0,0), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,2,2,1), strides=(0,30,2,0), offset=7, mask=None)
    assert v.minify() == target
    v = View.create(shape=(1,2,1,2), strides=(0,30,0,2), offset=7, mask=None)
    assert v.minify() == target

  def test_empty_mask_contiguous(self):
    v1 = View.create(shape=(2,2,2), strides=(4,2,1), mask=None)
    v2 = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    assert v1.contiguous == v2.contiguous
    v1 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=None)
    v2 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=((0,1),(0,1),(0,1),(0,4)))
    assert v1.contiguous == v2.contiguous
    v = View.create(shape=(2,3,4), mask=((0,2),(0,3),(0,4)))
    assert v.contiguous

class TestMergeDims(unittest.TestCase):
  def test_contiguous(self):
    shape = (2, 3, 4)
    strides = (12, 4, 1) #=strides_for_shape(shape)
    m = merge_dims(shape, strides)
    self.assertEqual(m, ((24, 1, 24),))

  def test_0_in_strides(self):
    shape = (2, 3, 4)
    self.assertEqual(merge_dims(shape, (0, 4, 1)), ((2, 0, 0), (12, 1, 12)))
    self.assertEqual(merge_dims(shape, (0, 0, 1)), ((6, 0, 0), (4, 1, 4)))
    self.assertEqual(merge_dims(shape, (3, 1, 0)), ((6, 1, 6), (4, 0, 4)))
    self.assertEqual(merge_dims(shape, (0, 0, 0)), ((24, 0, 0),))

  def test_pad(self):
    # print(ShapeTracker.from_shape((1, 2)).pad(((1, 0), (0, 1))).views[-1])
    self.assertEqual(merge_dims((2, 3), (0, 1), ((1, 2), (0, 2))), ((6, 1, 3),))

    # print(f"{ShapeTracker.from_shape((1, 1, 2)).pad(((1, 0), (1, 0), (0, 1))).views[-1]}")
    self.assertEqual(merge_dims((2, 2, 3), (0, 0, 1), ((1, 2), (1, 2), (0, 2))), ((12, 1, 3),))

    # print(f"{ShapeTracker.from_shape((1, 1, 2, 2)).pad(((1, 0), (1, 0), (0, 1), (0, 1))).views[-1]}")
    self.assertEqual(merge_dims((2, 2, 3, 3), (0, 0, 2, 1), ((1, 2), (1, 2), (0, 2), (0, 2))), ((12, 2, 3), (3, 1, 3)))

    # print(f"{ShapeTracker.from_shape((2, 1, 2)).pad(((0, 0), (1, 0), (0, 1))).views[-1]}")
    self.assertEqual(merge_dims((2, 2, 3), (2, 0, 1), ((0, 2), (1, 2), (0, 2))), ((2, 2, 2), (6, 1, 3)))

  def test_different_1_pad(self):
    # print(f"{ShapeTracker.from_shape((2, 2, 1)).pad(((0, 0), (0, 0), (0, 1))).views[-1]}")
    self.assertEqual(merge_dims((2, 2, 2), (2, 1, 0), ((0, 2), (0, 2), (0, 1))), ((4, 1, 4), (2, 0, 2)))

    # print(f"{ShapeTracker.from_shape((2, 1, 1)).pad(((0, 0), (0, 1), (0, 1))).views[-1]}")
    self.assertEqual(merge_dims((2, 2, 2), (1, 0, 0), ((0, 2), (0, 2), (0, 1))), ((2, 1, 2), (4, 0, 4)))

class TestMergeViews(unittest.TestCase):
  def test_with_mask_0(self):
    # from test/test_ops.py::TestOps::test_pad_reflect_mode
    v0 = View(shape=(1, 1, 5, 8), strides=(0, 0, 5, 1), offset=-3, mask=((0, 1), (0, 1), (0, 5), (3, 8)), contiguous=False)
    v1 = View(shape=(1, 1, 2, 2), strides=(0, 0, 8, 1), offset=3, mask=None, contiguous=False)
    v = v0 + v1
    self.assertIsNotNone(v)
    self.assertEqual(v, View(shape=(1, 1, 2, 2), strides=(0, 0, 5, 1), offset=0, mask=None, contiguous=False))

  def test_with_mask_1(self):
    # from test/test_ops.py::TestOps::test_pad_reflect_mode
    v0 = View(shape=(3, 3, 5, 3), strides=(27, 9, 3, 1), offset=-6, mask=((0, 3), (0, 3), (2, 4), (1, 3)), contiguous=False)
    v1 = View(shape=(3, 3, 2, 2), strides=(45, 15, 3, 1), offset=7, mask=None, contiguous=False)
    v = v0 + v1
    self.assertIsNotNone(v)
    self.assertEqual(v, View(shape=(3, 3, 2, 2), strides=(27, 9, 3, 1), offset=1, mask=None, contiguous=False))

  def test_with_mask_2(self):
    # from test/test_ops.py::TestOps::test_pad_reflect_mode
    v0 = View(shape=(3, 3, 5, 3), strides=(27, 9, -3, 1), offset=6, mask=((0, 3), (0, 3), (0, 2), (0, 2)), contiguous=False)
    v1 = View(shape=(3, 3, 2, 2), strides=(45, 15, -3, 1), offset=3, mask=None, contiguous=False)
    v = v0 + v1
    self.assertIsNotNone(v)
    self.assertEqual(v, View(shape=(3, 3, 2, 2), strides=(27, 9, 3, 1), offset=3, mask=None, contiguous=False))

if __name__ == '__main__':
  unittest.main()
