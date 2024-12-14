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

  def test_pad_reshape(self):
    # st = ShapeTracker.from_shape((1, 2)).pad(((1, 0), (0, 1))).reshape((3, 2))
    self.assertEqual(merge_dims((2, 3), (0, 1), ((1, 2), (0, 2))), ((6, 1, 3),))
    # shift mask on stride 0
    self.assertEqual(merge_dims((2, 3), (0, 1), ((0, 1), (0, 2))), ((6, 1, 3),))
    # permute 0 / 1
    self.assertEqual(merge_dims((3, 2), (1, 0), ((0, 2), (1, 2))), ((3, 1, 3), (2, 0, 2)))

    # st = ShapeTracker.from_shape((1, 1, 2)).pad(((1, 0), (1, 0), (0, 1)))
    # print(f"{st.views[-1]}")
    self.assertEqual(merge_dims((2, 2, 3), (0, 0, 1), ((1, 2), (1, 2), (0, 2))), ((12, 1, 3),))

    # st = ShapeTracker.from_shape((1, 1, 2, 2)).pad(((1, 0), (1, 0), (0, 1), (0, 1)))
    # print(f"{st.views[-1]}")
    self.assertEqual(merge_dims((2, 2, 3, 3), (0, 0, 2, 1), ((1, 2), (1, 2), (0, 2), (0, 2))), ((12, 2, 3), (3, 1, 3)))

    # st = ShapeTracker.from_shape((1, 1, 1, 1)).pad(((0, 2), (0, 0), (0, 1), (0, 3)))
    # print(f"{st.views[-1]}")
    # self.assertEqual(merge_dims((3, 1, 2, 4), (0, 0, 0, 0), ((0, 1), (0, 1), (0, 1), (0, 1))), ((24, 0, 0),))

  def test_different_1_pad(self):
    # st = ShapeTracker.from_shape((2, 2, 1)).pad(((0, 0), (0, 0), (0, 1)))
    # print(f"{st.views[-1]}")
    self.assertEqual(merge_dims((2, 2, 2), (2, 1, 0), ((0, 2), (0, 2), (0, 1))), ((4, 1, 4), (2, 0, 2)))

    # st = ShapeTracker.from_shape((2, 1, 1)).pad(((0, 0), (0, 1), (0, 1)))
    # print(f"{st.views[-1]}")
    self.assertEqual(merge_dims((2, 2, 2), (1, 0, 0), ((0, 2), (0, 2), (0, 1))), ((2, 1, 2), (4, 0, 4)))

if __name__ == '__main__':
  unittest.main()
