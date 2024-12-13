#!/usr/bin/env python
import unittest
from tinygrad.shape.view import View, merge_dims

class TestView(unittest.TestCase):
  def test_canonicalize_empty_mask(self):
    v = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    assert v.mask is None
    v = View.create(shape=(4,3,2), strides=(1,4,10), mask=((0,4),(0,3),(0,2)))
    assert v.mask is None

  def test_minify_zero_strided_dims(self):
    target = View.create(shape=(2,2), strides=(30,2), offset=7, mask=None)
    v = View.create(shape=(2,1,2), strides=(30,0,2), offset=7, mask=None)
    self.assertEqual(v.minify(), target)
    v = View.create(shape=(1,2,2), strides=(0,30,2), offset=7, mask=None)
    self.assertEqual(v.minify(), target)
    v = View.create(shape=(2,2,1), strides=(30,2,0), offset=7, mask=None)
    self.assertEqual(v.minify(), target)
    v = View.create(shape=(2,1,1,2), strides=(30,0,0,2), offset=7, mask=None)
    self.assertEqual(v.minify(), target)
    v = View.create(shape=(1,1,2,2), strides=(0,0,30,2), offset=7, mask=None)
    self.assertEqual(v.minify(), target)
    v = View.create(shape=(2,2,1,1), strides=(30,2,0,0), offset=7, mask=None)
    self.assertEqual(v.minify(), target)
    v = View.create(shape=(1,2,2,1), strides=(0,30,2,0), offset=7, mask=None)
    self.assertEqual(v.minify(), target)
    v = View.create(shape=(1,2,1,2), strides=(0,30,0,2), offset=7, mask=None)
    self.assertEqual(v.minify(), target)

  def test_empty_mask_contiguous(self):
    v1 = View.create(shape=(2,2,2), strides=(4,2,1), mask=None)
    v2 = View.create(shape=(2,2,2), strides=(4,2,1), mask=((0,2),(0,2),(0,2)))
    self.assertEqual(v1.contiguous, v2.contiguous)
    v1 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=None)
    v2 = View.create(shape=(1,1,1,4), strides=(0,0,0,1), offset=0, mask=((0,1),(0,1),(0,1),(0,4)))
    self.assertEqual(v1.contiguous, v2.contiguous)
    v = View.create(shape=(2,3,4), mask=((0,2),(0,3),(0,4)))
    self.assertTrue(v.contiguous)

  def test_merge_view_pairs(self):
    v1 = View(shape=(2, 4), strides=(2, 1), offset=-2, mask=((0, 2), (2, 4)), contiguous=False)
    v2 = View(shape=(2, 4, 2, 2), strides=(4, 0, -2, -1), offset=3, mask=None, contiguous=False)
    ans = View(shape=(2, 4, 2, 2), strides=(2, 0, 0, -1), offset=1, mask=((0, 2), (0, 4), (0, 1), (0, 2)), contiguous=False)
    self.assertEqual(v1 + v2, ans)

    v1 = View(shape=(5, 10, 12), strides=(100, 1, 10), offset=-20, mask=((0, 5), (0, 10), (2, 12)), contiguous=False)
    v2 = View(shape=(10, 6, 5, 2, 2), strides=(12, 2, 120, 1, 0), offset=0, mask=None, contiguous=False)
    ans = View(shape=(10, 6, 5, 2, 2), strides=(1, 20, 100, 10, 0), offset=-20, mask=((0, 10), (1, 6), (0, 5), (0, 2), (0, 2)), contiguous=False)
    self.assertEqual(v1 + v2, ans)

    v1 = View(shape=(8, 7, 3), strides=(1, 12, -4), offset=6, mask=((2, 6), (0, 7), (0, 3)), contiguous=False)
    v2 = View(shape=(4, 2, 6, 2, 1), strides=(42, 21, 3, 1, 0), offset=4, mask=None, contiguous=False)
    ans = View(shape=(4, 2, 6, 2, 1), strides=(2, 1, 12, -4, 0), offset=14, mask=((1, 3), (0, 2), (0, 6), (0, 2), (0, 1)), contiguous=False)
    self.assertEqual(v1 + v2, ans)

    v1 = View(shape=(7, 21, 3), strides=(54, 3, 1), offset=-9, mask=((0, 6), (3, 21), (0, 3)), contiguous=False)
    v2 = View(shape=(5, 3, 3, 7), strides=(63, 1, 3, 9), offset=63, mask=None, contiguous=False)
    ans = View(shape=(5, 3, 3, 7), strides=(54, 1, 3, 9), offset=45, mask=((0, 5), (0, 3), (0, 3), (1, 7)), contiguous=False)
    self.assertEqual(v1 + v2, ans)

    v1 = View(shape=(5, 1, 24), strides=(20, 0, 1), offset=-2, mask=((0, 5), (0, 1), (2, 22)), contiguous=False)
    v2 = View(shape=(12, 2, 5, 2, 1), strides=(2, 1, 24, 0, 0), offset=0, mask=None, contiguous=False)
    ans = View(shape=(12, 2, 5, 2, 1), strides=(2, 1, 20, 0, 0), offset=-2, mask=((1, 11), (0, 2), (0, 5), (0, 2), (0, 1)), contiguous=False)
    self.assertEqual(v1 + v2, ans)

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
    self.assertEqual(merge_dims(shape, (3, 1, 0)), ((6, 1, 6), (4, 0, 0)))
    self.assertEqual(merge_dims(shape, (0, 0, 0)), ((24, 0, 0),))

  def test_pad_reshape(self):
    # st = ShapeTracker.from_shape((1, 2)).pad(((1, 0), (0, 1))).reshape((3, 2))
    self.assertEqual(merge_dims((2, 3), (0, 1), ((1, 2), (0, 2))), ((6, 1, 3),))
    # shift mask on stride 0
    self.assertEqual(merge_dims((2, 3), (0, 1), ((0, 1), (0, 2))), ((6, 1, 3),))

if __name__ == '__main__':
  unittest.main()
