#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker

class DumbShapeTracker:
  def __init__(self, shape):
    self.t = np.arange(prod(shape), dtype=np.uint8).reshape(shape)

  @property
  def shape(self):
    return self.t.shape

  def reshape(self, *new_shape):
    self.t = self.t.reshape(new_shape)

  def permute(self, *axis):
    self.t = np.transpose(self.t, axis)

  def expand(self, *new_shape):
    self.t = np.broadcast_to(self.t, new_shape)

  def flip(self, *axis):
    self.t = np.flip(self.t, axis)

  def shrink(self, *arg):
    self.t = self.t[tuple([slice(x[0], x[1]) for x in arg])]

  def stride(self, *arg):
    self.t = self.t[tuple([slice(None, None, x) for x in arg])]

  def __getitem__(self, val):
    return self.t.flatten()[val]

# Tensor.zeros(2, 4).permute(1,0).reshape(2, 4)
# (d1*4 + d0%4), d1=x//4, d0=x%4 = ((x//4)*4) + (x%4)%4

class TestComplexShapeTracker(unittest.TestCase):
  def test_add_1s(self):
    self.st = ShapeTracker((4, 4))
    self.st.permute(1,0)
    self.st.reshape(1,4,1,4,1)
    assert not self.st.contiguous
    self.st.permute(0,3,2,1,4)
    assert self.st.contiguous

  def test_permute_1s_simple(self):
    self.st = ShapeTracker((1, 16, 9,9))
    self.st.permute(1,0,2,3)
    assert self.st.contiguous
    self.st = ShapeTracker((2, 16, 9,9))
    self.st.permute(1,0,2,3)
    assert not self.st.contiguous

  def test_remove_1s_simple(self):
    self.st = ShapeTracker((1, 16, 1, 1))
    self.st.reshape(16,)
    assert self.st.contiguous

  def test_remove_1s(self):
    self.st = ShapeTracker((1, 4, 1, 4, 1))
    self.st.permute(0,3,2,1,4)
    self.st.reshape(4,4)
    assert not self.st.contiguous
    self.st.permute(1,0)
    assert self.st.contiguous

  @unittest.skip("reshape is even more complex")
  def test_super_complex(self):
    self.st = ShapeTracker((4, 4))
    self.st.permute(1,0)
    self.st.reshape(2, 2, 2, 2)
    self.st.permute(2,3,0,1)
    assert self.st.contiguous

  def test_work(self):
    self.st = ShapeTracker((64, 1024, 4))
    self.st.reshape(1, 64, 128, 32)
    self.st.permute(0, 3, 1, 2)
    self.st.reshape(1, 32, 1, 64, 128)
    self.st.permute(0, 3, 4, 1, 2)
    assert self.st.contiguous

  def test_work2(self):
    self.st = ShapeTracker((64, 1024, 4))
    self.st.reshape(1, 64, 128, 32)
    self.st.permute(0, 3, 1, 2)
    self.st.reshape(1, 1, 32, 64, 128)
    self.st.permute(0, 3, 4, 1, 2)
    self.st.reshape(64, 1024, 4)
    print(self.st.views)
    assert self.st.contiguous

class TestSingleShapeTracker(unittest.TestCase):
  def setUp(self):
    self.st = ShapeTracker((7,4))

  def test_reshape(self):
    self.st.reshape(7,1,4)
    assert self.st.contiguous

  def test_permute(self):
    self.st.permute(1,0)
    assert not self.st.contiguous

  def test_shrink(self):
    self.st.shrink((1,2), (0,4))
    assert not self.st.contiguous

  def test_double_permute(self):
    self.st.permute(1,0)
    self.st.permute(1,0)
    assert self.st.contiguous

  def test_reshape_permute(self):
    self.st.reshape(7,1,4)
    self.st.permute(0,1,2)
    assert self.st.contiguous

  def test_reshape_permute_yes(self):
    self.st.reshape(7,1,4)
    self.st.permute(0,2,1)
    assert self.st.contiguous

  def test_reshape_permute_no(self):
    self.st.reshape(4,7)
    self.st.permute(1,0)
    assert not self.st.contiguous

def shapetracker_getitem(st, val):
  locals = {"idx": val, "valid": 1}
  exec(st.expr(), None, locals)
  return locals["idx"] if locals["valid"] else -1

class TestShapeTracker(unittest.TestCase):
  def setUp(self):
    self.st = ShapeTracker((7,4))
    self.dt = DumbShapeTracker((7,4))
    self.apply = lambda fxn: [fxn(x) for x in [self.st, self.dt]]

  def tearDown(self):
    x = [shapetracker_getitem(self.st, i) for i in range(prod(self.st.shape))]
    y = [self.dt[i] for i in range(prod(self.dt.shape))]
    print(x,y, self.st.shape, self.dt.shape, self.st.expr())
    assert self.st.shape == self.dt.shape
    assert x == y

  def test_noop(self):
    pass

  def test_simple_split(self):
    self.test_permute()
    self.apply(lambda x: x.reshape(prod(self.st.shape)))

  def test_reshape(self):
    assert self.st.shape == self.dt.shape
    new_shape = self.st.shape[::-1]
    self.apply(lambda x: x.reshape(*new_shape))

  def test_permute(self):
    assert self.st.shape == self.dt.shape
    if len(self.st.shape) == 2: self.apply(lambda x: x.permute(1,0))
    elif len(self.st.shape) == 3: self.apply(lambda x: x.permute(2,0,1))

  def test_reshape_with_1(self):
    assert self.st.shape == self.dt.shape
    new_shape = [self.st.shape[0], 1, self.st.shape[1]]
    self.apply(lambda x: x.reshape(*new_shape))

  def test_expand(self):
    self.test_reshape_with_1()
    new_shape = list(self.st.shape)
    new_shape[1] = 2
    self.apply(lambda x: x.expand(*new_shape))

  def test_flip_0(self):
    self.apply(lambda x: x.flip(0))

  def test_flip_1(self):
    self.apply(lambda x: x.flip(1))

  def test_flip_01(self):
    self.apply(lambda x: x.flip(0,1))

  def test_slice_0(self):
    self.apply(lambda x: x.shrink((1, x.shape[0]), (0, x.shape[1])))

  def test_slice_1(self):
    self.apply(lambda x: x.shrink((0, x.shape[0]), (1, x.shape[1])))

  def test_slice_1c1(self):
    self.apply(lambda x: x.shrink((0, 1), (0, 1)))

  def test_slice_1c2(self):
    self.apply(lambda x: x.shrink((1, 2), (1, 2)))
  
  def test_double_permute(self):
    self.apply(lambda x: x.permute(1, 0))
    self.apply(lambda x: x.permute(1, 0))

  def test_slice_permute(self):
    self.apply(lambda x: x.shrink((0, 2), (2, 4)))
    self.apply(lambda x: x.permute(1, 0))

  def test_slice_expand(self):
    self.apply(lambda x: x.shrink((0, 2), (3, 4)))
    self.apply(lambda x: x.expand(2, 10))

  def test_double_stride(self):
    self.apply(lambda x: x.stride(1, 2))
    self.apply(lambda x: x.stride(2, 1))

  def test_stride(self): self.apply(lambda x: x.stride(2,1))
  def test_stride_int(self): self.apply(lambda x: x.stride(1,2))
  def test_stride_2(self): self.apply(lambda x: x.stride(2,2))
  def test_stride_n(self): self.apply(lambda x: x.stride(-2,1))
  def test_stride_int_n(self): self.apply(lambda x: x.stride(-1,2))
  def test_stride_2_n(self): self.apply(lambda x: x.stride(-2,-2))

  def test_reshape_then_permute(self):
    self.test_reshape()
    self.test_permute()

  def test_reshape_then_expand(self):
    self.test_reshape()
    self.test_expand()

  def test_permute_then_reshape(self):
    self.test_permute()
    self.test_reshape()

  def test_expand_then_reshape(self):
    self.test_expand()
    self.test_reshape()

  def test_combo(self):
    self.test_permute()
    self.test_reshape()
    self.test_slice_1()
    self.test_expand()
    self.test_permute()

if __name__ == '__main__':
  unittest.main()
