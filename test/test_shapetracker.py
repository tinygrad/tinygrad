#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.helpers import prod
from tinygrad.shapetracker import ShapeTracker

def strides_for_shape(shape):
  strides = [1]
  for d in shape[::-1][:-1]:
    strides = [d*strides[0]] + strides
  return strides

class View:
  def __init__(self, shape, strides, offset=0):
    self.shape = shape
    self.strides = strides
    self.offset = offset
  
  def __getitem__(self, val):
    ret = self.offset
    for d,s in zip(self.shape[::-1], self.strides[::-1]):
      ret += (val%d) * s
      val //= d
    return ret

class StackedViewShapeTracker:
  def __init__(self, *shape):
    self.views = []
    self.views.append(View(shape, strides_for_shape(shape)))

  @property
  def shape(self):
    return self.views[-1].shape

  def reshape(self, *new_shape):
    self.views.append(View(new_shape, strides_for_shape(new_shape)))


class DumbShapeTracker:
  def __init__(self, *shape):
    self.t = np.arange(prod(shape), dtype=np.uint8).reshape(shape)

  @property
  def shape(self):
    return self.t.shape

  def reshape(self, *new_shape):
    self.t = self.t.reshape(new_shape)
    #print("reshape", self.t.shape, self.t.strides)

  def permute(self, *axis):
    self.t = np.transpose(self.t, axis)
    #print("permute", self.t.shape, self.t.strides)

  def expand(self, *new_shape):
    self.t = np.broadcast_to(self.t, new_shape)
    #print("expand", self.t.shape, self.t.strides)

  def flip(self, *axis):
    self.t = np.flip(self.t, axis)
    #print("flip", self.t.shape, self.t.strides)

  def slice(self, arg):
    # TODO: negative means pad with 0s, not negative indexing like in numpy
    # Use -1 to represent index of 0
    pass

  def __getitem__(self, val):
    return self.t.flatten()[val]

# Tensor.zeros(2, 4).permute(1,0).reshape(2, 4)
# (d1*4 + d0%4), d1=x//4, d0=x%4 = ((x//4)*4) + (x%4)%4



class TestShapeTracker(unittest.TestCase):
  def setUp(self):
    self.st = ShapeTracker(2,4)
    self.dt = DumbShapeTracker(2,4)


  def tearDown(self):
    x = [self.st[i] for i in range(prod(self.st.shape))]
    y = [self.dt[i] for i in range(prod(self.dt.shape))]
    print(x,y)
    assert self.st.shape == self.dt.shape
    assert x == y

  def test_noop(self):
    pass

  def test_simple_split(self):
    self.test_permute()
    fxn = lambda x: x.reshape(8)
    [fxn(x) for x in [self.st, self.dt]]

  def test_reshape(self):
    assert self.st.shape == self.dt.shape
    new_shape = self.st.shape[::-1]
    fxn = lambda x: x.reshape(*new_shape)
    [fxn(x) for x in [self.st, self.dt]]

  def test_permute(self):
    fxn = lambda x: x.permute(1,0)
    [fxn(x) for x in [self.st, self.dt]]

  def test_expand(self):
    assert self.st.shape == self.dt.shape
    new_shape = [self.st.shape[0], 1, self.st.shape[1]]
    fxn = lambda x: x.reshape(*new_shape)
    [fxn(x) for x in [self.st, self.dt]]

    new_shape[1] = 2
    fxn = lambda x: x.expand(*new_shape)
    [fxn(x) for x in [self.st, self.dt]]

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

if __name__ == '__main__':
  unittest.main()
