#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.shapetracker import ShapeTracker

class TestShapeTracker(unittest.TestCase):
  def setUp(self):
    self.buf = np.arange(10*20).reshape(10, 20)
    self.st = ShapeTracker(10,20)

  def tearDown(self):
    assert self.st.shape == self.buf.shape
    assert self.st[4, 5] == self.buf[4, 5]
    assert self.st[8, 3] == self.buf[8, 3]

  def test_noop(self):
    pass

  def test_reshape(self):
    self.buf = self.buf.reshape(20, 10)
    self.st.reshape(20,10)

  def test_permute(self):
    self.buf = self.buf.transpose(1,0)
    self.st.permute(1,0)

  def test_reshape_then_permute(self):
    self.test_reshape()
    self.test_permute()

  def test_permute_then_reshape(self):
    self.test_permute()
    self.test_reshape()

if __name__ == '__main__':
  unittest.main()
