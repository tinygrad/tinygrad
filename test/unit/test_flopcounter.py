#!/usr/bin/env python
import unittest
from typing import NamedTuple, Tuple
from tinygrad.ops import LazyOp, BinaryOps, ReduceOps, get_lazyop_info
from tinygrad.helpers import DType, dtypes

class TestBuffer(NamedTuple):
  shape: Tuple[int, ...]
  dtype: DType

class TestFlopCounter(unittest.TestCase):
  def setUp(self):
    self.buf0 = TestBuffer(shape=(4,), dtype=dtypes.float32)
    self.buf1 = TestBuffer(shape=(4,), dtype=dtypes.float32)

  def test_flops_add(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    info = get_lazyop_info(op0)
    self.assertEqual(info.flops, 4)

  def test_flops_add_twice(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,self.buf1,), None)
    info = get_lazyop_info(op1)
    self.assertEqual(info.flops, 8)

  def test_flops_add_self(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,op0,), None)
    info = get_lazyop_info(op1)
    self.assertEqual(info.flops, 8)

  def test_flops_add_roundabout_self(self):
    op0 = LazyOp(BinaryOps.ADD, (self.buf0,self.buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,self.buf1,), None)
    op2 = LazyOp(BinaryOps.ADD, (op0,op1,), None)
    info = get_lazyop_info(op2)
    self.assertEqual(info.flops, 12)

  def test_flops_red(self):
    op0 = LazyOp(BinaryOps.MUL, (self.buf0,self.buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1,))
    op2 = LazyOp(BinaryOps.ADD, (op1, op1,), None)
    info = get_lazyop_info(op2)
    self.assertEqual(info.flops, 9)

if __name__ == '__main__':
  unittest.main()
