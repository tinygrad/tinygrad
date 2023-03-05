#!/usr/bin/env python
import unittest
from tinygrad.ops import InterpretedBuffer, GenericShape, LazyOp, BinaryOps, get_lazyop_info

class TestFlopCounter(unittest.TestCase):
  def test_flops_add(self):
    buf0 = InterpretedBuffer(GenericShape((4,)))
    buf1 = InterpretedBuffer(GenericShape((4,)))
    op0 = LazyOp(BinaryOps.ADD, (buf0,buf1,), None)
    info = get_lazyop_info(op0)
    self.assertEqual(info.flops, 4)

  def test_flops_add_twice(self):
    buf0 = InterpretedBuffer(GenericShape((4,)))
    buf1 = InterpretedBuffer(GenericShape((4,)))
    op0 = LazyOp(BinaryOps.ADD, (buf0,buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,buf1,), None)
    info = get_lazyop_info(op1)
    self.assertEqual(info.flops, 8)

  def test_flops_add_self(self):
    buf0 = InterpretedBuffer(GenericShape((4,)))
    buf1 = InterpretedBuffer(GenericShape((4,)))
    op0 = LazyOp(BinaryOps.ADD, (buf0,buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,op0,), None)
    info = get_lazyop_info(op1)
    self.assertEqual(info.flops, 8)

  def test_flops_add_roundabout_self(self):
    buf0 = InterpretedBuffer(GenericShape((4,)))
    buf1 = InterpretedBuffer(GenericShape((4,)))
    op0 = LazyOp(BinaryOps.ADD, (buf0,buf1,), None)
    op1 = LazyOp(BinaryOps.ADD, (op0,buf1,), None)
    op2 = LazyOp(BinaryOps.ADD, (op0,op1,), None)
    info = get_lazyop_info(op2)
    self.assertEqual(info.flops, 12)
  
if __name__ == '__main__':
  unittest.main()
