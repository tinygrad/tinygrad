#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.ops import LazyOp, BinaryOps, ReduceOps, MovementOps
from accel.llvm.ops_llvm import LLVMBuffer
from tinygrad.tensor import Tensor

class TestLLVM(unittest.TestCase):
  def test_add(self):
    a = LLVMBuffer.fromCPU(np.ones((4,4)))
    b = LLVMBuffer.fromCPU(np.ones((4,4)))
    ast = LazyOp(BinaryOps.ADD, (a,b))
    ret = LLVMBuffer((4,4)).exec_ast(ast)
    print(ret.toCPU())

  def test_sum(self):
    a = LLVMBuffer.fromCPU(np.ones((4,4)))
    ast = LazyOp(ReduceOps.SUM, (a,), (1,1))
    ret = LLVMBuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())

  def test_sum_add(self):
    a = LLVMBuffer.fromCPU(np.ones((4,4)))
    b = LLVMBuffer.fromCPU(np.ones((1,1)))
    ast = LazyOp(ReduceOps.SUM, (a,), (1,1))
    ast = LazyOp(BinaryOps.ADD, (ast,b))
    ret = LLVMBuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())

  def test_add_sum(self):
    a = LLVMBuffer.fromCPU(np.ones((4,4)))
    b = LLVMBuffer.fromCPU(np.ones((4,4)))
    ast = LazyOp(BinaryOps.ADD, (a,b))
    ast = LazyOp(ReduceOps.SUM, (ast,), (1,1))
    ret = LLVMBuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())

  def test_add_sum_add(self):
    a = LLVMBuffer.fromCPU(np.ones((4,4)))
    b = LLVMBuffer.fromCPU(np.ones((4,4)))
    c = LLVMBuffer.fromCPU(np.ones((1,1)))
    ast = LazyOp(BinaryOps.ADD, (a,b))
    ast = LazyOp(ReduceOps.SUM, (ast,), (1,1))
    ast = LazyOp(BinaryOps.ADD, (ast,c))
    ret = LLVMBuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())

  #TODO: Fix This!! repo using
  # OPT=1 LLVM=1 LLVM_CACHE_DEBUG=1 python3 test/external_test_llvm.py
  def test_add_add(self):
    # This below is the high level op, and results 4's not 3's
    """
    x = Tensor(np.ones((4,4)), requires_grad=True)
    y = (x + x + x).realize().numpy()
    print(y)
    expected = (np.ones((4,4)) * 3)
    #self.assertEqual (y[0][0], (np.ones((4,4)) * 3)[0][0])
    """
    # This is the LLVMBuff way of doing it, 
    # since its sharing a buff a, stuff goes wrong,
    # without the RESHAPE, the CFUNC throws a buffer error,
    # but the reshapes prevent that causing wrong output
    a = LLVMBuffer.fromCPU(np.ones((4,4))) 
    ast = LazyOp(BinaryOps.ADD, (a,a))
    #ast = LazyOp(MovementOps.RESHAPE, (ast,), (4,4))
    c = LLVMBuffer((4,4)).exec_ast(ast)
    ast = LazyOp(BinaryOps.ADD, (c,a))
    #ast = LazyOp(MovementOps.RESHAPE, (ast,), (4,4))
    ret = LLVMBuffer((4,4)).exec_ast(ast)
    print(ret.toCPU())


if __name__ == "__main__":
  unittest.main()

  """
  ast = LazyOp(BinaryOps.ADD, (a,b))
  ast = LazyOp(ReduceOps.SUM, (ast,), (1,1))
  ret = LLVMBuffer((4,4)).exec_ast(ast)
  print(ret.toCPU())
  """



