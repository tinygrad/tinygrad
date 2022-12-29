#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.ops import LazyOp, BinaryOps, ReduceOps
from accel.cuda.ops_cuda import CUDABuffer

class TestCUDA(unittest.TestCase):
  def test_add(self):
    a = CUDABuffer.fromCPU(np.ones((4,4)))
    b = CUDABuffer.fromCPU(np.ones((4,4)))
    ast = LazyOp(BinaryOps.ADD, (a,b))
    ret = CUDABuffer((4,4)).exec_ast(ast)
    print(ret.toCPU())
"""
  def test_sum(self):
    a = CUDABuffer.fromCPU(np.ones((4,4)))
    ast = LazyOp(ReduceOps.SUM, (a,), (1,1))
    ret = CUDABuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())

  def test_sum_add(self):
    a = CUDABuffer.fromCPU(np.ones((4,4)))
    b = CUDABuffer.fromCPU(np.ones((1,1)))
    ast = LazyOp(ReduceOps.SUM, (a,), (1,1))
    ast = LazyOp(BinaryOps.ADD, (ast,b))
    ret = CUDABuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())

  def test_add_sum(self):
    a = CUDABuffer.fromCPU(np.ones((4,4)))
    b = CUDABuffer.fromCPU(np.ones((4,4)))
    ast = LazyOp(BinaryOps.ADD, (a,b))
    ast = LazyOp(ReduceOps.SUM, (ast,), (1,1))
    ret = CUDABuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())

  def test_add_sum_add(self):
    a = CUDABuffer.fromCPU(np.ones((4,4)))
    b = CUDABuffer.fromCPU(np.ones((4,4)))
    c = CUDABuffer.fromCPU(np.ones((1,1)))
    ast = LazyOp(BinaryOps.ADD, (a,b))
    ast = LazyOp(ReduceOps.SUM, (ast,), (1,1))
    ast = LazyOp(BinaryOps.ADD, (ast,c))
    ret = CUDABuffer((1,1)).exec_ast(ast)
    print(ret.toCPU())
"""


if __name__ == "__main__":
  unittest.main()

  """
  ast = LazyOp(BinaryOps.ADD, (a,b))
  ast = LazyOp(ReduceOps.SUM, (ast,), (1,1))
  ret = CUDABuffer((4,4)).exec_ast(ast)
  print(ret.toCPU())
  """



