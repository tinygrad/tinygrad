#!/usr/bin/env python
import random
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.llops.ops_gpu import GPUBuffer, CLASTKernel

def search(ast):
  for i in range(20):
    k = CLASTKernel(ast)

    if i != 0:
      # do search space
      order_outer = list(range(0, k.first_reduce))
      random.shuffle(order_outer)
      order_inner = list(range(k.first_reduce, k.shape_len))
      random.shuffle(order_inner)
      order = order_outer + order_inner
      print(order)
      k.reshape_and_permute(None, order)

    k.codegen()(*k.bufs)



if __name__ == "__main__":
  buf0 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 3, 225, 225), (150528, 50176, 224, 1), 0), ZeroView((8, 3, 224, 224), ((0, 8), (0, 3), (0, 225), (0, 225))), View((8, 1, 32, 112, 112, 3, 3, 3), (151875, 151875, 0, 450, 2, 50625, 225, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 3, 224, 224), force_create=True))
  buf1 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 1, 32, 112, 112, 3, 3, 3), (0, 0, 27, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 3, 3), force_create=True))
  op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
  op1 = LazyOp(ReduceOps.SUM, (op0,), (8, 1, 32, 112, 112, 1, 1, 1))
  ast = LazyOp(MovementOps.RESHAPE, (op1,), (8, 32, 112, 112))
  search(ast)
