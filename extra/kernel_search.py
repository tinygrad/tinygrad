#!/usr/bin/env python
import random
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.llops.ops_gpu import GPUBuffer, CLASTKernel, CL
from test.lib_test_ast import test_ast

def search(ast):

  # get baseline
  k = CLASTKernel(ast)
  order = list(range(0, k.shape_len))
  CL.time_sum = 0
  k.codegen()(*k.bufs)
  best_time = CL.time_sum

  for i in range(200):
    k = CLASTKernel(ast)

    a1 = random.randint(0, k.shape_len-1)
    a2 = random.randint(0, k.shape_len-1)
    if a1 == a2: continue

    new_order = order[:]
    new_order[a1], new_order[a2] = new_order[a2], new_order[a1] 
    k.reshape_and_permute(None, new_order)

    # TODO: support upcasting, splitting, and local grouping
    CL.time_sum = 0
    try:
      k.codegen()(*k.bufs)
    except Exception:
      # reject
      continue

    if CL.time_sum < best_time:
      print(f"accepting {order} -> {new_order}")
      best_time = CL.time_sum
      order = new_order

    #print(CL.time_sum)

  # run best
  print(f"best order {order}")
  for i in range(3):
    k = CLASTKernel(ast)
    k.reshape_and_permute(None, order)
    k.codegen()(*k.bufs)

if __name__ == "__main__":
  # big conv
  #buf0 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 3, 225, 225), (150528, 50176, 224, 1), 0), ZeroView((8, 3, 224, 224), ((0, 8), (0, 3), (0, 225), (0, 225))), View((8, 1, 32, 112, 112, 3, 3, 3), (151875, 151875, 0, 450, 2, 50625, 225, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 3, 224, 224), force_create=True))
  #buf1 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 1, 32, 112, 112, 3, 3, 3), (0, 0, 27, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 3, 3), force_create=True))
  #op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
  #op1 = LazyOp(ReduceOps.SUM, (op0,), (8, 1, 32, 112, 112, 1, 1, 1))
  #ast = LazyOp(MovementOps.RESHAPE, (op1,), (8, 32, 112, 112))

  # reduce
  buf0 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 32, 225, 225), (50176, 150528, 224, 1), 0), ZeroView((3, 32, 224, 224), ((0, 3), (0, 32), (0, 225), (0, 225))), View((3, 1, 32, 3, 3, 32, 112, 112), (1620000, 1620000, 0, 225, 1, 50625, 450, 2), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 224, 224), force_create=True))
  buf1 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 1, 32, 3, 3, 32, 112, 112), (0, 12845056, 401408, 0, 0, 12544, 112, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 1, 32, 1, 1, 32, 112, 112), force_create=True))
  op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
  op1 = LazyOp(ReduceOps.SUM, (op0,), (3, 1, 32, 3, 3, 1, 1, 1))
  ast = LazyOp(MovementOps.RESHAPE, (op1,), (3, 32, 3, 3))
  search(ast)
