#!/usr/bin/env python
import random, traceback
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.llops.ops_gpu import GPUBuffer, CLASTKernel, CL
from test.lib_test_ast import test_ast

def get_pair(k):
  while 1:
    a1 = random.randint(0, k.shape_len-1)
    a2 = random.randint(0, k.shape_len-1)
    if a1 == a2: continue
    if a1 < k.first_reduce and a2 >= k.first_reduce: continue
    if a1 >= k.first_reduce and a2 < k.first_reduce: continue
    return a1, a2

def search(ast):
  # get baseline
  k = CLASTKernel(ast)
  CL.time_sum = 0
  k.codegen()(*k.bufs)

  order = list(range(0, k.shape_len))
  best_time = CL.time_sum
  def test():
    nonlocal order, best_time
    k = CLASTKernel(ast)

    a1, a2 = get_pair(k)
    new_order = order[:]
    new_order[a1], new_order[a2] = new_order[a2], new_order[a1] 
    k.reshape_and_permute(None, new_order)

    """
    up_axis = random.randint(0, k.shape_len-1)
    # no change, we added a dimension
    k.reshape_and_permute(
      lambda x: list(x[0:up_axis]) + ([x[up_axis]//4, 4] if x[up_axis] > 1 else [1,1]) + list(x[up_axis+1:]),
      [i for i in range(k.shape_len+1) if i != up_axis+1] + [up_axis+1])
    # drop the last dimension
    k.upcast()
    """

    # TODO: support upcasting, splitting, and local grouping for reduce
    CL.time_sum = 0
    k.codegen()(*k.bufs)
    if CL.time_sum < best_time:
      print(f"accepting {order} -> {new_order} with time {best_time} -> {CL.time_sum}")
      best_time = CL.time_sum
      order = new_order

    #print(CL.time_sum)

  for i in range(100):
    try:
      test()
    except Exception:
      #traceback.print_exc()
      continue

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
