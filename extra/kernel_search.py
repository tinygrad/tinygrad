#!/usr/bin/env python
import os, random, traceback
import numpy as np
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.llops.ops_gpu import GPUBuffer, CLASTKernel, CL
from test.lib_test_ast import test_ast

def get_random_intervention(k):
  typ = random.randint(0, 1)
  if typ == 0:
    while 1:
      a1 = random.randint(0, k.shape_len-1)
      a2 = random.randint(0, k.shape_len-1)
      if a1 == a2: continue
      if a1 < k.first_reduce and a2 >= k.first_reduce: continue
      if a1 >= k.first_reduce and a2 < k.first_reduce: continue
      return 0, a1, a2
  elif typ == 1:
    while 1:
      up_axis = random.randint(0, k.shape_len-1)
      amount = random.choice([4, 8])
      if not all(st.shape[up_axis] == 1 or st.shape[up_axis]%amount == 0 for st in k.sts): continue
      return 1, up_axis, amount

def apply_intervention(k, typ, *dat):
  if typ == 0:
    # swap axes
    a1, a2 = dat
    new_order = list(range(0, k.shape_len))
    new_order[a1], new_order[a2] = new_order[a2], new_order[a1] 
    k.reshape_and_permute(None, new_order)
  elif typ == 1:
    # upcast
    up_axis, amount = dat[0], dat[1]
    # no change, we added a dimension
    k.reshape_and_permute(
      lambda x: list(x[0:up_axis]) + ([x[up_axis]//amount, amount] if x[up_axis] > 1 else [1,1]) + list(x[up_axis+1:]),
      [i for i in range(k.shape_len+1) if i != up_axis+1] + [up_axis+1])
    # drop the last dimension
    k.upcast()


def search(ast):
  # get baseline
  k = CLASTKernel(ast)
  for i in range(3):
    CL.time_sum = 0
    k.codegen()(*k.bufs)

  winning_interventions = []
  best_time = baseline = CL.time_sum

  def test():
    nonlocal winning_interventions, best_time
    k = CLASTKernel(ast)
    for w in winning_interventions: apply_intervention(k, *w)

    inter = get_random_intervention(k)
    apply_intervention(k, *inter)
    k.simplify_ones()

    # TODO: support upcasting, splitting, and local grouping for reduce
    CL.time_sum = 0
    k.codegen()(*k.bufs)
    if CL.time_sum < best_time * 0.95:
      print(f"accepting {inter} with time {best_time:.2f} -> {CL.time_sum:.2f} ratio {best_time/CL.time_sum:.2f}x")
      best_time = CL.time_sum
      winning_interventions.append(inter)

  for i in range(100):
    try:
      test()
    except Exception as e:
      traceback.print_exc()
      pass

  # run best
  print(f"winning interventions {winning_interventions}")
  for i in range(3):
    k = CLASTKernel(ast)
    for w in winning_interventions: apply_intervention(k, *w)
    k.codegen()(*k.bufs)
  test_ast(k)
  print(f"improved from {baseline/1e6:.2f} ms to {best_time/1e6:.2f} ms, a {baseline/best_time:.2f}x speedup")

if __name__ == "__main__":
  if int(os.getenv("OP", "0")) == 1:
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 3, 3, 3, 4), views=[View((1, 130, 258, 1, 12), (393216, 3072, 12, 12, 1), -3084), ZeroView((1, 128, 256, 1, 12), ((0, 1), (-1, 129), (-1, 257), (0, 1), (0, 12))), View((1, 64, 128, 8, 4, 3, 3, 3, 4), (0, 6192, 24, 0, 0, 3096, 12, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(128, 768, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 3, 3, 3, 4), views=[View((1, 64, 128, 8, 4, 3, 3, 3, 4), (0, 0, 0, 432, 4, 144, 16, 48, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 108, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 64, 128, 8, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(32,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    op3 = LazyOp(UnaryOps.RELU, (op2,), None)
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    buf4 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    op4 = LazyOp(UnaryOps.EXP, (op2,), None)
    op5 = LazyOp(BinaryOps.SUB, (buf4,op4,), None)
    op6 = LazyOp(UnaryOps.RELU, (op5,), None)
    op7 = LazyOp(BinaryOps.MUL, (buf3,op6,), None)
    op8 = LazyOp(BinaryOps.SUB, (op3,op7,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op8,), (64, 1024, 4))
  elif int(os.getenv("OP", "0")) == 2:
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 3, 3), views=[View((1, 66, 130, 32, 1), (262144, 4096, 32, 1, 1), -4128), ZeroView((1, 64, 128, 32, 1), ((0, 1), (-1, 65), (-1, 129), (0, 32), (0, 1))), View((1, 64, 128, 8, 4, 1, 1, 3, 3), (266240, 4160, 32, 4, 1, 12480, 12480, 4160, 32), 0)]), hostbuf=GPUBuffer(shape=(64, 1024, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 3, 3), views=[View((1, 64, 128, 8, 4, 1, 1, 3, 3), (0, 0, 0, 36, 1, 0, 0, 12, 4), 0)]), hostbuf=GPUBuffer(shape=(8, 9, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 64, 128, 8, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(32,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    op3 = LazyOp(UnaryOps.RELU, (op2,), None)
    buf3 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    buf4 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 8, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 8, 4, 1, 1, 1, 1), (0, 0, 0, 0, 0, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.], dtype=np.float32)))
    op4 = LazyOp(UnaryOps.EXP, (op2,), None)
    op5 = LazyOp(BinaryOps.SUB, (buf4,op4,), None)
    op6 = LazyOp(UnaryOps.RELU, (op5,), None)
    op7 = LazyOp(BinaryOps.MUL, (buf3,op6,), None)
    op8 = LazyOp(BinaryOps.SUB, (op3,op7,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op8,), (64, 1024, 4))
  elif int(os.getenv("OP", "0")) == 3:
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 8, 4), views=[View((1, 64, 128, 4, 4, 1, 1, 8, 4), (0, 4096, 32, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(64, 1024, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 8, 4), views=[View((1, 64, 128, 4, 4, 1, 1, 8, 4), (0, 0, 0, 128, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(4, 32, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 64, 128, 4, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 4, 4, 1, 1, 1, 1), (0, 0, 0, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(16,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op2,), (64, 512, 4))
  elif int(os.getenv("BC", "0")):
    # big conv
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 3, 225, 225), (150528, 50176, 224, 1), 0), ZeroView((8, 3, 224, 224), ((0, 8), (0, 3), (0, 225), (0, 225))), View((8, 1, 32, 112, 112, 3, 3, 3), (151875, 151875, 0, 450, 2, 50625, 225, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 3, 224, 224), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 1, 32, 112, 112, 3, 3, 3), (0, 0, 27, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 3, 3), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (8, 1, 32, 112, 112, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (8, 32, 112, 112))
  elif int(os.getenv("GEMM", "0")):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 512, 512, 1, 1, 1, 512), views=[View((1, 512, 512, 1), (0, 1, 512, 0), 0), View((1, 1, 512, 512, 1, 1, 1, 512), (0, 0, 0, 1, 0, 0, 0, 512), 0)]), hostbuf=GPUBuffer(shape=(512, 512), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, 512, 512, 1, 1, 1, 512), views=[View((1, 1, 512, 512, 1, 1, 1, 512), (0, 0, 1, 0, 0, 0, 0, 512), 0)]), hostbuf=GPUBuffer(shape=(512, 512), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 1, 512, 512, 1, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (512, 512))
  elif int(os.getenv("FASTCONV", "0")):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(32, 1, 32, 32, 32, 64, 3, 3), views=[View((32, 1, 32, 32, 32, 64, 3, 3), (73984, 73984, 0, 34, 1, 1156, 34, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 64, 34, 34), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(32, 1, 32, 32, 32, 64, 3, 3), views=[View((32, 1, 32, 32, 32, 64, 3, 3), (0, 0, 576, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 64, 3, 3), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (32, 1, 32, 32, 32, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (32, 32, 32, 32))
  else:
    # reduce
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 32, 225, 225), (50176, 150528, 224, 1), 0), ZeroView((3, 32, 224, 224), ((0, 3), (0, 32), (0, 225), (0, 225))), View((3, 1, 32, 3, 3, 32, 112, 112), (1620000, 1620000, 0, 225, 1, 50625, 450, 2), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 224, 224), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 1, 32, 3, 3, 32, 112, 112), (0, 12845056, 401408, 0, 0, 12544, 112, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 1, 32, 1, 1, 32, 112, 112), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (3, 1, 32, 3, 3, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (3, 32, 3, 3))
  search(ast)
