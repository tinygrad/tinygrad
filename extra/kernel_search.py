#!/usr/bin/env python
import random, traceback
import time
import itertools
from enum import Enum
import numpy as np
from tinygrad.ops import LazyOp, ReduceOps, BinaryOps, UnaryOps, MovementOps
from tinygrad.shape import ShapeTracker, View, ZeroView
from tinygrad.runtime.ops_gpu import GPUBuffer, CLASTKernel
from tinygrad.runtime.opencl import OSX_TIMING_RATIO
from tinygrad.helpers import getenv, DEBUG
from extra.lib_test_ast import test_ast

import pickle, dbm
intervention_cache = None

Interventions = Enum("Interventions", ["SWAP", "UPCAST", "SHIFT", "REDUCE"])
def get_interventions(k, winning_interventions=[]):
  k.process()
  p1, p2, p3, p4, p5 = [], [], [], [], []
  p1 = [(Interventions.SWAP, x) for x in itertools.combinations(range(k.first_reduce), 2)]
  p2 = [(Interventions.SWAP, x) for x in itertools.combinations(range(k.first_reduce + len(k.group_for_reduce), k.shape_len), 2)]
  p3 = [(Interventions.UPCAST, None)] if max(st.shape[-1] for st in k.sts) <= 32 else []
  for up_axis in range(k.shape_len):
    if up_axis >= k.first_reduce and up_axis < (k.first_reduce + len(k.group_for_reduce)): continue
    max_up = max(st.shape[up_axis] for st in k.sts)
    if max_up == 1: continue
    for amount in sorted(list(set([2,4,8,max_up]))):
      if amount >= 32: continue
      if not all(st.shape[up_axis] == 1 or st.shape[up_axis]%amount == 0 for st in k.sts): continue
      p3.append((Interventions.UPCAST, (up_axis, amount)))
  """
  for up_axis in range(1,k.first_reduce):
    for amount in [4,8,16,32]:
      if k.sts[0].shape[up_axis] % amount == 0:
        p4.append((Interventions.SHIFT, (up_axis, amount, True)))
        p4.append((Interventions.SHIFT, (up_axis, amount, False)))
  """
  # no double reduce
  #if len([x for x in winning_interventions if x[0] == Interventions.REDUCE]) == 0:
  # in fact, reduce first
  if len(winning_interventions) == 0:
    for axis in range(k.first_reduce + len(k.group_for_reduce), k.shape_len):
      max_up = max(st.shape[axis] for st in k.sts)
      if max_up <= 1024: p5 += [(Interventions.REDUCE, (axis, max_up))]
      if max_up % 256 == 0: p5 += [(Interventions.REDUCE, (axis, 256))]
      if max_up % 16 == 0: p5 += [(Interventions.REDUCE, (axis, 16))]
  return p1+p2+p3+p4+p5

def apply_intervention(k, typ, dat):
  k.process()
  if typ == Interventions.SWAP:
    # swap axes
    a1, a2 = dat
    new_order = list(range(0, k.shape_len))
    new_order[a1], new_order[a2] = new_order[a2], new_order[a1]
    k.reshape_and_permute(None, new_order)
  elif typ == Interventions.UPCAST:
    if dat is not None:
      # upcast
      up_axis, amount = dat[0], dat[1]
      # no change, we added a dimension
      k.reshape_and_permute(
        lambda x: list(x[0:up_axis]) + ([x[up_axis]//amount, amount] if x[up_axis] > 1 else [1,1]) + list(x[up_axis+1:]),
        [i for i in range(k.shape_len+1) if i != up_axis+1] + [up_axis+1])
    # drop the last dimension
    k.upcast()
  elif typ == Interventions.SHIFT:
    up_axis, amount, flip = dat[0], dat[1], dat[2]
    k.reshape_and_permute(
      lambda x: list(x[0:up_axis]) + (([amount, x[up_axis]//amount] if flip else [x[up_axis]//amount, amount]) if x[up_axis] > 1 else [1,1]) + list(x[up_axis+1:]),
      [up_axis] + [i for i in range(k.shape_len+1) if i != up_axis])
  elif typ == Interventions.REDUCE:
    up_axis, amount = dat[0], dat[1]
    # no change, we added a dimension
    k.reshape_and_permute(
      lambda x: list(x[0:up_axis]) + ([x[up_axis]//amount, amount] if x[up_axis] > 1 else [1,1]) + list(x[up_axis+1:]),
      [i for i in range(k.first_reduce) if i != up_axis+1] + [up_axis+1] + [i for i in range(k.first_reduce, k.shape_len+1) if i != up_axis+1])
    k.group_for_reduce.append(amount)
  k.simplify_ones()
  k.simplify_merge_adjacent()

def run_and_time(k,cnt=3,local_override=None):
  prog = k.codegen()
  ret = []
  for i in range(cnt):
    t1 = time.monotonic_ns()
    if local_override: prog.local_work_size = local_override
    e = prog(*k.bufs)
    e.wait()
    t4 = time.monotonic_ns()
    t2, t3 = e.profile.start * OSX_TIMING_RATIO, e.profile.end * OSX_TIMING_RATIO
    #print(*[f"{(x-t1)*1e-3:7.2f} us" for x in [t1, t2, t3, t4]])  # TODO: this may be wrong on non OS X
    #assert t1 < t2 < t3 < t4, "timings not in order"
    ret.append(t3-t2)
    #ret.append(t4-t1)
  return min(ret)

def search_one(ast, winning_interventions=[], debug=False):
  k = CLASTKernel(ast)
  for w in winning_interventions: apply_intervention(k, *w)
  ints = get_interventions(k, winning_interventions)
  options = [(run_and_time(k), None, 0.9)]
  name = k.fxn.name
  ops = k.fxn.op_estimate
  if debug: print(f"{options[-1][1]} : {options[-1][0]*1e-3:.2f}")
  for int in ints:
    try:
      k = CLASTKernel(ast)
      for w in winning_interventions: apply_intervention(k, *w)
      apply_intervention(k, *int)
      options.append((run_and_time(k), int, 1.0))
      #test_ast(k)
      if debug: print(f"{options[-1][1]} : {options[-1][0]*1e-3:.2f}")
    except Exception:
      if debug: print(int, "FAILED")
      #traceback.print_exc()
      pass
  baseline = options[0]
  options = sorted(options, key=lambda x: x[0]*x[2])
  best = options[0]
  print(f"{name:30s} {baseline[0]/1e3:9.2f} us -> {best[0]/1e3:9.2f} us {baseline[0]/best[0]:7.2f}x {ops/best[0]*1e-3:5.2f}T *with* {winning_interventions} + {best[1]}")
  return best

def apply_optimization(k, ast, max_interventions=1, cache=True):
  global intervention_cache
  if intervention_cache is None: intervention_cache = dbm.open('/tmp/kopt.db', 'c')
  from extra.kernel_search import search_one, apply_intervention
  if k.key not in intervention_cache or cache == False:
    winning_interventions = []
    for i in range(max_interventions):   # NOTE: multiple interventions is breaking the ASTs
      oo = search_one(ast, winning_interventions)
      if oo[1] is None: break
      winning_interventions.append(oo[1])
    intervention_cache[k.key] = pickle.dumps(winning_interventions)
  ic = pickle.loads(intervention_cache[k.key])
  if DEBUG >= 3: print("intervention", ic)
  for w in ic: apply_intervention(k, *w)


def randomize_buffers(ast):
  # before testing, we need to fill the buffers with randomness
  bufs = get_buffers(ast)
  for b in bufs:
    randomness = np.random.default_rng().standard_normal(size=b._base_shape, dtype=np.float32)
    if b._buf is not None: b._buf.copyin(randomness)

def one(ast, winning_interventions, local_override=None):
  randomize_buffers(ast)
  k = CLASTKernel(ast)
  baseline = run_and_time(k, 1)

  k = CLASTKernel(ast)
  for w in winning_interventions: apply_intervention(k, *w)
  best = run_and_time(k, 1, local_override)

  name = k.fxn.name
  print(f"{name:30s} {baseline/1e3:9.2f} us -> {best/1e3:9.2f} us {baseline/best:7.2f}x *with* {winning_interventions}")
  if not getenv("NOTEST"): test_ast(k)

def search(ast, start_interventions=[], depth=10):
  winning_interventions = start_interventions[:]
  randomize_buffers(ast)
  k = CLASTKernel(ast)
  for w in winning_interventions: apply_intervention(k, *w)
  best_time = baseline = run_and_time(k)

  for i in range(depth):
    print(winning_interventions)
    oo = search_one(ast, winning_interventions, True)
    print(oo)
    if oo[1] is None: break
    winning_interventions.append(oo[1])
    best_time = oo[0]

  # run best
  print(f"winning interventions {winning_interventions}")
  for i in range(3):
    k = CLASTKernel(ast)
    for w in winning_interventions: apply_intervention(k, *w)
    k.codegen()(*k.bufs)
  #k.print()
  if not getenv("NOTEST"): test_ast(k)
  print(f"improved from {baseline/1e6:.2f} ms to {best_time/1e6:.2f} ms, a {baseline/best_time:.2f}x speedup @ {k.info.flops/best_time:.2f} GFLOPS")

from tinygrad.ops import get_buffers
def test_correctness(ast):
  randomize_buffers(ast)
  from extra.lib_test_ast import test_ast
  k = CLASTKernel(ast)
  ints = get_interventions(k)
  k.codegen()(*k.bufs)
  test_ast(k)
  print("correct at baseline")
  for int in ints:
    print("***** APPLYING INTERVENTION", int)
    k = CLASTKernel(ast)
    k.printbufs("old:")
    apply_intervention(k, *int)
    k.printbufs("new:")
    k.codegen()(*k.bufs)
    print("***** TESTING INTERVENTION", int)
    test_ast(k)

if __name__ == "__main__":
  if intervention_cache is None: intervention_cache = dbm.open('/tmp/kopt.db', 'c')
  if getenv("DUMP"):
    keys = list(intervention_cache.keys())
    from collections import defaultdict
    cnts = defaultdict(int)
    for k in keys:
      ic = pickle.loads(intervention_cache[k])
      for i in ic:
        cnts[i] += 1
    for k,v in sorted(cnts.items(), key=lambda x: -x[1]):
      print(k, v)
    exit(0)
  if getenv("OP", 0) == 1:
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
  elif getenv("OP", 0) == 2:
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
  elif getenv("OP", 0) == 3:
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 8, 4), views=[View((1, 64, 128, 4, 4, 1, 1, 8, 4), (0, 4096, 32, 0, 0, 0, 0, 4, 1), 0)]), hostbuf=GPUBuffer(shape=(64, 1024, 4), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 8, 4), views=[View((1, 64, 128, 4, 4, 1, 1, 8, 4), (0, 0, 0, 128, 4, 0, 0, 16, 1), 0)]), hostbuf=GPUBuffer(shape=(4, 32, 4), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 64, 128, 4, 4, 1, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(1, 64, 128, 4, 4, 1, 1, 1, 1), views=[View((1, 64, 128, 4, 4, 1, 1, 1, 1), (0, 0, 0, 4, 1, 1, 1, 1, 1), 0)]), hostbuf=GPUBuffer(shape=(16,), force_create=True))
    op2 = LazyOp(BinaryOps.ADD, (op1,buf2,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op2,), (64, 512, 4))
  elif getenv("REDUCE", 0):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(32, 8, 112, 112), views=[View((32, 8, 112, 112), (12544, 401408, 112, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 32, 112, 112), force_create=True))
    op0 = LazyOp(ReduceOps.SUM, (buf0,), (32, 1, 1, 1))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(32, 1, 1, 1), views=[View((32, 1, 1, 1), (0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([9.964923e-06], dtype=np.float32)))
    op1 = LazyOp(BinaryOps.MUL, (op0,buf1,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (1, 32, 1, 1))
  elif getenv("CONVW", 0):
    # re_S64_128_3_3
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(64, 1, 128, 3, 3, 512, 32, 32), views=[View((64, 512, 34, 34), (1024, 65536, 32, 1), -33), ZeroView((64, 512, 32, 32), ((0, 64), (0, 512), (-1, 33), (-1, 33))), View((64, 1, 128, 3, 3, 512, 32, 32), (591872, 591872, 0, 34, 1, 1156, 34, 1), 0)]), hostbuf=GPUBuffer(shape=(512, 64, 32, 32), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(64, 1, 128, 3, 3, 512, 32, 32), views=[View((64, 1, 128, 3, 3, 512, 32, 32), (0, 0, 1024, 0, 0, 131072, 32, 1), 0)]), hostbuf=GPUBuffer(shape=(512, 128, 32, 32), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (64, 1, 128, 3, 3, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (64, 128, 3, 3))
    ii = []
    #ii.append((Interventions.REDUCE, (6, 32)))
    #ii.append((Interventions.UPCAST, (3, 3)))
    #ii.append((Interventions.UPCAST, (2, 3)))
    #ii.append((Interventions.UPCAST, (3, 32)))
    #ii.append((Interventions.UPCAST, (0, 2)))
    #ii.append((Interventions.UPCAST, (0, 8)))
    #ii.append((Interventions.SWAP, (1, 3)))
    #ii.append((Interventions.UPCAST, (2, 3)))
    #ii.append((Interventions.UPCAST, (1, 128)))
    search(ast, ii)
    #one(ast, ii)
    #one(ast, [(Interventions.SWAP, (1, 3))])
    #one(ast, [(Interventions.UPCAST, (0, 8)), (Interventions.SWAP, (1, 3))])
    #one(ast, [(Interventions.UPCAST, (0, 8)), (Interventions.SWAP, (1, 3)), (Interventions.UPCAST, (6, 8))])
    exit(0)
  elif getenv("BC", 0):
    # big conv
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 3, 225, 225), (150528, 50176, 224, 1), 0), ZeroView((8, 3, 224, 224), ((0, 8), (0, 3), (0, 225), (0, 225))), View((8, 1, 32, 112, 112, 3, 3, 3), (151875, 151875, 0, 450, 2, 50625, 225, 1), 0)]), hostbuf=GPUBuffer(shape=(8, 3, 224, 224), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(8, 1, 32, 112, 112, 3, 3, 3), views=[View((8, 1, 32, 112, 112, 3, 3, 3), (0, 0, 27, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 3, 3), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (8, 1, 32, 112, 112, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (8, 32, 112, 112))
  elif getenv("SIMPLE_REDUCE", 0):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(64, 512, 32, 32), views=[View((64, 512, 32, 32), (524288, 1024, 32, 1), 0)]), hostbuf=GPUBuffer(shape=(64, 512, 32, 32), force_create=True))
    op0 = LazyOp(ReduceOps.SUM, (buf0,), (64, 1, 1, 1))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(64, 1, 1, 1), views=[View((64, 1, 1, 1), (0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([1.9073486e-06], dtype=np.float32)))
    op1 = LazyOp(BinaryOps.MUL, (op0,buf1,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (1, 64, 1, 1))
  elif getenv("GEMM", 0):
    N = 768
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, N, N, 1, 1, 1, N), views=[View((1, N, N, 1), (0, 1, N, 0), 0), View((1, 1, N, N, 1, 1, 1, N), (0, 0, 0, 1, 0, 0, 0, N), 0)]), hostbuf=GPUBuffer(shape=(N, N), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(1, 1, N, N, 1, 1, 1, N), views=[View((1, 1, N, N, 1, 1, 1, N), (0, 0, 1, 0, 0, 0, 0, N), 0)]), hostbuf=GPUBuffer(shape=(N, N), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (1, 1, N, N, 1, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (N, N))
    ii = []
    ii.append((Interventions.SHIFT, (1, 8, False)))
    ii.append((Interventions.SHIFT, (1, 8, False)))
    #ii.append((Interventions.UPCAST, (1, 4)))
    #ii.append((Interventions.UPCAST, (0, 4)))
    one(ast, ii, local_override=[8,8,1])
    #search(ast, ii) #, depth=0)
    exit(0)
  elif getenv("FASTCONV", 0):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(32, 1, 32, 32, 32, 64, 3, 3), views=[View((32, 1, 32, 32, 32, 64, 3, 3), (73984, 73984, 0, 34, 1, 1156, 34, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 64, 34, 34), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(32, 1, 32, 32, 32, 64, 3, 3), views=[View((32, 1, 32, 32, 32, 64, 3, 3), (0, 0, 576, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(32, 64, 3, 3), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (32, 1, 32, 32, 32, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (32, 32, 32, 32))
  elif getenv("BROKEN", 0):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(64, 1, 1, 1), views=[View((64, 1, 1, 1), (1, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(64,), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(64, 5, 32, 32), views=[View((64, 5, 32, 32), (5120, 1024, 32, 1), 0)]), hostbuf=GPUBuffer(shape=(64, 5, 32, 32), force_create=True))
    op0 = LazyOp(ReduceOps.SUM, (buf1,), (64, 1, 1, 1))
    buf2 = GPUBuffer(shape=ShapeTracker(shape=(64, 1, 1, 1), views=[View((64, 1, 1, 1), (0, 0, 0, 0), 0)]), hostbuf=GPUBuffer(shape=(1,), backing=np.array([0.001], dtype=np.float32)))
    op1 = LazyOp(BinaryOps.MUL, (op0,buf2,), None)
    op2 = LazyOp(BinaryOps.SUB, (buf0,op1,), None)
    ast = LazyOp(MovementOps.RESHAPE, (op2,), (64,))
  elif getenv("BROKEN3"):
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(5, 1, 128, 16, 16, 128, 3, 3), views=[View((5, 128, 18, 18), (32768, 256, 16, 1), -17), ZeroView((5, 128, 16, 16), ((0, 5), (0, 128), (-1, 17), (-1, 17))), View((5, 1, 128, 16, 16, 128, 3, 3), (41472, 41472, 0, 18, 1, 324, 18, 1), 0)]), hostbuf=GPUBuffer(shape=(5, 128, 16, 16), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(5, 1, 128, 16, 16, 128, 3, 3), views=[View((5, 1, 128, 16, 16, 128, 3, 3), (0, 0, 1152, 0, 0, 9, 3, 1), 0)]), hostbuf=GPUBuffer(shape=(128, 128, 3, 3), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (5, 1, 128, 16, 16, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (5, 128, 16, 16))
  else:
    # reduce
    buf0 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 32, 225, 225), (50176, 150528, 224, 1), 0), ZeroView((3, 32, 224, 224), ((0, 3), (0, 32), (0, 225), (0, 225))), View((3, 1, 32, 3, 3, 32, 112, 112), (1620000, 1620000, 0, 225, 1, 50625, 450, 2), 0)]), hostbuf=GPUBuffer(shape=(32, 3, 224, 224), force_create=True))
    buf1 = GPUBuffer(shape=ShapeTracker(shape=(3, 1, 32, 3, 3, 32, 112, 112), views=[View((3, 1, 32, 3, 3, 32, 112, 112), (0, 12845056, 401408, 0, 0, 12544, 112, 1), 0)]), hostbuf=GPUBuffer(shape=(1, 1, 32, 1, 1, 32, 112, 112), force_create=True))
    op0 = LazyOp(BinaryOps.MUL, (buf0,buf1,), None)
    op1 = LazyOp(ReduceOps.SUM, (op0,), (3, 1, 32, 3, 3, 1, 1, 1))
    ast = LazyOp(MovementOps.RESHAPE, (op1,), (3, 32, 3, 3))
  search(ast)
  #test_correctness(ast)
