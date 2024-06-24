import unittest
from typing import Callable
from test.helpers import TestUOps, compare_uop_tree, print_uop_tree, NodeVisitCounter
from tinygrad import dtypes, Variable
from tinygrad.dtype import PtrDType
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, UOp, constant_folder

def arange():
  const_0 = UOp(UOps.CONST, dtypes.float, (), 0.0)
  const_neg_1 = UOp(UOps.CONST, dtypes.float, (), -1.0)
  const_1 = UOp(UOps.CONST, dtypes.float, (), 1.0)
  const_2 = UOp(UOps.CONST, dtypes.float, (), 2.0)
  const_neg_2 = UOp(UOps.CONST, dtypes.float, (), -2.0)
  const_3 = UOp(UOps.CONST, dtypes.float, (), 3.0)
  _special = UOp(UOps.SPECIAL, dtypes.float, (), (0, 'gidx0', 4))
  _global = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), (0, True))
  _range = UOp(UOps.RANGE, dtypes.float, (const_0,const_2), (2, 0))
  _acc = UOp(UOps.DEFINE_ACC, dtypes.float, (_range,), (0,0,0))
  special_mul_one = UOp(UOps.ALU, dtypes.float, (_special, const_neg_1), BinaryOps.MUL) # rewrite
  range_mul_one = UOp(UOps.ALU, dtypes.float, (_range, const_neg_1), BinaryOps.MUL) # rewrite
  add_two_muls = UOp(UOps.ALU, dtypes.float, (special_mul_one, range_mul_one), BinaryOps.ADD) # rewrite
  cmplt = UOp(UOps.ALU, dtypes.bool, (add_two_muls, const_neg_2), BinaryOps.CMPLT)
  _where = UOp(UOps.ALU, dtypes.float, (cmplt, const_3, const_0), TernaryOps.WHERE)
  add_where_acc = UOp(UOps.ALU, dtypes.float, (_where, _acc), BinaryOps.ADD)
  phi = UOp(UOps.PHI, dtypes.float, (_acc, add_where_acc))
  store = UOp(UOps.STORE, None, (_global, _special, phi))
  return store

def sum_collapse():
  global_buffer0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=(0, True))
  global_buffer1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=(0, False))
  const_0 = UOp(UOps.CONST, dtypes.float, (), 0.0)
  const_1 = UOp(UOps.CONST, dtypes.float, (), 1.0)
  const_2 = UOp(UOps.CONST, dtypes.float, (), 2.0)
  const_42 = UOp(UOps.CONST, dtypes.float32, (), 42.0)

  loop = UOp(UOps.RANGE, dtypes.float, (
      UOp(UOps.CONST, dtypes.int, arg=0),
      UOp(UOps.CONST, dtypes.int, arg=10)
  ), (2.0,0.0))
  acc = UOp(UOps.DEFINE_ACC, dtypes.float, (loop,), (0.0,0.0,0.0))
  alu = UOp(UOps.ALU, dtypes.float, (
      UOp(UOps.LOAD, dtypes.float, (global_buffer1, const_0)),
      acc,
    ), BinaryOps.ADD)
  phi = UOp(UOps.PHI, dtypes.float, (
      acc,
      alu,
  ))
  store = UOp(UOps.STORE, None, (global_buffer0, const_1, phi))
  return store

def run(mode, factory):
  print(f"Running with {mode} for {factory.__name__}")
  counter = NodeVisitCounter()
  uop = factory()
  g = UOpGraph([uop])
  g.nodes = {}
  if mode == 'bottomup':
    rewritten = g.graph_rewrite_bottomup_no_backtrack(uop, constant_folder, counter)
  elif mode == 'topdown':
    rewritten = g.graph_rewrite(uop, constant_folder, counter)
  print("Total visit", counter.total)
  print_uop_tree(rewritten, counter)


run('bottomup', arange)
run('topdown', arange)
run('bottomup', sum_collapse)
run('topdown', sum_collapse)