import unittest
from typing import Callable
from test.helpers import TestUOps, compare_uop_tree, print_uop_tree, NodeVisitCounter
from tinygrad import dtypes, Variable
from tinygrad.dtype import PtrDType
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, UOp, constant_folder

def factory():
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

def run(mode):
  print(f"Running with {mode}")
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


run('bottomup')
run('topdown')