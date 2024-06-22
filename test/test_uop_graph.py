import unittest
from typing import Callable
from test.helpers import TestUOps, compare_uop_tree, print_uop_tree
from tinygrad import dtypes, Variable
from tinygrad.dtype import PtrDType
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, UOp, constant_folder

class TestUOpGraph(TestUOps):
  # TODO: move to test.helpers
  def test_add_constant_fold(self):
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
    c0 = UOp(UOps.CONST, dtypes.int, arg=0)
    vc = UOp(UOps.ALU, dtypes.bool, (v, c0), BinaryOps.CMPNE)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    out = UOp(UOps.ALU, dtypes.float, (vc, c1, c1), TernaryOps.WHERE)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (bf, c1, c2), TernaryOps.WHERE)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    out = UOp(UOps.CAST, dtypes.int, (bf,))
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 0)

  def test_cast_vectorized_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=(0, True))
    idx = UOp(UOps.CONST, dtypes.int, arg=0)
    ld = UOp(UOps.LOAD, dtypes.float.vec(2), (d0, idx))
    cast = UOp(UOps.CAST, dtypes.float.vec(2), (ld,))
    x = UOp(UOps.GEP, dtypes.float, (cast, ), arg=0)
    alu = UOp(UOps.ALU, dtypes.float, (x, ), UnaryOps.SQRT)
    out = UOp(UOps.STORE, dtypes.float, (d0, idx, alu))
    g = UOpGraph([out])
    self.assertEqual(len([x for x in g.uops if x.op is UOps.CAST]), 0)

  def test_depth_2_const_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
    c2 = UOp(UOps.CONST, dtypes.int, arg=2)
    c4 = UOp(UOps.CONST, dtypes.int, arg=4)
    vc = UOp(UOps.ALU, dtypes.int, (v, c2), BinaryOps.ADD)
    out = UOp(UOps.ALU, dtypes.int, (vc, c4), BinaryOps.ADD)
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 3)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.ALU)
    self.assertEqual(out.arg, BinaryOps.ADD)
    self.assertEqual(out.src[1].op, UOps.CONST)
    self.assertEqual(out.src[1].arg, 6)

  def test_fold_gated_load(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
    glbl1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (1, False))
    glbl2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (2, False))
    idx = UOp.const(dtypes.int, 0)
    ld0 = UOp(UOps.LOAD, dtypes.int, (glbl1, idx, UOp.const(dtypes.bool, False), UOp.const(dtypes.int, 2)))
    ld1 = UOp(UOps.LOAD, dtypes.int, (glbl2, idx, UOp.const(dtypes.bool, True), UOp.const(dtypes.int, 3)))
    uops = UOpGraph([UOp(UOps.STORE, None, (glbl0, idx, ld0+ld1))])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    self.assert_equiv_uops(ld0, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    self.assert_equiv_uops(ld1, UOp.load(glbl2, idx, dtype=dtypes.int))

  def test_fold_gated_load_local(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
    smem = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.int), (), ("temp", 1))
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), (0, "lidx1", 16))
    st = UOp(UOps.STORE, None, (smem, lidx, UOp.load(glbl0, lidx, dtype=dtypes.int)))
    barrier = UOp(UOps.BARRIER, None, (st, ))
    ld0 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+1, UOp.const(dtypes.bool, False), UOp.const(dtypes.int, 2), barrier))
    ld1 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+2, UOp.const(dtypes.bool, True), UOp.const(dtypes.int, 3), barrier))
    uops = UOpGraph([UOp(UOps.STORE, None, (glbl0, lidx, ld0+ld1))])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    self.assert_equiv_uops(ld0, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    self.assert_equiv_uops(ld1, UOp.load(smem, lidx+2, barrier, dtype=dtypes.int))

  def test_fold_gated_store(self):
    glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 0)
    val = UOp.const(dtypes.int, 42)
    st0 = UOp(UOps.STORE, None, (glbl, idx0, val, UOp.const(dtypes.bool, False)))
    st1 = UOp(UOps.STORE, None, (glbl, idx1, val, UOp.const(dtypes.bool, True)))
    uops = UOpGraph([st0, st1])
    # only the second store happens
    self.assertEqual(len(uops.uops), 4)
    self.assert_equiv_uops(uops[-1], UOp.store(glbl, idx1, val))

def create_uop_linearize_and_compare_bottomup_topdown(uop_factory: Callable[[], UOp]):
  def attach_sink_and_create_graph():
    uop_output = uop_factory()
    sink = UOp(UOps.SINK, None, (uop_output,))
    graph = UOpGraph([uop_output])
    graph.nodes = {}
    return sink, graph
  sink1, graph1 = attach_sink_and_create_graph()
  sink2, graph2 = attach_sink_and_create_graph()
  sink1_rewritten = graph1.graph_rewrite(sink1, constant_folder)
  sink2_rewritten = graph2.graph_rewrite_bottomup_no_backtrack(sink2, constant_folder)
  result, reason = compare_uop_tree(sink1_rewritten, sink2_rewritten)
  reason_and_tree = reason + '\nUOp1 (topdown): \n' + print_uop_tree(sink1_rewritten, _print=False)
  reason_and_tree += 'UOp2 (bottomup): \n' + print_uop_tree(sink2_rewritten, _print=False)
  return result, reason_and_tree

class TestBottomupVsTopdownRewrite(TestUOps):
  def setup_and_assert(self, uop_factory: Callable[[], UOp]):
    result, reason = create_uop_linearize_and_compare_bottomup_topdown(uop_factory)
    self.assertTrue(result, reason)

  def test_add_const(self):
    self.setup_and_assert(lambda: UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
      UOp(UOps.CONST, dtypes.float, arg=1.0),
      UOp(UOps.CONST, dtypes.float, arg=2.0),
    )))

  def test_nested_add(self):
    self.setup_and_assert(lambda: UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
      UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
        UOp(UOps.CONST, dtypes.float, arg=1.0),
        UOp(UOps.CONST, dtypes.float, arg=2.0),
      )),
      UOp(UOps.CONST, dtypes.float, arg=3.0),
    )))
  
  @unittest.skip("Skip until patterns rule return node that cannot further be simplified")
  def test_arange(self):
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
    self.setup_and_assert(factory)  

    

if __name__ == '__main__':
  unittest.main(verbosity=2)
