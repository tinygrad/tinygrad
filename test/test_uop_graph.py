import unittest
from typing import Callable
from test.helpers import TestUOps, compare_uop_tree, print_uop_tree
from tinygrad import dtypes, Variable
from tinygrad.dtype import PtrDType
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps
from tinygrad.codegen.uops import UOpGraph, UOps, UOp, constant_folder, UPat, sum_collapse, PatternMatcher, exec_alu, loop_collapse

class TestUOpGraph(TestUOps):
  # TODO: move to test.helpers
  def test_add_constant_fold(self):
    def factory():
      c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
      c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
      out = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
      return out
    out = factory()
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 3.0)
    self.setup_and_assert_uop_graph_equal(factory)

  def test_where_same_fold(self):
    def factory():
      v = UOp(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
      c0 = UOp(UOps.CONST, dtypes.int, arg=0)
      vc = UOp(UOps.ALU, dtypes.bool, (v, c0), BinaryOps.CMPNE)
      c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
      out = UOp(UOps.ALU, dtypes.float, (vc, c1, c1), TernaryOps.WHERE)
      return out
    out = factory()
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 1.0)
    self.setup_and_assert_uop_graph_equal(factory)

  def test_where_const_fold(self):
    def factory():
      bf = UOp(UOps.CONST, dtypes.bool, arg=False)
      c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
      c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
      out = UOp(UOps.ALU, dtypes.float, (bf, c1, c2), TernaryOps.WHERE)
      return out
    out = factory()
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 2.0)
    self.setup_and_assert_uop_graph_equal(factory)

  def test_const_cast(self):
    def factory():
      bf = UOp(UOps.CONST, dtypes.bool, arg=False)
      out = UOp(UOps.CAST, dtypes.int, (bf,))
      return out
    out = factory()
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 1)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 0)
    self.setup_and_assert_uop_graph_equal(factory)

  def test_cast_vectorized_fold(self):
    def factory():
      d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=(0, True))
      idx = UOp(UOps.CONST, dtypes.int, arg=0)
      ld = UOp(UOps.LOAD, dtypes.float.vec(2), (d0, idx))
      cast = UOp(UOps.CAST, dtypes.float.vec(2), (ld,))
      x = UOp(UOps.GEP, dtypes.float, (cast, ), arg=0)
      alu = UOp(UOps.ALU, dtypes.float, (x, ), UnaryOps.SQRT)
      out = UOp(UOps.STORE, dtypes.float, (d0, idx, alu))
      return out
    out = factory()
    g = UOpGraph([out])
    self.assertEqual(len([x for x in g.uops if x.op is UOps.CAST]), 0)
    self.setup_and_assert_uop_graph_equal(factory)

  def test_depth_2_const_fold(self):
    def factory():
      v = UOp(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1))
      c2 = UOp(UOps.CONST, dtypes.int, arg=2)
      c4 = UOp(UOps.CONST, dtypes.int, arg=4)
      vc = UOp(UOps.ALU, dtypes.int, (v, c2), BinaryOps.ADD)
      out = UOp(UOps.ALU, dtypes.int, (vc, c4), BinaryOps.ADD)
      return out
    out = factory()
    g = UOpGraph([out])
    self.assertEqual(len(g.uops), 3)
    out = g.uops[-1]
    self.assertEqual(out.op, UOps.ALU)
    self.assertEqual(out.arg, BinaryOps.ADD)
    self.assertEqual(out.src[1].op, UOps.CONST)
    self.assertEqual(out.src[1].arg, 6)
    self.setup_and_assert_uop_graph_equal(factory)

  def test_fold_gated_load(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
    glbl1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (1, False))
    glbl2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (2, False))
    idx = UOp.const(dtypes.int, 0)
    ld0 = UOp(UOps.LOAD, dtypes.int, (glbl1, idx, UOp.const(dtypes.bool, False), UOp.const(dtypes.int, 2)))
    ld1 = UOp(UOps.LOAD, dtypes.int, (glbl2, idx, UOp.const(dtypes.bool, True), UOp.const(dtypes.int, 3)))
    store = UOp(UOps.STORE, None, (glbl0, idx, ld0+ld1))
    uops = UOpGraph([store])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    self.assert_equiv_uops(ld0, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    self.assert_equiv_uops(ld1, UOp.load(glbl2, idx, dtype=dtypes.int))

    def factory():
      glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
      glbl1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (1, False))
      glbl2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (2, False))
      idx = UOp.const(dtypes.int, 0)
      ld0 = UOp(UOps.LOAD, dtypes.int, (glbl1, idx, UOp.const(dtypes.bool, False), UOp.const(dtypes.int, 2)))
      ld1 = UOp(UOps.LOAD, dtypes.int, (glbl2, idx, UOp.const(dtypes.bool, True), UOp.const(dtypes.int, 3)))
      store = UOp(UOps.STORE, None, (glbl0, idx, ld0+ld1))
      return store
    self.setup_and_assert_uop_graph_equal(factory)

  def test_fold_gated_load_local(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
    smem = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.int), (), ("temp", 1))
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), (0, "lidx1", 16))
    st = UOp(UOps.STORE, None, (smem, lidx, UOp.load(glbl0, lidx, dtype=dtypes.int)))
    barrier = UOp(UOps.BARRIER, None, (st, ))
    ld0 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+1, UOp.const(dtypes.bool, False), UOp.const(dtypes.int, 2), barrier))
    ld1 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+2, UOp.const(dtypes.bool, True), UOp.const(dtypes.int, 3), barrier))
    store = UOp(UOps.STORE, None, (glbl0, lidx, ld0+ld1))
    uops = UOpGraph([store])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    self.assert_equiv_uops(ld0, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    self.assert_equiv_uops(ld1, UOp.load(smem, lidx+2, barrier, dtype=dtypes.int))
    def factory():
      glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), (0, True))
      smem = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.int), (), ("temp", 1))
      lidx = UOp(UOps.SPECIAL, dtypes.int, (), (0, "lidx1", 16))
      st = UOp(UOps.STORE, None, (smem, lidx, UOp.load(glbl0, lidx, dtype=dtypes.int)))
      barrier = UOp(UOps.BARRIER, None, (st, ))
      ld0 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+1, UOp.const(dtypes.bool, False), UOp.const(dtypes.int, 2), barrier))
      ld1 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+2, UOp.const(dtypes.bool, True), UOp.const(dtypes.int, 3), barrier))
      store = UOp(UOps.STORE, None, (glbl0, lidx, ld0+ld1))
      return store
    self.setup_and_assert_uop_graph_equal(factory)

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


class TestBottomupVsTopdownRewrite(TestUOps):  
  def test_sum_collapse(self):
    def factory():
      global_buffer0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=(0, True))
      global_buffer1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=(0, False))
      const_0 = UOp(UOps.CONST, dtypes.float, (), 0.0)
      const_1 = UOp(UOps.CONST, dtypes.float, (), 1.0)

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
    self.setup_and_assert_uop_graph_equal(factory)  
    
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
    self.setup_and_assert_uop_graph_equal(factory)  

  def test_pattern_that_can_be_inlined(self):
    sa = self.setup_and_assert_uop_graph_equal

    # Long list x + x + x + x
    sa(lambda: UOp(UOps.CONST, dtypes.float, arg=2.0) + 2 + 2 + 2 + 2)

    sa(lambda: UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
      UOp(UOps.CONST, dtypes.float, arg=1.0),
      UOp(UOps.CONST, dtypes.float, arg=2.0),
    )))
    sa(lambda: UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
      UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
        UOp(UOps.CONST, dtypes.float, arg=1.0),
        UOp(UOps.CONST, dtypes.float, arg=2.0),
      )),
      UOp(UOps.CONST, dtypes.float, arg=3.0),
    )))
    sa(lambda: UOp(UOps.ALU, dtypes.float, arg=TernaryOps.WHERE, src=(
      UOp(UOps.CONST, dtypes.bool, arg=False),
      UOp(UOps.CONST, dtypes.float, arg=1.0),
      UOp(UOps.CONST, dtypes.float, arg=2.0),
    )))
    sa(lambda: UOp(UOps.ALU, dtypes.float, (
      UOp(UOps.CONST, dtypes.float, arg=1.0),
      UOp(UOps.CONST, dtypes.float, arg=2.0)
    ), BinaryOps.ADD))
    sa(lambda: UOp(UOps.ALU, dtypes.float, (
      UOp(UOps.ALU, dtypes.bool, (
        UOp(UOps.DEFINE_VAR, dtypes.int, arg=Variable('tmp', 0, 1)),
        UOp(UOps.CONST, dtypes.int, arg=0)
      ), BinaryOps.CMPNE),
      UOp(UOps.CONST, dtypes.float, arg=1.0),
      UOp(UOps.CONST, dtypes.float, arg=1.0)
    ), TernaryOps.WHERE))
    sa(lambda: UOp(UOps.ALU, dtypes.float, (
      UOp(UOps.CONST, dtypes.bool, arg=False), 
      UOp(UOps.CONST, dtypes.float, arg=1.0), 
      UOp(UOps.CONST, dtypes.float, arg=2.0)
    ), TernaryOps.WHERE))
    sa(lambda: UOp(UOps.CAST, dtypes.int, (
      UOp(UOps.CONST, dtypes.bool, arg=False),
    )))
    sa(lambda: -(-UOp.var('x')))
    sa(lambda: -1 * ( UOp.var('x')))
    sa(lambda: UOp.var().lt(UOp.const(dtypes.bool, False)))
    sa(lambda: (UOp.const(dtypes.bool, True).lt(UOp.var())))
    sa(lambda: UOp.alu(TernaryOps.WHERE, UOp.var(), UOp.var("val"), UOp.var("val")))
    sa(lambda: UOp.alu(TernaryOps.WHERE, UOp.cvar('gate'), UOp.var('c0'), UOp.var('c1')))
    sa(lambda: UOp.var('x') + 0)
    sa(lambda: UOp.var('x') - 0)
    sa(lambda: UOp.var('x') * 1)
    sa(lambda: UOp.var('x') // 1)
    sa(lambda: UOp.var('x') // -1)
    sa(lambda: UOp.var('x') * 0)
    sa(lambda: UOp.var('x') - UOp.var('x'))
    sa(lambda: UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.load(UOp.var("buf"), UOp.var("idx"))))
    sa(lambda: (UOp.var('x') + UOp.cvar('c1')) + UOp.cvar('c2'))
    sa(lambda: UOp.var('x') - UOp.cvar('c1')) + UOp.cvar('c2')
    sa(lambda: (UOp.var("x") * UOp.cvar("c1")) * UOp.cvar("c2"))
    sa(lambda: UOp.var("x") % UOp.const(None, 1))
    sa(lambda: UOp.var("x") * UOp.cvar("c0") + UOp.var("x") * UOp.cvar("c1"))
    sa(lambda: (UOp.var("x") * UOp.cvar("c0")) // UOp.cvar("c0"))
    sa(lambda: (UOp.var("x") // UOp.cvar("c0")) // UOp.cvar("c1"))
    sa(lambda: (UOp.cvar("c0") + UOp.var("x")).lt(UOp.cvar("c1")))
    sa(lambda: UOp.var("x") + UOp.var("x") * UOp.cvar("c0"))
    sa(lambda: UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.alu(TernaryOps.WHERE, UOp.var("gate"), UOp.var("alt"), UOp.load(UOp.var("buf"), UOp.var("idx")))))
    sa(lambda: UOp.store(UOp.var("buf"), UOp.var("idx"), UOp(UOps.CAST, src=tuple(UOp(UOps.GEP, arg=i, src=(UOp.var("val"),)) for i in range(4)))))
    sa(lambda: UOp.store(UOp.var("buf"), UOp.var("idx"), UOp(UOps.CAST, src=tuple(UOp(UOps.GEP, arg=i, src=(UOp.var("val"),)) for i in range(2)))))
    sa(lambda: UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(None, 1), UOp.cvar("var")))
    sa(lambda: UOp.load(UOp.var("buf"), UOp.var("idx"), UOp.const(None, 1), UOp.cvar("var"), UOp.var("barrier")))
    sa(lambda: UOp.load(UOp.var(), UOp.var(), UOp.const(None, 0), UOp.cvar("var")))
    sa(lambda: UOp.load(UOp.var(), UOp.var(), UOp.const(None, 0), UOp.cvar("var"), UOp.var()))
    sa(lambda: UOp.store(UOp.var("buf"), UOp.var("idx"), UOp.var("val"), UOp.const(None, 1)))
    sa(lambda: UOp.store(UOp.var(), UOp.var(), UOp.var(), UOp.const(None, 0)))
    sa(lambda: UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
      UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
        UOp(UOps.CONST, dtypes.float, arg=0.0),
        UOp(UOps.CONST, dtypes.float, arg=0.0),
      )),
      UOp(UOps.CONST, dtypes.float, arg=float('inf')),
    )))

  

if __name__ == '__main__':
  unittest.main(verbosity=2)
