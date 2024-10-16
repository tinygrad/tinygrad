from typing import List
import unittest, time
from test.helpers import assert_equiv_uops
from tinygrad import dtypes, Device
from tinygrad.dtype import PtrDType
from tinygrad.helpers import DEBUG
from tinygrad.ops import BinaryOps, TernaryOps, UnaryOps, UOps, UOp, KernelInfo
from tinygrad.ops import UPat, PatternMatcher
from tinygrad.codegen.lowerer import rewrite_shapetracker_with_index
from tinygrad.codegen.uopgraph import full_graph_rewrite, graph_rewrite, expander, reducer, sym, float4_folding
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.shape.shapetracker import ShapeTracker, View

simple_pm = PatternMatcher([
  (UPat.cvar('x', dtypes.int), lambda x: UOp.const(dtypes.float, 1.0) + UOp.const(dtypes.float, 2.0)),
  (UPat.cvar('x') + UPat.cvar('y'), lambda x,y: UOp.const(dtypes.float, x.arg+y.arg)),
  (UPat.cvar('x') * UPat.cvar('y') * UPat.cvar('z'), lambda x,y,z: UOp.const(dtypes.float, x.arg*y.arg*z.arg)),
  ((UPat.var('x') + UPat.cvar('c1')) + UPat.cvar('c2'), lambda x,c1,c2: x + (c1.arg+c2.arg)),
])

def to_uops_list(u:List[UOp]) -> List[UOp]: return linearize_uop(full_graph_rewrite(UOp.sink(*u)))

class TestGraphRewriteEfficiency(unittest.TestCase):
  def test_create_many_uops(self):
    c1 = UOp.const(dtypes.int, 1)
    c2 = UOp.const(dtypes.int, 2)
    st = time.perf_counter()
    uops = [UOp(UOps.ALU, dtypes.int, (c1, c2), BinaryOps.ADD) for _ in range(10000)]
    et = time.perf_counter() - st
    print(f"created {len(uops)} uops in {et*1000:.2f} ms")

  def test_expand_rewrite(self):
    sink = UOp(UOps.SINK, dtypes.void, arg=KernelInfo(local_dims=2, upcasted=4, dont_use_locals=False), src=(
      UOp(UOps.STORE, dtypes.void, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0, src=()),
        UOp(UOps.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(2, 4, 64, 8, 16, 1, 1, 3, 3, 4, 1),
                                                                  strides=(1179648, 9216, 1, 147456, 576, 0, 0, 64, 192, 36864, 0),
                                                                  offset=0, mask=None, contiguous=False),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (5, 6, 10)), src=(
          UOp(UOps.CAST, dtypes.float, arg=None, src=(
            UOp(UOps.ALU, dtypes.half, arg=BinaryOps.MUL, src=(
              UOp(UOps.LOAD, dtypes.half, arg=None, src=(
                UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.half), arg=1, src=()),
                UOp(UOps.VIEW, dtypes.void, arg=ShapeTracker(views=(
                  View(shape=(1, 1024, 1, 64, 4, 17, 4, 17), strides=(0, 14400, 0, 225, 0, 15, 0, 1), offset=-16,
                       mask=((0, 1), (0, 1024), (0, 1), (0, 64), (0, 4), (1, 16), (0, 4), (1, 16)), contiguous=False),
                  View(shape=(2, 4, 64, 8, 16, 16, 15, 3, 3, 4, 15), strides=(0, 73984, 4734976, 0, 4624, 295936, 68, 18, 1224, 0, 1), offset=0,
                       mask=None, contiguous=False))), src=()),)),
              UOp(UOps.LOAD, dtypes.half, arg=None, src=(
                UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.half), arg=2, src=()),
                UOp(UOps.VIEW, dtypes.void, arg=ShapeTracker(views=(
                  View(shape=(2, 4, 64, 8, 16, 16, 15, 3, 3, 4, 15), strides=(7200, 0, 230400, 900, 0, 14400, 15, 0, 0, 225, 1), offset=0,
                       mask=None, contiguous=False),)), src=()),)),)),)),)),)),))
    lower_sink = rewrite_shapetracker_with_index(sink, Device[Device.DEFAULT].renderer)
    cnt = [0]
    old_init = UOp.__init__
    def uop_hook(self, *args, **kwargs):
      cnt[0] += 1
      old_init(self, *args, **kwargs)
    UOp.__init__ = uop_hook
    st = time.perf_counter()
    new_sink = full_graph_rewrite(lower_sink)
    et = time.perf_counter() - st
    UOp.__init__ = old_init
    print(f"rewrote in {et*1000:.2f} ms, from {len(lower_sink.sparents)} -> {len(new_sink.sparents)}, creating {cnt[0]} uops")

class TestGraphRewriteConst(unittest.TestCase):
  def test_gep_const(self):
    v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    v2 = v1.gep(1)
    ret = graph_rewrite(v2, sym)
    self.assertEqual(ret.dtype, dtypes.int)
    self.assertEqual(ret.arg, 1)

  def test_gep_const_single(self):
    v1 = UOp.const(dtypes.int.vec(3), 4)
    v2 = v1.gep(1)
    ret = graph_rewrite(v2, sym)
    self.assertEqual(ret.dtype, dtypes.int)
    self.assertEqual(ret.arg, 4)

  def test_add_const(self):
    v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    v2 = UOp.const(dtypes.int.vec(3), (5,6,7))
    ret = graph_rewrite(v1+v2, sym)
    self.assertEqual(ret.op, UOps.VCONST)
    self.assertEqual(ret.dtype, dtypes.int.vec(3))
    self.assertEqual(ret.arg, (5,7,9))

  def test_add_const_lose_v(self):
    v1 = UOp.const(dtypes.int.vec(3), (0,1,2))
    v2 = UOp.const(dtypes.int.vec(3), (2,1,0))
    ret = graph_rewrite(v1+v2, sym)
    self.assertEqual(ret.op, UOps.CONST)
    self.assertEqual(ret.dtype, dtypes.int.vec(3))
    self.assertEqual(ret.arg, 2)

class TestGraphRewrite(unittest.TestCase):
  def test_dedup(self):
    v1 = UOp(UOps.DEFINE_VAR, dtypes.float)
    v2 = UOp(UOps.DEFINE_VAR, dtypes.float)
    nout = graph_rewrite(v1+v2, PatternMatcher([]))
    self.assertIs(nout.src[0], nout.src[1])

  # NOTE: this shows why we can't have a UOp in arg
  @unittest.expectedFailure
  def test_no_dedup_args(self):
    a1 = UOp(UOps.DEFINE_VAR, dtypes.int, (), ("a1", UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 11)))
    a2 = UOp(UOps.DEFINE_VAR, dtypes.int, (), ("a2", UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 11)))
    sink = a1.sink(a2)
    define_vars = [x for x in graph_rewrite(sink, PatternMatcher([])).sparents if x.op is UOps.DEFINE_VAR]
    self.assertEqual(len(define_vars), 1)

  def test_simple(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    nout = graph_rewrite(c1+c2, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 3.0)

  def test_depth_2_late(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite(c1*c2*(c3+c3), simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 12.0)

  def test_double(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite(c1+c2+c3, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 6.0)

  def test_triple(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    c4 = UOp.const(dtypes.float, 4.0)
    nout = graph_rewrite(c1+c2+c3+c4, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 10.0)

  def test_diamond(self):
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    c3 = UOp.const(dtypes.float, 3.0)
    nout = graph_rewrite((c1+c2)+(c1+c3), simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 7.0)

  def test_magic_4(self):
    c1 = UOp.const(dtypes.int, 4.0)
    nout = graph_rewrite(c1, simple_pm)
    self.assertEqual(nout.op, UOps.CONST)
    self.assertEqual(nout.arg, 3.0)

  def test_depth_2_fold(self):
    v = UOp(UOps.DEFINE_VAR, dtypes.float)
    c1 = UOp.const(dtypes.float, 1.0)
    c2 = UOp.const(dtypes.float, 2.0)
    nout = graph_rewrite(v+c1+c2, simple_pm)
    self.assertEqual(nout.op, UOps.ALU)
    self.assertEqual(nout.src[0].op, UOps.DEFINE_VAR)
    self.assertEqual(nout.src[1].op, UOps.CONST)
    self.assertEqual(nout.src[1].arg, 3.0)

  def test_consts_go_last(self):
    a = UOp.variable('a', 0, 1)
    b = UOp.variable('b', 0, 1)
    c = UOp.variable('c', 0, 1)
    d = UOp.variable('d', 0, 1)
    outs = [2+a, 2+a+d+3+b+c+4, UOp(UOps.ALU, a.dtype, src=(a.const_like(2), a), arg=BinaryOps.ADD), (4+d)+c+(2+a)+b]
    for out in outs:
      sink = graph_rewrite(out, sym)
      print(sink)
      self.assertEqual(sink.op, UOps.ALU)
      self.assertEqual(sink.src[1].op, UOps.CONST)
      self.assertEqual(len([x for x in sink.sparents if x.op is UOps.CONST]), 1)

class TestUOpGraph(unittest.TestCase):
  def test_add_constant_fold(self):
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (c1, c2), BinaryOps.ADD)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 3.0)

  def test_where_same_fold(self):
    v = UOp.variable('tmp', 0, 1)
    c0 = UOp(UOps.CONST, dtypes.int, arg=0)
    vc = UOp(UOps.ALU, dtypes.bool, (v, c0), BinaryOps.CMPNE)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    out = UOp(UOps.ALU, dtypes.float, (vc, c1, c1), TernaryOps.WHERE)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 1.0)

  def test_where_const_fold(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    c1 = UOp(UOps.CONST, dtypes.float, arg=1.0)
    c2 = UOp(UOps.CONST, dtypes.float, arg=2.0)
    out = UOp(UOps.ALU, dtypes.float, (bf, c1, c2), TernaryOps.WHERE)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 2.0)

  def test_const_cast(self):
    bf = UOp(UOps.CONST, dtypes.bool, arg=False)
    out = UOp(UOps.CAST, dtypes.int, (bf,))
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 1)
    out = uops[-1]
    self.assertEqual(out.op, UOps.CONST)
    self.assertEqual(out.arg, 0)

  @unittest.skip("this test isn't valid uops")
  def test_noop_vectorize_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(UOps.LOAD, dtypes.float.vec(2), (d0, idx))
    vec = UOp(UOps.VECTORIZE, dtypes.float.vec(2), (ld,))
    x = UOp(UOps.GEP, dtypes.float, (vec, ), arg=0)
    alu = UOp(UOps.ALU, dtypes.float, (x, ), UnaryOps.SQRT)
    out = UOp(UOps.STORE, dtypes.void, (d0, idx, alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is UOps.VECTORIZE]), 0)

  def test_gep_vec_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    d1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 1)
    d2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 2)
    idx = UOp.const(dtypes.int, 0)
    def _test_vec(geps, count=4):
      vec = UOp(UOps.VECTORIZE, dtypes.float.vec(count), geps)
      out = UOp(UOps.STORE, dtypes.void, (d0, idx, vec))
      uops = to_uops_list([out])
      if DEBUG >= 4:
        from tinygrad import Device
        print(Device[Device.DEFAULT].renderer.render("test", uops))
      return uops[-1].src[-1]

    # possible
    val = UOp(UOps.LOAD, dtypes.float.vec(4), (d1, idx))
    xyzw = tuple(UOp(UOps.GEP, dtypes.float, (val,), (i,)) for i in range(4))
    assert_equiv_uops(_test_vec(xyzw), val)

    # unaligned
    val = UOp(UOps.LOAD, dtypes.float.vec(4), (d1, idx))
    wzyx = tuple(UOp(UOps.GEP, dtypes.float, (val,), (i,)) for i in reversed(range(4)))
    self.assertIs(_test_vec(wzyx).op, UOps.VECTORIZE)

    # different_size
    val = UOp(UOps.LOAD, dtypes.float.vec(2), (d1, idx))
    xy = tuple(UOp(UOps.GEP, dtypes.float, (val, ), (i,)) for i in range(2))
    self.assertIs(_test_vec(xy+xy).op, UOps.VECTORIZE)
    val = UOp(UOps.LOAD, dtypes.float.vec(4), (d1, idx))
    xy = tuple(UOp(UOps.GEP, dtypes.float, (val, ), (i,)) for i in range(2))
    self.assertIs(_test_vec(xy, count=2).op, UOps.VECTORIZE)

    # different vals
    val1 = UOp(UOps.LOAD, dtypes.float.vec(2), (d1, idx))
    val2 = UOp(UOps.LOAD, dtypes.float.vec(2), (d2, idx))
    xy1 = tuple(UOp(UOps.GEP, dtypes.float, (val1, ), (i,)) for i in range(2))
    xy2 = tuple(UOp(UOps.GEP, dtypes.float, (val2, ), (i,)) for i in range(2))
    self.assertIs(_test_vec(xy1+xy2).op, UOps.VECTORIZE)

  def test_gep_vec_const_fold(self):
    for vec_size in [2, 4, 8]:
      consts = [UOp.const(dtypes.float, float(i)) for i in range(vec_size)]
      vec = UOp(UOps.VECTORIZE, dtypes.float.vec(vec_size), tuple(consts))
      uops = to_uops_list([UOp(UOps.GEP, dtypes.float, (vec,), (i,)) for i in range(vec_size)])
      for uop, const in zip(uops, consts):
        assert_equiv_uops(uop, const)

  def test_wmma_vectorize_fold(self):
    for i in [2, 4, 8]:
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i), tuple(UOp.const(dtypes.half, 0.0) for _ in range(i)))
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i))
      acc = UOp.variable('acc', 0, 1, dtypes.half.vec(i))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = to_uops_list([wmma])
      assert_equiv_uops(uops[0], acc)
      self.assertEqual(len(uops), 1)

    for i in [2, 4, 8]:
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i))
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i), tuple(UOp.const(dtypes.half, 0.0) for _ in range(i)))
      acc = UOp.variable('acc', 0, 1, dtypes.half.vec(i))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      assert_equiv_uops(uops[0], acc)
      self.assertEqual(len(uops), 1)

  @unittest.skip("wmma is wrong here, it needs an arg")
  def test_wmma_vectorize_no_fold(self):
    for i in [4, 8]:
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 0.0) for _ in range(i//2)) +
                tuple(UOp(UOps.DEFINE_VAR, dtypes.half, arg=(f'tmp{j}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1))) for j in range(i//2)))
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = to_uops_list([wmma])
      assert_equiv_uops(uops[-1], wmma)

    for i in [4, 8]:
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 0.0) for _ in range(i//2)) +
                tuple(UOp(UOps.DEFINE_VAR, dtypes.half, arg=(f'tmp{j}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1))) for j in range(i//2)))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      assert_equiv_uops(uops[-1], wmma)

    for i in [2, 4, 8]:
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (vec, var, acc))
      uops = to_uops_list([wmma])
      assert_equiv_uops(uops[-1], wmma)

    for i in [2, 4, 8]:
      var = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=(f'tmp{i}', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      vec = UOp(UOps.VECTORIZE, dtypes.half.vec(i),
                tuple(UOp.const(dtypes.half, 1.0 if j == 0 else 0.0) for j in range(i)))
      acc = UOp(UOps.DEFINE_VAR, dtypes.half.vec(i), arg=('acc', UOp.const(dtypes.half, 0), UOp.const(dtypes.half, 1)))
      wmma = UOp(UOps.WMMA, dtypes.half.vec(i), (var, vec, acc))
      uops = to_uops_list([wmma])
      assert_equiv_uops(uops[-1], wmma)

  def test_cast_alu_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.bool), arg=0)
    d1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(UOps.LOAD, dtypes.int, (d1, idx))
    alu = ld.lt(1).cast(dtypes.bool)
    out = UOp(UOps.STORE, dtypes.void, (d0, idx, alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is UOps.CAST]), 0)

  def test_double_cast_fold(self):
    d0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0)
    d1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(UOps.LOAD, dtypes.int, (d1, idx))
    alu = ld.cast(dtypes.float).cast(dtypes.float)
    out = UOp(UOps.STORE, dtypes.void, (d0, idx, alu))
    uops = to_uops_list([out])
    self.assertEqual(len([x for x in uops if x.op is UOps.CAST]), 1)

  def test_depth_2_const_fold(self):
    v = UOp.variable("tmp", 0, 1)
    c2 = UOp(UOps.CONST, dtypes.int, arg=2)
    c4 = UOp(UOps.CONST, dtypes.int, arg=4)
    vc = UOp(UOps.ALU, dtypes.int, (v, c2), BinaryOps.ADD)
    out = UOp(UOps.ALU, dtypes.int, (vc, c4), BinaryOps.ADD)
    uops = to_uops_list([out])
    self.assertEqual(len(uops), 3)
    out = uops[-1]
    self.assertEqual(out.op, UOps.ALU)
    self.assertEqual(out.arg, BinaryOps.ADD)
    self.assertEqual(out.src[1].op, UOps.CONST)
    self.assertEqual(out.src[1].arg, 6)

  def test_fold_gated_load(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    glbl1 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 1)
    glbl2 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 2)
    idx = UOp.const(dtypes.int, 0)
    ld0 = UOp(UOps.LOAD, dtypes.int, (glbl1, idx, UOp.const(dtypes.int, 2), UOp.const(dtypes.bool, False)))
    ld1 = UOp(UOps.LOAD, dtypes.int, (glbl2, idx, UOp.const(dtypes.int, 3), UOp.const(dtypes.bool, True)))
    uops = to_uops_list([UOp(UOps.STORE, dtypes.void, (glbl0, idx, ld1+ld0))])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    assert_equiv_uops(ld1, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    assert_equiv_uops(ld0, UOp.load(glbl2, idx, dtype=dtypes.int))

  def test_fold_gated_load_local(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    smem = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.int, local=True), (), ("temp", 1))
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 16))
    st = UOp(UOps.STORE, dtypes.void, (smem, lidx, UOp.load(glbl0, lidx, dtype=dtypes.int)))
    barrier = UOp(UOps.BARRIER, dtypes.void, (st, ))
    ld0 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+1, UOp.const(dtypes.int, 2), UOp.const(dtypes.bool, False), barrier))
    ld1 = UOp(UOps.LOAD, dtypes.int, (smem, lidx+2, UOp.const(dtypes.int, 3), UOp.const(dtypes.bool, True), barrier))
    uops = to_uops_list([UOp(UOps.STORE, dtypes.void, (glbl0, lidx, ld1+ld0))])
    ld0, ld1 = uops[-1].src[2].src
    # ld0 becomes the invalid value
    assert_equiv_uops(ld1, UOp.const(dtypes.int, 2))
    # the gate and invalid value are deleted from ld1
    assert_equiv_uops(ld0, UOp.load(smem, lidx+2, barrier, dtype=dtypes.int))

  def test_fold_gated_store(self):
    glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    idx0 = UOp.const(dtypes.int, 0)
    idx1 = UOp.const(dtypes.int, 0)
    val = UOp.const(dtypes.int, 42)
    st0 = UOp(UOps.STORE, dtypes.void, (glbl, idx0, val, UOp.const(dtypes.bool, False)))
    st1 = UOp(UOps.STORE, dtypes.void, (glbl, idx1, val, UOp.const(dtypes.bool, True)))
    uops = to_uops_list([st0, st1])
    # only the second store happens
    self.assertEqual(len(uops), 4)
    assert_equiv_uops(uops[-1], UOp.store(glbl, idx1, val))

  @unittest.skip("this is a uop type error")
  def test_asserts_bad_gate(self):
    glbl0 = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    idx = UOp.const(dtypes.int, 0)
    bad_gate = UOp.const(dtypes.int, 1)
    with self.assertRaises(AssertionError): to_uops_list([UOp(UOps.STORE, dtypes.void, (glbl0, idx, UOp.const(dtypes.int, 42), bad_gate))])

  def test_switched_range_order(self):
    glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    c0 = UOp.const(dtypes.int, 0)
    c2 = UOp.const(dtypes.int, 2)
    cf = UOp.const(dtypes.float, 0.0)
    r1 = UOp(UOps.RANGE, dtypes.int, (c0, c2), (1, 0, False))
    r2 = UOp(UOps.RANGE, dtypes.int, (c0, c2), (1, 1, False))
    alu = UOp(UOps.ALU, dtypes.int, (r2, r1), BinaryOps.MUL)
    store = UOp(UOps.STORE, dtypes.void, (glbl, alu, cf))
    uops = to_uops_list([store])
    ranges = [x for x in uops if x.op is UOps.RANGE]
    endranges = [x for x in uops if x.op is UOps.ENDRANGE]
    # ranges are closed in the right order
    self.assertEqual(endranges[-1].src[0], ranges[0])

def expander_rewrite(sink):
  sink = graph_rewrite(sink, sym + expander)
  return graph_rewrite(sink, sym + reducer)
def float4_rewrite(sink): return graph_rewrite(sink, sym + expander + float4_folding)

class TestExpander(unittest.TestCase):
  def test_expand_add_broadcast(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    sink = expander_rewrite(e1+3)
    assert sink.op is UOps.EXPAND and len(sink.src[0].src) == 4
    self.assertListEqual([x.arg for x in sink.src[0].src], [3,4,5,6])

  def test_contract_simple(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((1,4),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.VECTORIZE and len(sink.src) == 4
    self.assertListEqual([x.arg for x in sink.src], [0,1,2,3])

  def test_contract_axis_1(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,4),(2,4)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((1,4),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and len(sink.src[0].src) == 16 and sink.arg == ((2,4),)
    assert sink.src[0].op is UOps.VECTORIZE and len(sink.src[0].src) == 16
    self.assertListEqual([x.arg for x in sink.src[0].src][0:4], [0,4,8,12])
    self.assertListEqual([x.arg for x in sink.src[0].src][12:], [3,7,11,15])

  def test_contract_axis_2(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,4),(2,4)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((2,4),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and len(sink.src[0].src) == 16 and sink.arg == ((1,4),)
    assert sink.src[0].op is UOps.VECTORIZE and len(sink.src[0].src) == 16
    self.assertListEqual([x.arg for x in sink.src[0].src][0:4], [0,1,2,3])
    self.assertListEqual([x.arg for x in sink.src[0].src][12:], [12,13,14,15])

  def test_contract_axis_2_big(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,2),(2,2),(3,2),(4,2)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and sink.arg == ((1, 2), (3, 2), (4, 2))
    self.assertListEqual([x.arg for x in sink.src[0].src][0:2], [0,4])
    self.assertListEqual([x.arg for x in sink.src[0].src][12:14], [10,14])

  def test_contract_multi_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(16), tuple(x for x in range(16))),), ((1,2),(2,2),(3,2),(4,2)))
    sink = expander_rewrite(UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((3, 2), (2, 2))))
    assert sink.op is UOps.EXPAND and sink.arg == ((1, 2), (4, 2))
    self.assertListEqual([x.arg for x in sink.src[0].src][0:4], [0, 4, 2, 6])
    sink = expander_rewrite(UOp(UOps.CONTRACT, dtypes.int.vec(4), (e1,), ((2, 2), (3, 2))))
    assert sink.op is UOps.EXPAND and sink.arg == ((1, 2), (4, 2))
    self.assertListEqual([x.arg for x in sink.src[0].src][0:4], [0, 2, 4, 6])

  def test_contract_mid(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(8), tuple(x for x in range(8))),), ((1,2),(2,2),(3,2)))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.EXPAND and sink.arg == ((1,2),(3,2))
    assert sink.src[0].op is UOps.VECTORIZE and len(sink.src[0].src) == 8
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,2,1,3,4,6,5,7])

  def test_contract_no_expand(self):
    e1 = UOp(UOps.DEFINE_VAR, dtypes.int)
    con = UOp(UOps.CONTRACT, dtypes.int.vec(2), (e1,), ((2,2),))
    sink = expander_rewrite(con)
    assert sink.op is UOps.VECTORIZE and len(sink.src) == 2
    assert sink.src[0] == sink.src[1]

  def test_contract_half_expand(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    con = UOp(UOps.CONTRACT, dtypes.int.vec(8), (e1,), ((1,4), (2,2)))
    sink = expander_rewrite(con)
    assert sink.op is UOps.VECTORIZE and len(sink.src) == 8
    assert sink.src[0] == sink.src[1]
    assert sink.src[0] != sink.src[2]
    assert sink.src[6] == sink.src[7]

  def test_expand_same_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(4*x for x in range(4))),), ((1,4),))
    sink = expander_rewrite(e1+e2)
    assert sink.op is UOps.EXPAND and len(sink.src[0].src) == 4
    self.assertListEqual([x.arg for x in sink.src[0].src], [0,5,10,15])

  def test_expand_different_axis(self, flip=False):
    e1 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(4*x for x in range(4))),), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x for x in range(4))),), ((2,4),))
    sink = expander_rewrite((e2+e1) if flip else (e1+e2))
    assert sink.op is UOps.EXPAND and len(sink.src[0].src) == 16
    assert sink.arg == ((1, 4), (2, 4))
    self.assertListEqual([x.arg for x in sink.src[0].src], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

  def test_expand_different_axis_flip(self): self.test_expand_different_axis(True)

  @unittest.skip("no longer supported")
  def test_reduce_known_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    sink = UOp(UOps.REDUCE, dtypes.int, (3*e1,e1), BinaryOps.ADD)
    sink = expander_rewrite(sink)
    assert sink.op is UOps.CONST
    self.assertEqual(sink.arg, 3*(0+1+2+3))

  @unittest.skip("no longer supported")
  def test_reduce_const(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    sink = UOp(UOps.REDUCE, dtypes.int, (UOp.const(dtypes.int, 3), e1), BinaryOps.ADD)
    sink = expander_rewrite(sink)
    assert sink.op is UOps.CONST
    self.assertEqual(sink.arg, 3*4)

  @unittest.skip("no longer supported")
  def test_double_expand(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((2,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((2,4),))
    e = UOp(UOps.EXPAND, dtypes.int, (e1, e2), ((1,2),))
    sink = expander_rewrite(e)
    assert sink.op is UOps.EXPAND and len(sink.src) == 8
    assert sink.arg == ((1, 2), (2, 4))
    self.assertListEqual([x.arg for x in sink.src], [0,1,2,3,4,5,6,7])

  @unittest.skip("no longer supported")
  def test_double_expand_reverse(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((1,4),))
    e = UOp(UOps.EXPAND, dtypes.int, (e1, e2), ((2,2),))
    sink = expander_rewrite(e)
    assert sink.op is UOps.EXPAND and len(sink.src) == 8
    assert sink.arg == ((1, 4), (2, 2))
    self.assertListEqual([x.arg for x in sink.src], [0, 4, 1, 5, 2, 6, 3, 7])

  @unittest.skip("no longer supported")
  def test_double_expand_middle(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,2),(3,2)))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, 4+x) for x in range(4)), ((1,2),(3,2)))
    e = UOp(UOps.EXPAND, dtypes.int, (e1, e2), ((2,2),))
    sink = expander_rewrite(e)
    assert sink.op is UOps.EXPAND and len(sink.src) == 8
    assert sink.arg == ((1, 2), (2, 2), (3, 2))
    self.assertListEqual([x.arg for x in sink.src], [0, 1, 4, 5, 2, 3, 6, 7])

  # does this need to work?
  @unittest.expectedFailure
  @unittest.skip
  def test_reduce_different_axis(self):
    e1 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((1,4),))
    e2 = UOp(UOps.EXPAND, dtypes.int, tuple(UOp.const(dtypes.int, x) for x in range(4)), ((2,4),))
    sink = UOp(UOps.REDUCE, dtypes.int, (e1,e2), BinaryOps.ADD)
    sink = expander_rewrite(sink)
    print(sink)

class TestLoadStoreFolder(unittest.TestCase):
  def test_simple_load_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i))) for i in range(4)]
    sink = UOp(UOps.VECTORIZE, dtypes.float.vec(len(load)), tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 1

  def test_two_load_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i))) for i in range(8)]
    sink = UOp(UOps.VECTORIZE, dtypes.float.vec(len(load)), tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 2

  def test_simple_load_fold_gated(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp(UOps.DEFINE_VAR, dtypes.bool)
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate)) for i in range(4)]
    sink = UOp(UOps.VECTORIZE, dtypes.float.vec(len(load)), tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 1
    single_load = [x for x in sink.sparents if x.op is UOps.LOAD][0]
    self.assertListEqual(list(single_load.src[2].arg), [0.0, 1.0, 2.0, 3.0])

  def test_simple_load_dont_fold_different_gated(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp.variable("g1", False, True, dtypes.bool)
    gate2 = UOp.variable("g2", False, True, dtypes.bool)
    load = [UOp(UOps.LOAD, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate if i == 0 else gate2)) for i in range(4)]
    sink = UOp(UOps.VECTORIZE, dtypes.float.vec(len(load)), tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.LOAD]) == 3

  def test_simple_store_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    load = [UOp(UOps.STORE, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i))) for i in range(4)]
    sink = UOp(UOps.SINK, dtypes.void, tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.STORE]) == 1

  def test_simple_store_fold_gate(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp.variable("g1", False, True, dtypes.bool)
    load = [UOp(UOps.STORE, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate)) for i in range(4)]
    sink = UOp(UOps.SINK, dtypes.void, tuple(load))
    sink = float4_rewrite(sink)
    assert len([x for x in sink.sparents if x.op is UOps.STORE]) == 1
    one_store = [x for x in sink.sparents if x.op is UOps.STORE][0]
    assert len(one_store.src) == 4
    assert str(one_store.src[3]) == str(gate)  # huh, why do i need str here?

  def test_simple_store_dont_fold(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float))
    gate = UOp.variable("g1", False, True, dtypes.bool)
    gate2 = UOp.variable("g2", False, True, dtypes.bool)
    load = [UOp(UOps.STORE, dtypes.float, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate if i == 0 else gate2)) for i in range(4)]
    sink = UOp(UOps.SINK, dtypes.void, tuple(load))
    sink = float4_rewrite(sink)
    print(sink)
    assert len([x for x in sink.sparents if x.op is UOps.STORE]) == 3

def gate_rewrite(sink): return graph_rewrite(sink, sym + expander + reducer)

class TestIFUOps(unittest.TestCase):
  def test_create_ifs(self):
    gbuf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    sbuf = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.float, local=True), (), ("smem", 4))
    valid = UOp(UOps.SPECIAL, dtypes.int, (), ("gidx0", 10)).lt(5)
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 4))
    gate = valid&(lidx.ne(2))
    idx = UOp.const(dtypes.int, 0)
    st = UOp(UOps.STORE, dtypes.void, (sbuf, idx, UOp.const(dtypes.float, 42)))
    barrier = UOp(UOps.BARRIER, dtypes.void, (st,))
    lbuf = UOp(UOps.LOAD, dtypes.float, (sbuf, UOp.const(dtypes.int, 0), barrier))
    store = UOp(UOps.STORE, dtypes.void, (gbuf, UOp.const(dtypes.int, 0), lbuf, gate))
    sink = UOp(UOps.SINK, dtypes.void, (store,))
    sink = gate_rewrite(sink)
    if_uops = [u for u in sink.parents if u.op is UOps.IF]
    self.assertEqual(len(if_uops), 1)
    assert_equiv_uops(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 3)

  def test_expand_ifs_one_gate(self):
    gbuf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    sbuf = UOp(UOps.DEFINE_LOCAL, PtrDType(dtypes.float, local=True), (), ("smem", 16))
    valid = UOp(UOps.SPECIAL, dtypes.int, (), ("gidx0", 4)).lt(1)
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 16))
    gate = valid&(lidx.ne(2))
    st = UOp(UOps.STORE, dtypes.void, (sbuf, lidx, UOp.const(dtypes.float, 42)))
    barrier = UOp(UOps.BARRIER, dtypes.void, (st,))
    lbufs = [UOp(UOps.LOAD, dtypes.float, (sbuf, UOp.const(dtypes.int, i), barrier)) for i in range(4)]
    stores = [UOp(UOps.STORE, dtypes.void, (gbuf, UOp.const(dtypes.int, i), lbufs[i], gate)) for i in range(4)]
    sink = UOp(UOps.SINK, dtypes.void, tuple(stores))
    sink = gate_rewrite(sink)
    if_uops = [u for u in sink.parents if u.op is UOps.IF]
    self.assertEqual(len(if_uops), 1)
    assert_equiv_uops(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 3)

  # this will be fixed with the merge gated stores bounty
  @unittest.expectedFailure
  def test_expand_ifs_dumb(self):
    buf = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), 0)
    valid = UOp(UOps.SPECIAL, dtypes.int, (), ("gidx0", 10)).lt(5)
    lidx = UOp(UOps.SPECIAL, dtypes.int, (), ("lidx0", 4))
    gate = valid&(lidx.ne(2))
    stores = [UOp(UOps.STORE, dtypes.void, (buf, UOp.const(dtypes.int, i), UOp.const(dtypes.float, i), gate)) for i in range(4)]
    sink = UOp(UOps.SINK, dtypes.void, tuple(stores))
    sink = gate_rewrite(sink)
    if_uops = [u for u in sink.parents if u.op is UOps.IF]
    self.assertEqual(len(if_uops), 1)
    assert_equiv_uops(if_uops[0].src[0], gate)
    for st in sink.src:
      self.assertEqual(len(st.src), 3)


if __name__ == '__main__':
  unittest.main(verbosity=2)
