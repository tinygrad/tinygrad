from typing import Any, List, Tuple
import unittest
import os, itertools
os.environ["TRACK_MATCH_STATS"] = "2"
os.environ["PRINT_MATCH_STATS"] = "0"
from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import lower_schedule
from tinygrad.dtype import PtrDType
from tinygrad.helpers import Context, all_same, getenv
from tinygrad.ops import TrackedRewriteContext, UOp, UOps, graph_rewrite, PatternMatcher, UPat, contexts, KernelInfo, BinaryOps, track_rewrites
from tinygrad.codegen.uopgraph import sym, devectorize, float4_folding
from viz.serve import GraphRewriteMetadata, get_metadata, get_details, _uop_to_json

def group_rewrites(kernels:List[GraphRewriteMetadata]): return {k:list(v) for k,v in itertools.groupby(kernels, lambda x:x.loc)}

class TestViz(unittest.TestCase):
  def tearDown(self) -> None:
    from tinygrad.ops import contexts
    if not getenv("VIZ"): contexts.clear()

  def assert_valid_ctx(self, contexts:List[Tuple[Any,List[TrackedRewriteContext]]]):
    assert len(contexts) != 0
    get_metadata(contexts)

  def assert_valid_graph(self, t):
    contexts.clear()
    s = t.schedule()
    list(lower_schedule(s))
    self.assert_valid_ctx(contexts)

  def test_ctx_diff(self):
    a = Tensor.ones(4, 1).contiguous().realize()
    out = a + a.reshape(1, 4)
    self.assert_valid_graph(out)

  def test_ctx_groups(self):
    contexts.clear()
    schedule1 = Tensor.zeros(4, 1).contiguous().exp().schedule()
    schedule2 = Tensor.zeros(4, 1).contiguous().exp().schedule()
    list(lower_schedule(schedule1))
    list(lower_schedule(schedule2))
    with Context(TRACK_MATCH_STATS=0): ret = get_metadata(contexts)
    assert len(ret) == 3
    assert all(len([x for _,_,x in y if "schedule" in x.loc[0]]) == 0 for y in ret[1:])
    assert all(len([x for _,_,x in y if "uopgraph" in x.loc[0]]) != 0 for y in ret[1:])

  def test_gemm_diff(self):
    x = Tensor.empty(64, 64).realize()
    y = Tensor.empty(64, 64).realize()
    out = x.matmul(y)
    self.assert_valid_graph(out)

  def test_removed_node(self):
    vec = UOp(UOps.VECTORIZE, dtypes.int.vec(4), tuple((UOp.const(dtypes.int, 1),)*4))
    gep = UOp(UOps.GEP, dtypes.int, (vec,), (0,))
    sink = UOp(UOps.STORE, dtypes.void, (UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0), UOp.const(dtypes.int, 0), gep)).sink()
    pm = PatternMatcher([
      (UPat(UOps.VECTORIZE, name="root", src=(UPat(UOps.CONST, name="const"),), allow_any_len=True, location="test"),
       lambda root,const: UOp.const_like(root, const.arg) if all_same(root.src) else None),
      (UPat(UOps.GEP, name="root", src=(UPat(UOps.CONST, name="x"),), location="test"), lambda root,x: root.const_like(x.arg))
    ])
    @track_rewrites
    def f(k): return graph_rewrite(sink, pm)
    ret = f("test_rewrite")
    self.assert_valid_ctx(contexts)
    args = get_metadata(contexts)[0][0]
    g = get_details(*args)
    assert g.graphs[-1] == _uop_to_json(ret)

  def test_devectorize_viz(self):
    sink = UOp(UOps.SINK, dtypes.void, arg=KernelInfo(local_dims=1, upcasted=1, dont_use_locals=False), src=(
      UOp(UOps.STORE, dtypes.void, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0, src=()),
        UOp(UOps.ALU, dtypes.int.vec(4), arg=BinaryOps.ADD, src=(
          UOp(UOps.VECTORIZE, dtypes.int.vec(4), arg=None, src=(
            x4:=UOp(UOps.ALU, dtypes.int, arg=BinaryOps.MUL, src=(
              x5:=UOp(UOps.SPECIAL, dtypes.int, arg=('lidx0', 4), src=()),
              UOp(UOps.CONST, dtypes.int, arg=4, src=()),)),
             x4,
             x4,
             x4,)),
          x7:=UOp(UOps.VCONST, dtypes.int.vec(4), arg=(0, 1, 2, 3), src=()),)),
        UOp(UOps.ALU, dtypes.float.vec(4), arg=BinaryOps.ADD, src=(
          UOp(UOps.VECTORIZE, dtypes.float.vec(4), arg=None, src=(
            x10:=UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              x11:=UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=1, src=()),
               x5,)),
             x10,
             x10,
             x10,)),
          UOp(UOps.LOAD, dtypes.float.vec(4), arg=None, src=(
             x11,
             x7,)),)),)),))
    pm = sym+(devectorize+float4_folding)
    @track_rewrites
    def f(k): return graph_rewrite(sink, pm)
    f("test_rewrite")
    self.assert_valid_ctx(contexts)
    assert all(ctx.loc[0].split("/")[-1] == __file__.split("/")[-1] for _,ctxs in contexts for ctx in ctxs)

  def test_no_ctx(self):
    simple_pm = PatternMatcher([(UPat(UOps.CONST), lambda:True)])
    simple_pm.rewrite(UOp.const(dtypes.int, 2))
    self.assertEqual(len(contexts), 0)

  def test_dedup_ast(self):
    contexts.clear()
    a = Tensor.empty(4, 4).contiguous().realize()+2
    b = Tensor.empty(4, 4).contiguous().realize()+2
    Tensor.schedule(a, b)
    with Context(TRACK_MATCH_STATS=0): kernels = get_metadata(contexts)
    self.assertEqual(len(kernels), 1)
    rewrites = [x[2] for x in kernels[0]]
    assert all(len(v) == 1 for k,v in group_rewrites(rewrites).items() if "schedule.py" in k)

  def test_no_dedup_different_opts(self):
    contexts.clear()
    a = Tensor.empty(4, 4)+Tensor.empty(4, 4)
    s = a.schedule()
    with Context(NOOPT=1): list(lower_schedule(s.copy()))
    with Context(NOOPT=0): list(lower_schedule(s.copy()))
    with Context(TRACK_MATCH_STATS=0): kernels = get_metadata(contexts)[1:]
    self.assertEqual(len(kernels), 2)
    rewrites = [x[2] for x in kernels[0]]
    assert all(len(v) == 1 for _,v in group_rewrites(rewrites).items())

  def test_fold_const_nodes(self):
    a = Tensor.empty(4, 4)+2
    contexts.clear()
    sink = a.schedule()[-1].ast
    ret = _uop_to_json(sink)
    assert not any(v[0].startswith("CONST") for v in ret.values())
    assert len([x for x in ret.values() if "CONST" in x[0]]) == 1

  @unittest.skip("VIZ for a single CONST isn't supported anymore")
  def test_no_fold_single_const(self):
    node = UOp(UOps.CONST, dtypes.float, (), 1.0)
    ret = _uop_to_json(node, base=node)
    assert len(ret) == 1

if __name__ == "__main__":
  unittest.main()
