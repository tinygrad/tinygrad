import unittest
import os
os.environ["TRACK_MATCH_STATS"] = "2"
from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.engine.realize import lower_schedule
from tinygrad.ops import UOp, UOps, graph_rewrite, PatternMatcher, UPat, contexts, KernelInfo, BinaryOps
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import CI, Context, all_same, DEBUG, colored, getenv
from tinygrad.codegen.uopgraph import constant_folder, devectorize, float4_folding
from test.external.process_replay.helpers import print_diff
from viz.serve import UOpRet, load_kernels

class TestViz(unittest.TestCase):
  def tearDown(self) -> None:
    from tinygrad.ops import contexts
    if not getenv("VIZ"): contexts.clear()

  def assert_valid_ctx(self, contexts):
    assert len(contexts) != 0
    for i,ctx in enumerate(contexts):
      try: ret = UOpRet.from_ctx(ctx)
      except Exception as e:
        print(colored(f"failed to create graph for ctx {i}", "red"))
        raise e
      rewrites = [x[0] for x in ret.graphs]
      for j,(x,y) in enumerate(zip(rewrites, rewrites[1:])):
        if x.key == y.key:
          raise AssertionError(f"failed to generate the correct diff at rewrite {j} ctx {i}")

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
    schedule1 = Tensor.randn(4, 1).contiguous().schedule()
    schedule2 = Tensor.randn(4, 4).contiguous().schedule()
    list(lower_schedule(schedule1))
    list(lower_schedule(schedule2))
    ret = load_kernels(contexts)
    assert len(ret) == 2
    assert all(len([x for x in y.ctxs.values() if "schedule" in x.loc]) != 0 for y in ret)
    assert all(len([x for x in y.ctxs.values() if "uopgraph" in x.loc]) != 0 for y in ret)

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
    ret = graph_rewrite(sink, pm)
    if DEBUG >= 4: print_diff(sink, ret)
    g = UOpRet.from_ctx(contexts[0])
    assert g.graphs[-1][0].key == ret.key
    self.assert_valid_ctx(contexts)

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
    pm = constant_folder+(devectorize+float4_folding)
    new_sink = graph_rewrite(sink, pm)
    if DEBUG >= 4: print_diff(sink, new_sink, unified=0)
    self.assert_valid_ctx(contexts)
    assert all(ctx.loc.split("/")[-1].split(":")[0] == __file__.split("/")[-1] for ctx in contexts)

  @unittest.skipIf(CI, "slow, it's generating diffs for 36202 rules")
  def test_fuzz_resnet(self):
    mdl = ResNet50()
    img = Tensor.empty(64, 3, 224, 224)
    out = mdl(img)
    sched = out.schedule()
    list(lower_schedule(sched))
    self.assert_valid_ctx(contexts)

  def test_no_ctx(self):
    simple_pm = PatternMatcher([(UPat(UOps.CONST), lambda:True)])
    simple_pm.rewrite(UOp.const(dtypes.int, 2))
    self.assertEqual(len(contexts), 0)

  def test_dedup_ast(self):
    contexts.clear()
    a = Tensor.randn(4, 4)+2
    b = Tensor.randn(4, 4)+2
    Tensor.schedule(a, b)
    kernels = load_kernels(contexts)
    self.assertEqual(len(kernels), 1)
    schedule_ctxs = [x for x in kernels[0].ctxs.values() if x.loc.split("/")[-1].split(":")[0] == "schedule.py"]
    self.assertEqual(len(schedule_ctxs), 1)

  def test_no_dedup_different_opts(self):
    contexts.clear()
    a = Tensor.empty(4, 4)+Tensor.empty(4, 4)
    s = a.schedule()
    with Context(NOOPT=1): list(lower_schedule(s.copy()))
    with Context(NOOPT=0): list(lower_schedule(s.copy()))
    kernels = load_kernels(contexts)
    self.assertEqual(len(kernels), 2)
    schedule_ctxs = [x for x in kernels[0].ctxs.values() if x.loc.split("/")[-1].split(":")[0] == "schedule.py"]
    self.assertEqual(len(schedule_ctxs), 1)
    schedule_ctxs = [x for x in kernels[1].ctxs.values() if x.loc.split("/")[-1].split(":")[0] == "schedule.py"]
    self.assertEqual(len(schedule_ctxs), 0)

if __name__ == "__main__":
  unittest.main()
