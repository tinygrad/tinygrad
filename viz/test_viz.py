import unittest
import os
os.environ["TRACK_MATCH_STATS"] = "2"
from extra.models.resnet import ResNet50
from tinygrad import Tensor
from tinygrad.engine.realize import lower_schedule
from tinygrad.ops import UOp, UOps, graph_rewrite, PatternMatcher, UPat, contexts, KernelInfo, BinaryOps
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.helpers import all_same, DEBUG, colored, getenv
from tinygrad.codegen.uopgraph import constant_folder, devectorize, float4_folding
from test.external.process_replay.helpers import print_diff
from viz.serve import create_graph

class TestViz(unittest.TestCase):
  def tearDown(self) -> None:
    from tinygrad.ops import contexts
    if not getenv("VIZ"): contexts.clear()

  def assert_valid_ctx(self, contexts):
    assert len(contexts) != 0
    for i,ctx in enumerate(contexts):
      try: ret = create_graph(ctx)
      except Exception as e:
        print(colored(f"failed to create graph for ctx {i}", "red"))
        raise e
      for j,(x,y) in enumerate(zip(ret.uops, ret.uops[1:])):
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
      (UPat(UOps.VECTORIZE, name="root", src=(UPat(UOps.CONST, name="const"),), allow_any_len=True),
       lambda root,const: UOp.const_like(root, const.arg) if all_same(root.src) else None),
      (UPat(UOps.GEP, name="root", src=(UPat(UOps.CONST, name="x"),)), lambda root,x: root.const_like(x.arg))
    ])
    ret = graph_rewrite(sink, pm)
    if DEBUG >= 4: print_diff(sink, ret)
    g = create_graph(contexts[0])
    assert g.uops[-1].key == ret.key
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

  def test_fuzz_resnet(self):
    mdl = ResNet50()
    img = Tensor.empty(64, 3, 224, 224)
    out = mdl(img)
    sched = out.schedule()
    list(lower_schedule(sched))
    self.assert_valid_ctx(contexts)

if __name__ == "__main__":
  unittest.main()
