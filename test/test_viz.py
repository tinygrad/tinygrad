from typing import List
import unittest
import itertools
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context, getenv
from tinygrad.engine.realize import lower_schedule
from tinygrad.viz.serve import GraphRewriteMetadata, get_metadata, _uop_to_json
from tinygrad.ops import TRACK_MATCH_STATS, TrackedPatternMatcher, UPat, UOps, UOp, graph_rewrite, contexts, track_rewrites

def group_rewrites(kernels:List[GraphRewriteMetadata]): return {k:list(v) for k,v in itertools.groupby(kernels, lambda x:x.loc)}

class TestViz(unittest.TestCase):
  def setUp(self) -> None:
    contexts.clear()
    self.prev_val = TRACK_MATCH_STATS.value
    TRACK_MATCH_STATS.value = 2
  def tearDown(self) -> None:
    from tinygrad.ops import TRACK_MATCH_STATS, contexts
    if not getenv("VIZ"): contexts.clear()
    TRACK_MATCH_STATS.value = self.prev_val

  def assert_valid_ctx(self):
    from tinygrad.ops import contexts
    assert len(contexts) != 0
    return get_metadata(contexts)

  def assert_valid_graph(self, t):
    s = t.schedule()
    list(lower_schedule(s))
    self.assert_valid_ctx()

  def test_ctx_diff(self):
    a = Tensor.ones(4, 1).contiguous().realize()
    out = a + a.reshape(1, 4)
    self.assert_valid_graph(out)

  def test_ctx_groups(self):
    schedule1 = Tensor.zeros(4, 1).contiguous().exp().schedule()
    schedule2 = Tensor.zeros(4, 1).contiguous().exp().schedule()
    list(lower_schedule(schedule1))
    list(lower_schedule(schedule2))
    ret = self.assert_valid_ctx()
    assert len(ret) == 3
    assert all(len([x for _,_,x in y if "schedule" in x.loc[0]]) == 0 for y in ret[1:])
    assert all(len([x for _,_,x in y if "uopgraph" in x.loc[0]]) != 0 for y in ret[1:])

  def test_gemm_diff(self):
    x = Tensor.empty(64, 64).realize()
    y = Tensor.empty(64, 64).realize()
    out = x.matmul(y)
    self.assert_valid_graph(out)

  def test_track_no_ctx(self):
    @track_rewrites
    def simplify_and_verify(u:UOp):
      simplify = TrackedPatternMatcher([(UPat.var("x")*1, lambda x:x)])
      verify = TrackedPatternMatcher([(UPat(UOps.CONST), lambda:True)])
      verify.rewrite(graph_rewrite(u, simplify))
    u = UOp(UOps.LOAD, dtypes.int, (UOp(UOps.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1), UOp.const(dtypes.int, 0)))*1
    simplify_and_verify(u)
    ret = self.assert_valid_ctx()
    self.assertEqual(len(ret), 1)
    key, ctx, metadata = ret[0][0]
    self.assertIs(key, u)
    self.assertIs(ctx.sink, u)
    self.assertEqual(len(metadata.upats), 1)

  def test_track_rewrites(self):
    simple = TrackedPatternMatcher([(UPat.var("x")*1, lambda x:x)])
    @track_rewrites
    def do_rewrite(key:str, x:UOp): return graph_rewrite(x, simple)
    ld = UOp(UOps.LOAD, dtypes.int, (UOp(UOps.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1), UOp.const(dtypes.int, 0)))
    do_rewrite("uop_0", ld*1)
    do_rewrite("uop_1", ld*2)
    ret = self.assert_valid_ctx()
    self.assertEqual(len(ret), 1)
    key, _, m = ret[0][0]
    self.assertEqual(key, "uop_0")
    self.assertEqual(len(m.upats), 1)
    key, _, m = ret[0][1]
    self.assertEqual(key, "uop_1")
    self.assertEqual(len(m.upats), 0)

  def test_track_with_exception(self):
    simple = TrackedPatternMatcher([(UPat.var("x")*1, lambda x:x)])
    @track_rewrites
    def do_rewrite(key:str, x:UOp):
      x = graph_rewrite(x, simple) # NOTE: viz tracks this
      raise Exception("test")
    ld = UOp(UOps.LOAD, dtypes.int, (UOp(UOps.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1), UOp.const(dtypes.int, 0)))
    with self.assertRaises(Exception): do_rewrite("uop_0", ld*1)
    ret = self.assert_valid_ctx()
    self.assertEqual(len(ret), 1)

  def test_dedup_ast(self):
    a = Tensor.empty(4, 4).contiguous().realize()+2
    b = Tensor.empty(4, 4).contiguous().realize()+2
    Tensor.schedule(a, b)
    kernels = self.assert_valid_ctx()
    self.assertEqual(len(kernels), 1)
    rewrites = [x[2] for x in kernels[0]]
    assert all(len(v) == 1 for k,v in group_rewrites(rewrites).items() if "schedule.py" in k)

  @unittest.skip("broken")
  def test_no_dedup_different_opts(self):
    a = Tensor.empty(4, 4)+Tensor.empty(4, 4)
    s = a.schedule()
    with Context(NOOPT=1): list(lower_schedule(s.copy()))
    with Context(NOOPT=0): list(lower_schedule(s.copy()))
    kernels = self.assert_valid_ctx()[1:]
    self.assertEqual(len(kernels), 2)
    rewrites = [x[2] for x in kernels[0]]
    assert all(len(v) == 1 for _,v in group_rewrites(rewrites).items())

  def test_fold_const_nodes(self):
    a = Tensor.empty(4, 4)+2
    sink = a.schedule()[-1].ast
    ret = _uop_to_json(sink)
    assert not any(v[0].startswith("CONST") for v in ret.values())
    assert len([x for x in ret.values() if "CONST" in x[0]]) == 1

if __name__ == "__main__":
  unittest.main()
