from typing import Dict, List, Optional
import unittest
from tinygrad.dtype import dtypes
from tinygrad.ops import TRACK_MATCH_STATS, TrackedPatternMatcher as PatternMatcher, UOp, Ops, UPat, graph_rewrite, contexts, track_rewrites
from tinygrad.viz.serve import get_details, get_metadata, uop_to_json

@track_rewrites(named=True)
def rewrite(sink:UOp, pm:PatternMatcher, **kwargs): return graph_rewrite(sink, pm, **kwargs)

def helper_test_viz(sink:UOp, pm:PatternMatcher, **kwargs) -> List[UOp]:
  rewrite(sink, pm, **kwargs)
  assert len(contexts) == 1
  assert len(contexts[0][1]) == 1
  k = get_metadata(contexts)[0][0]
  g = get_details(*k)
  return g.graphs[1:]

class TestViz(unittest.TestCase):
  def setUp(self):
    contexts.clear()
    self.tms = TRACK_MATCH_STATS.value
    TRACK_MATCH_STATS.value = 2
  def tearDown(self): TRACK_MATCH_STATS.value = self.tms

  def test_viz_simple(self):
    pm = PatternMatcher([
      (UPat.var("x")*1, lambda x:x),
    ])
    a = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0), UOp.const(dtypes.int, 0)))
    uops = helper_test_viz(a*1, pm)
    self.assertEqual(len(uops), 1)
    self.assertEqual(uops[0], a)

  def test_rewrite_twice(self):
    pm = PatternMatcher([
      (UPat.var("x")+UPat.var("x"), lambda x:x*2),
      (UPat.var("x", dtypes.int)*2, lambda x:x.alu(Ops.SHL, UOp.const(dtypes.int, 1))),
    ])
    a = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0), UOp.const(dtypes.int, 0)))
    uops = helper_test_viz(a+a, pm)
    self.assertEqual(len(uops), 2)
    self.assertEqual(uops[0], a*2)
    self.assertEqual(uops[1], graph_rewrite(a+a, pm))

  def test_rewrite_with_ctx(self):
    a = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0), UOp.const(dtypes.int, 0)))
    b = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 1), UOp.const(dtypes.int, 0)))
    def store_load(ctx:Dict[UOp, None], x:UOp) -> Optional[UOp]:
      if x in ctx: return None
      ctx[x] = None
      return UOp.store(*x.src, x)
    pm = PatternMatcher([
      (UPat(Ops.LOAD, name="x"), store_load),
    ])
    uops = helper_test_viz(a+b, pm, ctx={})
    self.assertEqual(len(uops), 2)
    self.assertEqual(uops[-1], graph_rewrite(a+b, pm, {}))

  def test_track_rewrites(self):
    simple = PatternMatcher([(UPat.var("x")*1, lambda x:x)])
    @track_rewrites(named=True)
    def do_rewrite(x:UOp): return graph_rewrite(x, simple)
    ld = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1), UOp.const(dtypes.int, 0)))
    do_rewrite(ld*1)
    do_rewrite(ld*2)
    ret = get_metadata(contexts)
    self.assertEqual(len(ret), 2)
    key, _, m = ret[0][0]
    self.assertEqual(key, "do_rewrite_1")
    self.assertEqual(len(m.upats), 1)
    key, _, m = ret[1][0]
    self.assertEqual(key, "do_rewrite_2")
    self.assertEqual(len(m.upats), 0)

  def test_track_rewrites_with_exception(self):
    simple = PatternMatcher([(UPat.var("x")*1, lambda x:x)])
    @track_rewrites()
    def do_rewrite(x:UOp):
      x = graph_rewrite(x, simple) # NOTE: viz tracks this
      raise Exception("test")
    ld = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=1), UOp.const(dtypes.int, 0)))
    with self.assertRaises(Exception): do_rewrite(ld*1)
    ret = get_metadata(contexts)
    self.assertEqual(len(ret), 1)

  def test_fold_const(self):
    a = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0), UOp.const(dtypes.int, 0)))
    graph = uop_to_json(a)
    assert not any(v[0].startswith("CONST") for v in graph.values())
    assert len([x for x in graph.values() if "CONST" in x[0]]) == 1

  def test_bottom_up_rewrite(self):
    a = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0), UOp.const(dtypes.int, 0)))
    n1 = a.sin()
    uop = n1.sin()
    pm = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None))])
    ret = helper_test_viz(uop, pm, ctx={a.sin():a.sqrt(), n1.sin():n1.sqrt()}, bottom_up=True)
    self.assertEqual(len(ret), 2)
    self.assertIs(ret[0], a.sin().sqrt()) # first rewrite
    self.assertIs(ret[1], a.sqrt().sqrt()) # second one

  def test_top_down_rewrite(self):
    a = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), (), 0), UOp.const(dtypes.int, 0)))
    n1 = a.sin()
    uop = n1.sin()
    pm = PatternMatcher([(UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None))])
    # if it wasn't bottom_up, it's rewritten once
    ret = helper_test_viz(uop, pm, ctx={a.sin():a.sqrt(), n1.sin():n1.sqrt()}, bottom_up=False)
    self.assertEqual(len(ret), 1)
    self.assertIs(ret[0], a.sqrt().sin()) # only rewrite

if __name__ == "__main__":
  unittest.main()
