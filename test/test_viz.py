from typing import Dict, List
import unittest
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.ops import TRACK_MATCH_STATS, BinaryOps, TrackedPatternMatcher as PatternMatcher, UOp, UOps, UPat, graph_rewrite, contexts, track_rewrites
from tinygrad.viz.serve import _replace_uop

@track_rewrites
def _rewrite(sink:UOp, pm:PatternMatcher): return graph_rewrite(sink, pm)

def viz(sink:UOp, pm:PatternMatcher) -> List[UOp]:
  _rewrite(sink, pm)
  assert len(contexts) == 1
  assert len(contexts[0][1]) == 1
  ctx = contexts[0][1][0]
  uops = [ctx.sink]
  replaces: Dict[UOp, UOp] = {}
  for u0,u1,_ in ctx.rewrites:
    replaces[u0] = u1
    new_sink = _replace_uop(uops[-1], {**replaces})
    uops.append(new_sink)
  return uops[1:]

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
    a = UOp(UOps.LOAD, dtypes.int, (UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0), UOp.const(dtypes.int, 0)))
    uops = viz(a*1, pm)
    self.assertEqual(len(uops), 1)
    self.assertEqual(uops[0], a)

  def test_rewrite_twice(self):
    pm = PatternMatcher([
      (UPat.var("x")+UPat.var("x"), lambda x:x*2),
      (UPat.var("x", dtypes.int)*2, lambda x:x.alu(BinaryOps.SHL, UOp.const(dtypes.int, 1))),
    ])
    a = UOp(UOps.LOAD, dtypes.int, (UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0), UOp.const(dtypes.int, 0)))
    uops = viz(a+a, pm)
    self.assertEqual(len(uops), 2)
    self.assertEqual(uops[0], a*2)
    self.assertEqual(uops[1], graph_rewrite(a+a, pm))

if __name__ == "__main__":
  unittest.main()
