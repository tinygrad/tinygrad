from typing import Dict, List
import unittest
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.ops import TRACK_MATCH_STATS, TrackedPatternMatcher as PatternMatcher, UOp, UOps, UPat, graph_rewrite, contexts, track_rewrites
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
    ld = UOp(UOps.LOAD, dtypes.int, (UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0), UOp.const(dtypes.int, 0)))
    uops = viz(sink:=ld*1, pm)
    assert len(uops) == 1
    assert uops[0] == graph_rewrite(sink, pm)

if __name__ == "__main__":
  unittest.main()
