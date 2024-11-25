from typing import Dict, List, Optional
import unittest
from tinygrad.codegen.kernel import Kernel
from tinygrad.dtype import dtypes
from tinygrad.ops import TRACK_MATCH_STATS, TrackedPatternMatcher as PatternMatcher, UOp, Ops, UPat, graph_rewrite, contexts, track_rewrites
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.viz.serve import get_details, get_metadata, uop_to_json

@track_rewrites(named=True)
def rewrite(sink:UOp, pm:PatternMatcher, **kwargs): return graph_rewrite(sink, pm, **kwargs)

def helper_test_viz(sink:UOp, pm:PatternMatcher, **kwargs) -> List[UOp]:
  rewrite(sink, pm, **kwargs)
  assert len(contexts) == 1
  assert len(contexts[0][1]) == 1
  kernels = get_metadata(contexts)
  k = kernels[0][0]
  g = get_details(*k)
  for v in kernels:
    for args in v: get_details(*args)
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

  def test_nested_graph_rewrite(self):
    a = UOp(Ops.DEFINE_VAR, dtypes.int, arg=('a', 0, 9))
    b = UOp(Ops.DEFINE_VAR, dtypes.int, arg=('b', 0, 9))
    start = (a*2)+(b*2)
    def rhs(root:UOp):
      fake = UOp.const(dtypes.int, 0)
      dvars = {UOp(Ops.DEFINE_VAR, dtypes.int, arg=('a', 0, 9)):fake}
      pm_replace = PatternMatcher([
        (UPat(tuple(Ops), name="x"), lambda ctx,x: ctx.get(x,None)),
        (UPat.var("x")*0, lambda ctx,x: UOp.const(x.dtype, 0)),
        (UPat.var("x")+0, lambda ctx,x: x),
      ])
      new_src = tuple(graph_rewrite(y, pm_replace, dvars, bottom_up=True) for y in root.src)
      return root.replace(src=new_src) if new_src != root.src else None
    pm = PatternMatcher([(UPat(tuple(Ops), name="root"), rhs)])
    @track_rewrites(named=True)
    def nested_rewrite(start:UOp): return graph_rewrite(start, pm)
    ret = nested_rewrite(start)
    self.assertIs(ret, b*2)

  # PYTHONPATH=. FUZZ_VIZ=1 VIZ=1 python3 test/test_viz.py TestViz.test_substitute_ops
  def test_substitute_ops(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 1, 1, 4, 1, 4), strides=(0, 0, 0, 0, 0, 4, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (4, 6)), src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
            UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 2, 5, 2, 5), strides=(0, 0, 2, 12, 1, 4), offset=0, mask=((0, 1), (0, 1), (0, 2), (0, 3), (0, 2), (0, 3)), contiguous=False), View(shape=(1, 1, 12, 12), strides=(0, 0, 10, 1), offset=0, mask=((0, 1), (0, 1), (0, 10), (0, 10)), contiguous=False), View(shape=(1, 1, 1, 1, 3, 4, 3, 4), strides=(0, 0, 0, 0, 48, 12, 4, 1), offset=0, mask=None, contiguous=True))), src=()),)),)),)),))
    Kernel(ast).to_program()

if __name__ == "__main__":
  unittest.main()
