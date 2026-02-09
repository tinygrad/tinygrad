import unittest
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp, GroupOp, PatternMatcher, UPat, graph_rewrite
from tinygrad.uop.egraph import uf_find, uf_union, rewrite_all, EGraph, egraph_saturate, egraph_extract, node_cost

# *** test union-find ***

class TestUnionFind(unittest.TestCase):
  def test_find_self(self):
    a, b = UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2)
    parent = {a: a, b: b}
    self.assertIs(uf_find(parent, a), a)
    self.assertIs(uf_find(parent, b), b)

  def test_union_basic(self):
    a, b = UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2)
    parent = {a: a, b: b}
    size = {a: 1, b: 1}
    root = uf_union(parent, size, a, b)
    self.assertIs(uf_find(parent, a), uf_find(parent, b))
    self.assertIs(root, uf_find(parent, a))

  def test_union_chain(self):
    a, b, c = UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2), UOp.const(dtypes.int, 3)
    parent = {a: a, b: b, c: c}
    size = {a: 1, b: 1, c: 1}
    uf_union(parent, size, a, b)
    uf_union(parent, size, b, c)
    self.assertIs(uf_find(parent, a), uf_find(parent, c))

  def test_union_idempotent(self):
    a, b = UOp.const(dtypes.int, 1), UOp.const(dtypes.int, 2)
    parent = {a: a, b: b}
    size = {a: 1, b: 1}
    r1 = uf_union(parent, size, a, b)
    r2 = uf_union(parent, size, a, b)
    self.assertIs(r1, r2)

# *** test rewrite_all ***

class TestRewriteAll(unittest.TestCase):
  def test_single_match(self):
    pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])
    a = UOp.variable("a", 0, 10)
    results = rewrite_all(pm, a + 0)
    self.assertEqual(len(results), 1)
    self.assertIs(results[0], a)

  def test_no_match(self):
    pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    results = rewrite_all(pm, a + b)
    self.assertEqual(len(results), 0)

  def test_multiple_matches(self):
    pm = PatternMatcher([
      (UPat.var("x") + 0, lambda x: x),
      (UPat.var("x") * 1, lambda x: x),
    ])
    a = UOp.variable("a", 0, 10)
    results = rewrite_all(pm, a + 0)
    self.assertEqual(len(results), 1)
    self.assertIs(results[0], a)

  def test_both_rules_fire(self):
    pm = PatternMatcher([
      (UPat.var("x") + UPat.var("x"), lambda x: x * 2),
      (UPat.var("x") + UPat.var("x"), lambda x: UOp(Ops.SHL, x.dtype, (x, x.const_like(1)))),
    ])
    a = UOp.variable("a", 0, 10)
    results = rewrite_all(pm, a + a)
    self.assertEqual(len(results), 2)

  def test_const_folding(self):
    pm = PatternMatcher([
      (UPat(GroupOp.Binary, src=(UPat((Ops.CONST, Ops.VCONST)),)*2, name="a"),
       lambda a: a.const_like(a.src[0].arg + a.src[1].arg) if a.op is Ops.ADD else None),
    ])
    results = rewrite_all(pm, UOp.const(dtypes.int, 3) + UOp.const(dtypes.int, 4))
    self.assertEqual(len(results), 1)
    self.assertEqual(results[0].arg, 7)

# *** test EGraph class ***

class TestEGraphClass(unittest.TestCase):
  def test_init(self):
    a = UOp.variable("a", 0, 10)
    expr = a + 0
    eg = EGraph(expr)
    self.assertEqual(len(eg.eclass), len(list(expr.toposort())))
    self.assertIn(expr, eg.all_nodes)

  def test_add_node(self):
    a = UOp.variable("a", 0, 10)
    eg = EGraph(a)
    b = UOp.variable("b", 0, 10)
    eg._add_node(b)
    self.assertIn(b, eg.all_nodes)

  def test_merge(self):
    a = UOp.variable("a", 0, 10)
    expr = a + 0
    eg = EGraph(expr)
    result = eg._merge(expr, a)
    self.assertIsNotNone(result)
    self.assertIs(uf_find(eg.parent, expr), uf_find(eg.parent, a))

  def test_merge_idempotent(self):
    a = UOp.variable("a", 0, 10)
    eg = EGraph(a)
    result = eg._merge(a, a)
    self.assertIsNone(result)

# *** test egraph_saturate ***

class TestEGraphSaturate(unittest.TestCase):
  def test_identity_rules(self):
    pm = PatternMatcher([
      (UPat.var("x") + 0, lambda x: x),
      (UPat.var("x") * 1, lambda x: x),
    ])
    a = UOp.variable("a", 0, 10)
    expr = a + 0
    eclass = egraph_saturate(expr, pm)
    # a+0 and a should be in the same e-class
    a_class = expr_class = None
    for canon, members in eclass.items():
      if a in members: a_class = canon
      if expr in members: expr_class = canon
    self.assertIsNotNone(a_class)
    self.assertIsNotNone(expr_class)
    self.assertIs(a_class, expr_class)

  def test_const_fold_saturation(self):
    from tinygrad.uop.symbolic import symbolic_simple
    c2, c3 = UOp.const(dtypes.int, 2), UOp.const(dtypes.int, 3)
    expr = c2 + c3
    eclass = egraph_saturate(expr, symbolic_simple)
    c5 = UOp.const(dtypes.int, 5)
    for canon, members in eclass.items():
      if expr in members:
        self.assertIn(c5, members, f"expected CONST(5) in eclass of 2+3, got {members}")
        return
    self.fail("expr not found in any eclass")

  def test_no_rules_match(self):
    pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    eclass = egraph_saturate(a + b, pm)
    for canon, members in eclass.items():
      self.assertEqual(len(members), 1)

  def test_max_iters_respected(self):
    pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])
    a = UOp.variable("a", 0, 10)
    expr = a + 0
    eclass = egraph_saturate(expr, pm, max_iters=1)
    a_class = expr_class = None
    for canon, members in eclass.items():
      if a in members: a_class = canon
      if expr in members: expr_class = canon
    self.assertIs(a_class, expr_class)

  def test_rebuilding_propagates(self):
    """After a*0 merges with 0, rebuilding should create (0+a) which then matches x+0 -> x."""
    pm = PatternMatcher([
      (UPat.var("x") * 0, lambda x: x.const_like(0)),
      (UPat.var("x") + 0, lambda x: x),
    ])
    a = UOp.variable("a", 0, 10)
    expr = (a * 0) + a
    eclass = egraph_saturate(expr, pm)
    expr_cls = a_cls = None
    for canon, members in eclass.items():
      if expr in members: expr_cls = canon
      if a in members: a_cls = canon
    self.assertIsNotNone(expr_cls)
    self.assertIsNotNone(a_cls)
    self.assertIs(expr_cls, a_cls)

  def test_rebuilding_chain(self):
    """((a*0)+0)+b should simplify to b through multiple rebuild steps."""
    pm = PatternMatcher([
      (UPat.var("x") * 0, lambda x: x.const_like(0)),
      (UPat.var("x") + 0, lambda x: x),
    ])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    expr = ((a * 0) + 0) + b
    eclass = egraph_saturate(expr, pm)
    expr_cls = b_cls = None
    for canon, members in eclass.items():
      if expr in members: expr_cls = canon
      if b in members: b_cls = canon
    self.assertIsNotNone(expr_cls)
    self.assertIsNotNone(b_cls)
    self.assertIs(expr_cls, b_cls)

# *** test egraph_extract ***

class TestEGraphExtract(unittest.TestCase):
  def test_extract_identity(self):
    pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])
    a = UOp.variable("a", 0, 10)
    self.assertIs(egraph_extract(a + 0, pm), a)

  def test_extract_mul_identity(self):
    pm = PatternMatcher([(UPat.var("x") * 1, lambda x: x)])
    a = UOp.variable("a", 0, 10)
    self.assertIs(egraph_extract(a * 1, pm), a)

  def test_extract_const_fold(self):
    from tinygrad.uop.symbolic import symbolic_simple
    result = egraph_extract(UOp.const(dtypes.int, 2) + UOp.const(dtypes.int, 3), symbolic_simple)
    self.assertEqual(result.op, Ops.CONST)
    self.assertEqual(result.arg, 5)

  def test_extract_chain(self):
    pm = PatternMatcher([
      (UPat.var("x") + 0, lambda x: x),
      (UPat.var("x") * 1, lambda x: x),
    ])
    a = UOp.variable("a", 0, 10)
    self.assertIs(egraph_extract((a + 0) * 1, pm), a)

  def test_extract_no_change(self):
    pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    self.assertIs(egraph_extract(a + b, pm), a + b)

  def test_extract_prefers_cheaper(self):
    pm = PatternMatcher([(UPat.var("x") + UPat.var("x"), lambda x: x * 2)])
    a = UOp.variable("a", 0, 10)
    result = egraph_extract(a + a, pm)
    self.assertEqual(result.op, Ops.ADD)  # ADD cost 1 < MUL cost 2

  def test_extract_with_symbolic_simple(self):
    from tinygrad.uop.symbolic import symbolic_simple
    a = UOp.variable("a", 0, 10)
    self.assertIs(egraph_extract((a + 0) * 1, symbolic_simple), a)

  def test_combine_terms(self):
    from tinygrad.uop.symbolic import symbolic
    a = UOp.variable("a", 0, 10)
    result = egraph_extract(a * 3 + a * 4, symbolic)
    self.assertEqual(result.op, Ops.MUL)
    self.assertEqual(result.src[1].arg, 7)

  # *** tests that REQUIRE rebuilding ***

  def test_rebuild_mul_zero_plus(self):
    pm = PatternMatcher([
      (UPat.var("x") * 0, lambda x: x.const_like(0)),
      (UPat.var("x") + 0, lambda x: x),
    ])
    a = UOp.variable("a", 0, 10)
    self.assertIs(egraph_extract((a * 0) + a, pm), a)

  def test_rebuild_nested_zero(self):
    pm = PatternMatcher([
      (UPat.var("x") * 0, lambda x: x.const_like(0)),
      (UPat.var("x") + 0, lambda x: x),
    ])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    self.assertIs(egraph_extract(((a * 0) + 0) + b, pm), b)

  def test_rebuild_distribute_then_fold(self):
    pm = PatternMatcher([(UPat.var("x") * 0, lambda x: x.const_like(0))])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    result = egraph_extract((a + b) * 0, pm)
    self.assertEqual(result.op, Ops.CONST)
    self.assertEqual(result.arg, 0)

  def test_rebuild_symmetric(self):
    pm = PatternMatcher([
      (UPat.var("x") * 0, lambda x: x.const_like(0)),
      (UPat(GroupOp.Binary, src=(UPat((Ops.CONST, Ops.VCONST)),)*2, name="a"),
       lambda a: a.const_like(a.src[0].arg + a.src[1].arg) if a.op is Ops.ADD else None),
    ])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    result = egraph_extract((a * 0) + (b * 0), pm)
    self.assertEqual(result.op, Ops.CONST)
    self.assertEqual(result.arg, 0)

  def test_rebuild_with_real_rules(self):
    from tinygrad.uop.symbolic import symbolic_simple
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    self.assertIs(egraph_extract((a * 0) + (b * 1), symbolic_simple), b)

  def test_rebuild_deep_chain(self):
    pm = PatternMatcher([
      (UPat.var("x") * 0, lambda x: x.const_like(0)),
      (UPat.var("x") + 0, lambda x: x),
      (UPat(GroupOp.Binary, src=(UPat((Ops.CONST, Ops.VCONST)),)*2, name="a"),
       lambda a: a.const_like(a.src[0].arg + a.src[1].arg) if a.op is Ops.ADD else None),
    ])
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    c = UOp.variable("c", 0, 10)
    self.assertIs(egraph_extract(((a * 0) + (b * 0)) + c, pm), c)

# *** test cost model ***

class TestCostModel(unittest.TestCase):
  def test_const_is_free(self):
    self.assertEqual(node_cost(UOp.const(dtypes.int, 0)), 0)

  def test_add_is_cheap(self):
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    self.assertEqual(node_cost(a + b), 1)

  def test_div_is_expensive(self):
    a = UOp.variable("a", 0, 10).cast(dtypes.index)
    b = UOp.variable("b", 1, 10).cast(dtypes.index)
    self.assertEqual(node_cost(a // b), 5)

  def test_mul_more_than_add(self):
    a = UOp.variable("a", 0, 10)
    b = UOp.variable("b", 0, 10)
    self.assertGreater(node_cost(a * b), node_cost(a + b))

# *** test e-graph matches greedy rewrite ***

class TestEGraphVsGreedy(unittest.TestCase):
  def test_matches_greedy_identity(self):
    from tinygrad.uop.ops import graph_rewrite
    from tinygrad.uop.symbolic import symbolic_simple
    a = UOp.variable("a", 0, 10)
    greedy = graph_rewrite(a + 0, symbolic_simple)
    egraph = egraph_extract(a + 0, symbolic_simple)
    self.assertIs(greedy, egraph)

  def test_matches_greedy_const_fold(self):
    from tinygrad.uop.ops import graph_rewrite
    from tinygrad.uop.symbolic import symbolic_simple
    expr = UOp.const(dtypes.int, 10) + UOp.const(dtypes.int, 20)
    greedy = graph_rewrite(expr, symbolic_simple)
    egraph = egraph_extract(expr, symbolic_simple)
    self.assertEqual(greedy.op, Ops.CONST)
    self.assertEqual(egraph.op, Ops.CONST)
    self.assertEqual(greedy.arg, egraph.arg)

  def test_matches_greedy_double_identity(self):
    from tinygrad.uop.ops import graph_rewrite
    from tinygrad.uop.symbolic import symbolic_simple
    a = UOp.variable("a", 0, 10)
    expr = (a + 0) * 1
    self.assertIs(graph_rewrite(expr, symbolic_simple), a)
    self.assertIs(egraph_extract(expr, symbolic_simple), a)

# *** test e-graph beats greedy (phase-ordering problems) ***

# helper PMs that create phase-ordering traps
_pm_strength_reduce = PatternMatcher([
  # strength reduction x*2 -> x+x fires FIRST and destroys the x*c form needed by combine-terms
  (UPat.var('x') * UPat.cvar('c', vec=False), lambda x,c: x+x if c.arg == 2 else None),
  # combine terms: x*c0 + x*c1 -> x*(c0+c1) can only match if both sides are x*c
  (UPat.var('x') * UPat.cvar('c0') + UPat.var('x') * UPat.cvar('c1'), lambda x,c0,c1: x*(c0+c1)),
  # constant folding
  (UPat(GroupOp.Binary, src=(UPat((Ops.CONST, Ops.VCONST)),)*2, name='a'),
   lambda a: a.const_like(a.src[0].arg + a.src[1].arg) if a.op is Ops.ADD else
             a.const_like(a.src[0].arg * a.src[1].arg) if a.op is Ops.MUL else None),
  (UPat.var('x') + 0, lambda x: x),
  (UPat.var('x') * 1, lambda x: x),
])

_pm_shift_reduce = PatternMatcher([
  # strength reduction x*2 -> x<<1 fires FIRST and destroys the x*c form
  (UPat.var('x') * UPat.cvar('c', vec=False),
   lambda x,c: UOp(Ops.SHL, x.dtype, (x, x.const_like(1))) if c.arg == 2 else None),
  (UPat.var('x') * UPat.cvar('c0') + UPat.var('x') * UPat.cvar('c1'), lambda x,c0,c1: x*(c0+c1)),
  (UPat(GroupOp.Binary, src=(UPat((Ops.CONST, Ops.VCONST)),)*2, name='a'),
   lambda a: a.const_like(a.src[0].arg + a.src[1].arg) if a.op is Ops.ADD else
             a.const_like(a.src[0].arg * a.src[1].arg) if a.op is Ops.MUL else None),
  (UPat.var('x') + 0, lambda x: x),
  (UPat.var('x') * 1, lambda x: x),
])

_pm_strength_fold = PatternMatcher([
  # strength reduction x*2 -> x+x blocks two-stage folding (x*c1)*c2 -> x*(c1*c2)
  (UPat.var('x') * UPat.cvar('c', vec=False), lambda x,c: x+x if c.arg == 2 else None),
  ((UPat.var('x') * UPat.cvar('c1')) * UPat.cvar('c2'), lambda x,c1,c2: x*(c1*c2)),
  (UPat(GroupOp.Binary, src=(UPat((Ops.CONST, Ops.VCONST)),)*2, name='a'),
   lambda a: a.const_like(a.src[0].arg * a.src[1].arg) if a.op is Ops.MUL else None),
])

def _total_cost(u:UOp) -> int:
  return sum(node_cost(n) for n in u.toposort())

class TestEGraphBeatsGreedy(unittest.TestCase):
  """Tests where the e-graph finds a cheaper result than the greedy rewriter due to phase-ordering.

  The core problem: when Rule A fires first and transforms a node, it can destroy the pattern
  that Rule B needs to match. Rule B would have led to a cheaper result, but the greedy rewriter
  never tries it. The e-graph explores BOTH paths and picks the cheapest.
  """
  def test_strength_reduce_blocks_combine(self):
    """a*2 + a*3: strength reduction x*2->x+x destroys the x*c form needed by combine-terms x*c0+x*c1->x*(c0+c1)."""
    a = UOp.variable("a", 0, 10)
    expr = a * 2 + a * 3
    greedy = graph_rewrite(expr, _pm_strength_reduce)
    egraph = egraph_extract(expr, _pm_strength_reduce)
    # greedy: (a+a) + a*3 (cost 4) — strength reduction destroyed the a*2 pattern
    self.assertEqual(greedy.op, Ops.ADD)
    self.assertGreater(_total_cost(greedy), _total_cost(egraph))
    # egraph: a*5 (cost 2) — combine-terms wins because the e-graph explored both paths
    self.assertEqual(egraph.op, Ops.MUL)
    self.assertEqual(egraph.src[1].arg, 5)

  def test_shift_reduce_blocks_combine(self):
    """a*2 + a*3: shift reduction x*2->x<<1 also destroys the combine-terms pattern."""
    a = UOp.variable("a", 0, 10)
    expr = a * 2 + a * 3
    greedy = graph_rewrite(expr, _pm_shift_reduce)
    egraph = egraph_extract(expr, _pm_shift_reduce)
    self.assertEqual(greedy.op, Ops.ADD)
    self.assertGreater(_total_cost(greedy), _total_cost(egraph))
    self.assertEqual(egraph.op, Ops.MUL)
    self.assertEqual(egraph.src[1].arg, 5)

  def test_strength_reduce_chain(self):
    """a*2 + a*3 + a*4: strength reduction causes greedy to miss the combined a*9."""
    a = UOp.variable("a", 0, 10)
    expr = a * 2 + a * 3 + a * 4
    greedy = graph_rewrite(expr, _pm_strength_reduce)
    egraph = egraph_extract(expr, _pm_strength_reduce)
    self.assertGreater(_total_cost(greedy), _total_cost(egraph))

  def test_strength_reduce_blocks_two_stage_fold(self):
    """(a*2)*3: strength reduction x*2->x+x blocks two-stage constant folding (x*c1)*c2->x*(c1*c2)."""
    a = UOp.variable("a", 0, 10)
    expr = (a * 2) * 3
    greedy = graph_rewrite(expr, _pm_strength_fold)
    egraph = egraph_extract(expr, _pm_strength_fold)
    # greedy: (a+a)*3 (cost 3) — can't fold constants because *2 was rewritten to +
    self.assertGreater(_total_cost(greedy), _total_cost(egraph))
    # egraph: a*6 (cost 2) — two-stage folding path was explored
    self.assertEqual(egraph.op, Ops.MUL)
    self.assertEqual(egraph.src[1].arg, 6)

  def test_both_sides_strength_reduced(self):
    """a*2 + a*2: both sides get strength-reduced, blocking combine-terms."""
    a = UOp.variable("a", 0, 10)
    expr = a * 2 + a * 2
    greedy = graph_rewrite(expr, _pm_strength_reduce)
    egraph = egraph_extract(expr, _pm_strength_reduce)
    # greedy: (a+a)+(a+a) — both a*2 were rewritten before combine could fire
    # egraph: a*4 — combine-terms path was found
    self.assertEqual(egraph.op, Ops.MUL)
    self.assertEqual(egraph.src[1].arg, 4)
    # both have cost 2 here (shared subexpression), but egraph result is canonical
    self.assertLessEqual(_total_cost(egraph), _total_cost(greedy))

if __name__ == '__main__':
  unittest.main(verbosity=2)
