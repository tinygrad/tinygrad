import random
import unittest
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.codegen.simplify import pm_load_collapse, pm_reduce_masks, pm_reduce_load_collapse, reduce_collapse


class TestReduceMasks(unittest.TestCase):
  def test_modulo_with_bounds(self):
    for n in (2, 3, 5, 32, 127, 4096):
      with self.subTest(n=n):
        r, i = UOp.range(n, 0), UOp.range(n, 1)
        buf = UOp.param(0, dtypes.float, (n,))
        gate = ((n*r+i >= n-1) & (n*r+i < n*n-1) & ((r+i) % (n-1) < 1)).simplify()
        red = gate.where(buf.index(r), 0).reduce(r, arg=Ops.ADD)
        result = graph_rewrite(red, pm_load_collapse).simplify()
        self.assertNotIn(Ops.REDUCE, [u.op for u in result.toposort()])
        self.assertEqual([u for u in result.toposort() if u.op is Ops.INDEX], [buf.index(n-1-i).simplify()])

  def test_negative_numerator(self):
    r, i = UOp.range(128, 0), UOp.range(128, 1)
    buf = UOp.param(0, dtypes.float, (128,))
    red = ((r-i) % 128 < 1).where(buf.index(r), 0).reduce(r, arg=Ops.ADD)
    self.assertIs(graph_rewrite(red, pm_load_collapse).simplify(), buf.index(i))

  def test_nested_reduce(self):
    r, i, k = UOp.range(2, 0), UOp.range(2, 1), UOp.range(7, 2)
    buf = UOp.param(0, dtypes.float, (14,))
    gate = (r+i >= 1) & (2*r+i < 3)
    inner = buf.index(7*r+k).reduce(k, arg=Ops.ADD)
    red = gate.where(inner, 0).reduce(r, arg=Ops.ADD)
    result = graph_rewrite(red, pm_load_collapse).simplify()
    self.assertEqual([u for u in result.toposort() if u.op is Ops.REDUCE], [inner.substitute({r:1-i}).simplify()])

  def test_nested_range_expressions_stay_in_scope(self):
    r, i, k = UOp.range(4, 0), UOp.range(4, 1), UOp.range(4, 2)
    buf = UOp.param(0, dtypes.float, (64,))
    inner = buf.index(16*r+k*k+1).reduce(k, arg=Ops.ADD)
    red = r.eq(i).where(inner, 0).reduce(r, arg=Ops.ADD)
    result = reduce_collapse(red, red.src[0], pm_reduce_load_collapse)
    self.assertIsNotNone(result)
    self.assertIs(result.simplify(), inner.substitute({r:i}).simplify())
    self.assertEqual(set(result.ranges), {i})

  def test_two_nested_scopes(self):
    r, i, k, j = UOp.range(4, 0), UOp.range(4, 1), UOp.range(4, 2), UOp.range(3, 3)
    buf = UOp.param(0, dtypes.float, (12,))
    # This inner reduction depends on k, but not r: it must not be abstracted out of k's scope.
    inner = (r.cast(dtypes.float) * buf.index(3*k+j).reduce(j, arg=Ops.ADD)).reduce(k, arg=Ops.ADD)
    red = r.eq(i).where(inner, 0).reduce(r, arg=Ops.ADD)
    result = reduce_collapse(red, red.src[0], pm_reduce_load_collapse)
    self.assertIsNotNone(result)
    self.assertIs(result.simplify(), inner.substitute({r:i}).simplify())
    self.assertEqual(set(result.ranges), {i})

  def test_nested_bound_depends_on_selected_range(self):
    r, i = UOp.range(4, 0), UOp.range(4, 1)
    k = UOp.range(r+2, 2)
    buf = UOp.param(0, dtypes.float, (32,))
    inner = buf.index(8*r+k).reduce(k, arg=Ops.ADD)
    red = r.eq(i).where(inner, 0).reduce(r, arg=Ops.ADD)
    result = reduce_collapse(red, red.src[0], pm_reduce_load_collapse)
    self.assertIsNotNone(result)
    self.assertIs(result.simplify(), inner.substitute({r:i}).simplify())
    self.assertEqual(set(result.ranges), {i})

  def test_symbolic_range_bound_not_abstracted(self):
    n = UOp.variable('n', 1, 7)
    r = UOp.range(n+1, 0)
    buf = UOp.param(0, dtypes.float, (8,))
    red = r.eq(n-1).where(buf.index(r), 0).reduce(r, arg=Ops.ADD)
    result = reduce_collapse(red, red.src[0], pm_reduce_load_collapse)
    self.assertIsNotNone(result)
    self.assertFalse(any(u.op is Ops.RANGE for u in result.toposort()))
    for size in range(1, 8):
      self.assertIs(result.substitute({n:UOp.const(size)}).simplify(), buf.index(UOp.const(size-1)))

  def test_nested_bound_becomes_zero_or_one(self):
    r = UOp.range(4, 0)
    k = UOp.range(r, 1)
    inner = (r+k).cast(dtypes.float).reduce(k, arg=Ops.ADD)
    for selected in (0, 1):
      with self.subTest(selected=selected):
        red = r.eq(selected).where(inner, 0).reduce(r, arg=Ops.ADD)
        result = reduce_collapse(red, red.src[0], pm_reduce_load_collapse)
        self.assertIsNotNone(result)
        self.assertFalse(result.ranges)
        if selected == 0:
          self.assertIs(result.op, Ops.REDUCE)
          self.assertIs(result.src[1].op, Ops.RANGE)
          self.assertEqual(result.src[1].src[0].ssimplify(), 0)
        else:
          self.assertEqual(result.ssimplify(), 1.0)

  def test_range_dependent_equality_not_a_single_selection(self):
    r = UOp.range(2, 0)
    red = r.eq(r*r).where(UOp.const(1.0), 0).reduce(r, arg=Ops.ADD)
    self.assertIsNone(reduce_collapse(red, red.src[0], pm_reduce_load_collapse))

  def test_empty_selection(self):
    r, i = UOp.range(5, 0), UOp.range(5, 1)
    buf = UOp.param(0, dtypes.float, (5,))
    red = ((r+i+1) % 32 < 1).where(buf.index(r), 0).reduce(r, arg=Ops.ADD)
    self.assertIs(graph_rewrite(red, pm_load_collapse).simplify(), UOp.const(0, dtypes.float))

  def test_conditional_selection(self):
    r, i = UOp.range(32, 0), UOp.variable('i', 0, 32)
    buf = UOp.param(0, dtypes.float, (32,))
    gate = ((r+i) % 64 < 1)
    result = graph_rewrite(gate.where(buf.index(r), 0).reduce(r, arg=Ops.ADD), pm_load_collapse)
    self.assertNotIn(Ops.REDUCE, [u.op for u in result.toposort()])
    self.assertIs(result.substitute({i:UOp.const(0)}).simplify(), buf.index(UOp.const(0)))
    for index in (1, 16, 32):
      self.assertIs(result.substitute({i:UOp.const(index)}).simplify(), UOp.const(0, dtypes.float))

  def test_multiple_hits_preserved(self):
    r, i = UOp.range(32, 0), UOp.range(32, 1)
    red = ((r+i) % 31 < 1).where((r*r+1).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
    result = graph_rewrite(red, pm_load_collapse)
    # At i=0 and i=31 both endpoints contribute: dropping either is incorrect.
    for j in range(32):
      expected = sum(k*k+1 for k in range(32) if (k+j) % 31 == 0)
      self.assertEqual(result.substitute({i:UOp.const(j)}).ssimplify(), expected)

  def test_affine_equality(self):
    r, i = UOp.range(7, 0), UOp.variable('i', -10, 10)
    gate = (r+i+2).eq(5) & (r % 2 < 1)
    result = graph_rewrite(gate.where((r*r+1).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD), pm_load_collapse)
    for j in range(-10, 11):
      expected = sum(k*k+1 for k in range(7) if k+j+2 == 5 and k % 2 == 0)
      self.assertEqual(result.substitute({i:UOp.const(j)}).ssimplify(), expected)

  def test_short_interval_multiple_hits(self):
    r, i = UOp.range(8, 0), UOp.variable('i', 0, 3)
    red = (r+i < 9).where(0, (r*r+1).cast(dtypes.float)).reduce(r, arg=Ops.ADD)
    result = graph_rewrite(red, pm_load_collapse)
    for j in range(4):
      expected = sum(k*k+1 for k in range(8) if k+j >= 9)
      self.assertEqual(result.substitute({i:UOp.const(j)}).ssimplify(), expected)

  def test_symbolic_modulus(self):
    r, m = UOp.range(4, 0), UOp.variable('m', 2, 4)
    red = (r % m < 1).where((r*r+1).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
    result = graph_rewrite(red, pm_load_collapse)
    for modulus in (2, 3, 4):
      expected = sum(k*k+1 for k in range(4) if k % modulus == 0)
      self.assertEqual(result.substitute({m:UOp.const(modulus)}).ssimplify(), expected)

  def test_nonpositive_modulus_not_expanded(self):
    r, m = UOp.range(4, 0), UOp.variable('m', -2, 2)
    red = (r % m < 1).where(r.cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
    self.assertIsNone(pm_reduce_masks.rewrite(red))

  def test_long_quotient_interval_not_expanded(self):
    r = UOp.range(4096, 0)
    red = (r % 4 < 1).where((r*r).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
    self.assertIs(graph_rewrite(red, pm_load_collapse), red)

  def test_symbolic_extent(self):
    n, i = UOp.variable('n', 0, 128), UOp.variable('i', 0, 127)
    r = UOp.range(n, 0)
    red = ((r-i) % 256 < 1).where((r+1).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
    result = graph_rewrite(red, pm_load_collapse)
    self.assertNotIn(Ops.REDUCE, [u.op for u in result.toposort()])
    for size in (0, 1, 2, 64, 128):
      for index in (0, 1, 63, 127):
        actual = result.substitute({n:UOp.const(size), i:UOp.const(index)}).ssimplify()
        self.assertEqual(actual, index+1 if index < size else 0)

  def test_against_enumeration(self):
    rng, collapsed = random.Random(0), 0
    for _ in range(100):
      n, period, offset = rng.randrange(2, 33), rng.randrange(2, 33), rng.randrange(-32, 33)
      lower, upper = sorted((rng.randrange(-32, n*n+1), rng.randrange(-32, n*n+1)))
      r, i = UOp.range(n, 0), UOp.variable('i', 0, 3)
      gate = ((n*r+i >= lower) & (n*r+i < upper) & ((r+i+offset) % period < 1)).simplify()
      red = gate.where((r*r+1).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
      result = graph_rewrite(red, pm_load_collapse)
      if any(u.op is Ops.REDUCE for u in result.toposort()): continue
      collapsed += 1
      for j in range(4):
        expected = sum(k*k+1 for k in range(n) if lower <= n*k+j < upper and (k+j+offset) % period == 0)
        self.assertEqual(result.substitute({i:UOp.const(j)}).ssimplify(), expected)
    self.assertGreater(collapsed, 10)

  def test_non_affine_unchanged(self):
    r = UOp.range(32, 0)
    red = (r*r != 0).where(0, r.cast(dtypes.float)).reduce(r, arg=Ops.ADD)
    self.assertIsNone(pm_reduce_masks.rewrite(red))

  def test_effects_and_data_dependent_masks_unchanged(self):
    r, i = UOp.range(2, 0), UOp.range(2, 1)
    buf = UOp.param(0, dtypes.int, (2,))
    gate = (r+i >= 1) & (2*r+i < 3)
    for cond, expr in ((gate, buf.index(r).after(UOp(Ops.NOOP))), (gate & (buf.index(r) < 1), r.cast(dtypes.float))):
      red = cond.where(expr, 0).reduce(r, arg=Ops.ADD)
      self.assertIs(graph_rewrite(red, pm_load_collapse), red)


if __name__ == '__main__':
  unittest.main()
