import unittest
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.codegen.simplify import pm_load_collapse, pm_reduce_load_collapse, reduce_collapse

class TestReduceMasks(unittest.TestCase):
  def test_masks_against_enumeration(self):
    r, i = UOp.range(8, 0), UOp.variable('i', 0, 7)
    # Each rule must preserve zero, one, and multiple hits, including negative numerators.
    value = (r*r+1).cast(dtypes.float)
    for expr, reference in (
      (((r+i+2).eq(5) & (r % 2 < 1)).where(value, 0), lambda k,j: k+j+2 == 5 and k % 2 == 0),
      ((r+i < 13).where(0, value), lambda k,j: k+j >= 13),
      (((r-i) % 8 < 1).where(value, 0), lambda k,j: (k-j) % 8 == 0),
      (((r+i) % 7 < 1).where(value, 0), lambda k,j: (k+j) % 7 == 0),
      (((r+i+1) % 32 < 1).where(value, 0), lambda k,j: (k+j+1) % 32 == 0),
      (((r+i) % 32 < 1).where(value, 0), lambda k,j: (k+j) % 32 == 0),
    ):
      with self.subTest(expr=expr.render()):
        result = graph_rewrite(expr.reduce(r, arg=Ops.ADD), pm_load_collapse)
        self.assertNotIn(Ops.REDUCE, [u.op for u in result.toposort()])
        for j in range(8):
          self.assertEqual(result.substitute({i:UOp.const(j)}).ssimplify(), sum(k*k+1 for k in range(8) if reference(k, j)))

  def test_nested_scopes(self):
    r, i, k, j = UOp.range(4, 0), UOp.range(4, 1), UOp.range(4, 2), UOp.range(3, 3)
    dependent, buf = UOp.range(r+2, 4), UOp.param(0, dtypes.float, (64,))
    for name, inner in (
      ('nonlinear', buf.index(16*r+k*k+1).reduce(k, arg=Ops.ADD)),
      ('two_scopes', (r.cast(dtypes.float)*buf.index(3*k+j).reduce(j, arg=Ops.ADD)).reduce(k, arg=Ops.ADD)),
      ('dependent_bound', buf.index(8*r+dependent).reduce(dependent, arg=Ops.ADD)),
    ):
      with self.subTest(name=name):
        red = r.eq(i).where(inner, 0).reduce(r, arg=Ops.ADD)
        result = reduce_collapse(red, red.src[0], pm_reduce_load_collapse)
        self.assertIsNotNone(result)
        self.assertIs(result.simplify(), inner.substitute({r:i}).simplify())
        self.assertEqual(set(result.ranges), {i})

  def test_nested_bound_becomes_zero_or_one(self):
    r = UOp.range(4, 0)
    k = UOp.range(r, 1)
    for selected in (0, 1):
      red = r.eq(selected).where((r+k).cast(dtypes.float).reduce(k, arg=Ops.ADD), 0).reduce(r, arg=Ops.ADD)
      result = reduce_collapse(red, red.src[0], pm_reduce_load_collapse)
      self.assertIsNotNone(result)
      self.assertFalse(result.ranges)
      if selected == 0:
        self.assertIs(result.op, Ops.REDUCE)
        self.assertEqual(result.src[1].src[0].ssimplify(), 0)
      else: self.assertEqual(result.ssimplify(), 1.0)

  def test_symbolic_bounds(self):
    n, m = UOp.variable('n', 0, 4), UOp.variable('m', 2, 4)
    r = UOp.range(n+1, 0)
    red = (r % m < 1).where((r*r+1).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
    result = graph_rewrite(red, pm_load_collapse)
    self.assertNotIn(Ops.REDUCE, [u.op for u in result.toposort()])
    for size in range(5):
      for modulus in range(2, 5):
        actual = result.substitute({n:UOp.const(size), m:UOp.const(modulus)}).ssimplify()
        self.assertEqual(actual, sum(k*k+1 for k in range(size+1) if k % modulus == 0))

  def test_unsafe_masks_not_collapsed(self):
    r, m = UOp.range(4096, 0), UOp.variable('m', -2, 2)
    for gate in (r % m < 1, r % 4 < 1, r.eq(r*r), (r+128).cast(dtypes.int8).eq(0)):
      red = gate.where((r*r+1).cast(dtypes.float), 0).reduce(r, arg=Ops.ADD)
      self.assertIsNone(pm_reduce_load_collapse.rewrite(red))
    r, i, buf = UOp.range(2, 0), UOp.range(2, 1), UOp.param(0, dtypes.int, (2,))
    gate = (r+i >= 1) & (2*r+i < 3)
    for cond, expr in ((gate, buf.index(r).after(UOp(Ops.NOOP))), (gate & (buf.index(r) < 1), r.cast(dtypes.float))):
      red = cond.where(expr, 0).reduce(r, arg=Ops.ADD)
      self.assertIs(graph_rewrite(red, pm_load_collapse), red)

if __name__ == '__main__': unittest.main()
