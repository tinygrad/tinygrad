import unittest
from tinygrad.uop.ops import UOp, AxisType, graph_rewrite
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.schedule.multi import multi_pm, factor_span, is_owner, resolve_axis_index

def rng(n, i, at=AxisType.LOCAL): return UOp.range(n, i, at)
def nodes(fs): return [(('o', int(factor_span(f))) if is_owner(f) else ('l', int(factor_span(f)))) for f in fs]

class TestFactorLayout(unittest.TestCase):
  """The factor-list algebra of UNSHARD: every axis carries an ordered factor list (owner factors with RANGEs,
  local const factors, spans = vmax+1). Reshape splits/merges the lists; this is how strided, block-inserted,
  and multi-owner shard layouts are expressed."""

  def frag(self, shape, axes, rngs):
    return UOp.placeholder(shape, dtypes.float32, 0, AddrSpace.REG).unshard(axes, rngs)

  def resolve(self, multi):
    return graph_rewrite(multi, multi_pm)

  def test_contiguous_device_layout(self):
    # unshard(axis) builds the canonical contiguous layout [rng, s]: full = count * s
    x = UOp.placeholder((2, 4), dtypes.float32, 0).unshard((0,), (rng(2, 0, AxisType.DEVICE),))
    self.assertEqual(x.shape, (4, 4))
    self.assertEqual([nodes(fs) for fs in x.factors], [[('o', 2), ('l', 2)], [('l', 4)]])
    self.assertEqual(x.axis, 0)
    self.assertEqual(x.bounds, ((0, 2), (2, 4)))

  def test_fragment_tile_to_tile_strided(self):
    # (TM, TY, TX, TN) -> (BLOCK_M, BLOCK_N): merging (TM, TY) puts the owner in the middle -> strided rows,
    # merging (TX, TN) keeps it major -> contiguous cols
    ty, tx = rng(8, 0), rng(16, 1)
    frag = self.frag((8, 1, 1, 4), (1, 2), (ty, tx))
    self.assertEqual(frag.shape, (8, 8, 16, 4))
    out = self.resolve(frag.reshape(64, 64))
    self.assertEqual(out.shape, (64, 64))
    self.assertEqual(nodes(out.factors[0]), [('l', 8), ('o', 8)])   # rows strided: pos = l*8 + ty
    self.assertEqual(nodes(out.factors[1]), [('o', 16), ('l', 4)])  # cols contiguous: pos = tx*4 + l
    self.assertEqual(out.src[0].shape, (8, 4))

  def test_index_resolution_strided_and_contiguous(self):
    ty, tx = rng(8, 0), rng(16, 1)
    frag = self.frag((8, 1, 1, 4), (1, 2), (ty, tx))
    out = self.resolve(frag.reshape(64, 64))
    ir, jj = rng(8, 2, AxisType.LOOP), rng(4, 3, AxisType.LOOP)
    # rows: full idx = ir*8 + ty -> local ir (strided); cols: full idx = tx*4 + jj -> local jj (contiguous)
    self.assertEqual(resolve_axis_index(ir*8 + ty, out.factors[0]) .ssimplify(), ir)
    self.assertEqual(resolve_axis_index(tx*4 + jj, out.factors[1]) .ssimplify(), jj)
    # wrong ownership does not resolve
    self.assertIsNone(resolve_axis_index(ir + ty, out.factors[0]))

  def test_middle_insert_flatten(self):
    # buf(4,1,4).unshard(1, rng(4)).flatten(): the shard lands in the middle of the merged axis
    r = rng(4, 0)
    mid = self.frag((4, 1, 4), (1,), (r,))
    out = self.resolve(mid.reshape(64,))
    self.assertEqual(out.shape, (64,))
    self.assertEqual(nodes(out.factors[0]), [('l', 4), ('o', 4), ('l', 4)])  # pos = a*16 + r*4 + c
    self.assertEqual(out.src[0].shape, (16,))
    # resolution through the middle insert
    a, c = rng(4, 1, AxisType.LOOP), rng(4, 2, AxisType.LOOP)
    idx = a*16 + r*4 + c
    local = resolve_axis_index(idx, out.factors[0])
    self.assertEqual(local.ssimplify(), (a*4 + c).ssimplify())

  def test_two_owners_one_axis(self):
    # (1,4,1) sharded on axes 0 and 2 -> flatten puts two owner factors on one axis
    ra, rb = rng(4, 0), rng(4, 1)
    two = UOp.placeholder((1, 4, 1), dtypes.float32, 0, AddrSpace.REG).unshard((0, 2), (ra, rb))
    out = self.resolve(two.reshape(64,))
    self.assertEqual(nodes(out.factors[0]), [('o', 4), ('l', 4), ('o', 4)])  # pos = ra*16 + l*4 + rb
    l = rng(4, 2, AxisType.LOOP)
    local = resolve_axis_index(ra*16 + l*4 + rb, out.factors[0])
    self.assertEqual(local.ssimplify(), l)

  def test_owner_span_split(self):
    # (1,) unshard rng(4) -> reshape (2,2): the owner span 4 splits 4 = 2*2 as (r//2, r%2)
    r = rng(4, 0)
    one = self.frag((1,), (0,), (r,))
    out = self.resolve(one.reshape(2, 2))
    self.assertEqual(out.shape, (2, 2))
    f0, f1 = out.factors[0][0], out.factors[1][0]
    self.assertTrue(is_owner(f0) and is_owner(f1))
    self.assertEqual(int(factor_span(f0)), 2)
    self.assertEqual(int(factor_span(f1)), 2)
    # the split coordinates are exactly the logical indices this shard owns: both resolve to local 0
    self.assertEqual(resolve_axis_index(f0, out.factors[0]).ssimplify(), UOp.const(None, 0).ssimplify())
    self.assertEqual(resolve_axis_index(f1, out.factors[1]).ssimplify(), UOp.const(None, 0).ssimplify())
    # a plain loop index is not the owner coordinate of either shard
    a = rng(2, 1, AxisType.LOOP)
    self.assertIsNone(resolve_axis_index(a, out.factors[0]))

  def test_local_split_and_merge(self):
    # (4, 8) local -> (8, 4): pure local factor split, no owners
    x = UOp.placeholder((4, 8), dtypes.float32, 0)
    uns = x.unshard((0,), (rng(2, 0, AxisType.DEVICE),))
    # full (8, 8): [rng, L4] [L8] -> (8, 8) major merge of L4+L8 into L32 on axis 1
    out = self.resolve(uns.reshape(8, 8))
    self.assertEqual(nodes(out.factors[0]), [('o', 2), ('l', 4)])
    self.assertEqual(nodes(out.factors[1]), [('l', 8)])
    # full (4, 16): split L8 of axis 1 into L2+L4
    out2 = self.resolve(uns.reshape(4, 16))
    self.assertEqual(nodes(out2.factors[1]), [('l', 16)])

  def test_reshape_moved_items_between_shards(self):
    # reshapes that would cut a factor asymmetrically raise: full will-change ownership of individual elements
    uns = self.frag((4, 4), (0,), (rng(4, 0),))    # full (16, 4), factors [rng(4), L4] [L4]
    # (2,4,2,4): legal, cuts the owner span 4 = 2*2 and the L4s evenly
    out = self.resolve(uns.reshape(2, 4, 2, 4))
    self.assertEqual(out.shape, (2, 4, 2, 4))
    # prod mismatch raises
    with self.assertRaises(ValueError):
      self.resolve(uns.reshape(3, 8))
    # (24, 6)-full: splitting evenly is fine ((12, 12) resolves to [rng,L3],[L2,L6])
    uns2 = self.frag((6, 6), (0,), (rng(4, 0),))
    self.assertEqual(self.resolve(uns2.reshape(12, 12)).shape, (12, 12))
    # but needing a multiplicative prefix that doesn't divide a factor raises "moved items between shards".
    # shard (6, 4) sharded count 6 -> full (36, 4); (4, 36) needs to prefix-take 4 out of the span-6 owner factor
    uns3 = self.frag((6, 4), (0,), (rng(6, 0),))
    with self.assertRaisesRegex(RuntimeError, "moved items between shards"):
      self.resolve(uns3.reshape(4, 36))
    # cutting a span-6 owner coordinate into an axis of 4 also raises
    uns4 = self.frag((2,), (0,), (rng(6, 0),))
    with self.assertRaisesRegex(RuntimeError, "moved items between shards"):
      self.resolve(uns4.reshape(4, 3))

  def test_bounds(self):
    dev = (rng(2, 0, AxisType.DEVICE),)
    x = UOp.placeholder((2, 4), dtypes.float32, 0).unshard((0,), dev)
    self.assertEqual(x.bounds, ((0, 2), (2, 4)))


if __name__ == "__main__":
  unittest.main()
