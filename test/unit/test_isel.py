import unittest
from tinygrad.uop import X86Ops, Ops
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite
from tinygrad.renderer.x86 import X86Renderer
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.helpers import SPEC

@unittest.skipIf(SPEC > 1, "x86 spec not supported in full_spec")
class TestIselX86(unittest.TestCase):
  def isel_rewrite(self, x:UOp): return graph_rewrite(x, X86Renderer().isel_matcher, IselContext(x), bottom_up=True)

  def test_cmove(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    b = UOp.variable("b", 0, 0, dtypes.int32)
    c = (a < b).where(a, b)
    d = (a != b).where(a, b)
    f = c + d
    n = self.isel_rewrite(f)
    self.assertTrue(n.src[0].op is X86Ops.CMOVL and n.src[1].op is X86Ops.CMOVNE)
    # both comparisons become the same instruction
    self.assertTrue(n.src[0].src[2] == n.src[1].src[2] and n.src[0].src[2].op is X86Ops.CMP)

  def test_cmove_and_blend_with_float_cmp(self):
    a = UOp.variable("a", 0, 0, dtypes.float32)
    b = UOp.variable("b", 0, 0, dtypes.float32)
    c = a < b
    d = c.where(a.cast(dtypes.int32), b.cast(dtypes.int32))
    e = c.where(a, b)
    f = d + e
    n = self.isel_rewrite(f)
    # the comparison instruction depends on the user, int cmove uses flag while float cmove uses mask
    # so both flag producing and mask producing comparisons must be present
    self.assertTrue(n.src[0].op is X86Ops.CMOVB and n.src[0].src[2].op is X86Ops.VUCOMISS)
    self.assertTrue(n.src[1].op is X86Ops.VBLENDVPS and n.src[1].src[2].op is X86Ops.VCMPSS and n.src[1].src[2].src[2].arg == 1)

  # lower 2 32 bits must come from the same register and upper 2 32 bits must come from the same register
  def test_vshufps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    b = UOp.variable("b", 0, 0, dtypes.float32.vec(4))
    c = UOp.variable("c", 0, 0, dtypes.float32)
    d = UOp.variable("d", 0, 0, dtypes.float32)
    # shuffle between 2 vectors
    n = self.isel_rewrite(UOp(Ops.VECTORIZE, a.dtype, (a.gep(0), a.gep(1), b.gep(2), b.gep(3))))
    self.assertTrue(n.op is X86Ops.VSHUFPS)
    # shuffle between 2 scalars
    n = self.isel_rewrite(UOp(Ops.VECTORIZE, a.dtype, (c, c, d, d)))
    self.assertTrue(n.op is X86Ops.VSHUFPS)
    # shuffle between vector and scalar
    n = self.isel_rewrite(UOp(Ops.VECTORIZE, a.dtype, (a.gep(0), a.gep(1), c, c)))
    self.assertTrue(n.op is X86Ops.VSHUFPS)
    # shuffle between 1 vector
    n = self.isel_rewrite(UOp(Ops.VECTORIZE, a.dtype, (a.gep(1), a.gep(2), a.gep(3), a.gep(0))))
    self.assertTrue(n.op is X86Ops.VSHUFPS and n.src[0] is n.src[1])
    # a shuffle between 1 scalar is just a broadcast and matches X86Ops.VBROADCASTSS to allow for load fusion

   # this is the fallback slow VECTORIZE, 1 vinsertps per src in VECTORIZE
  def test_vinsertps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    b = UOp.variable("b", 0, 0, dtypes.float32.vec(4))
    c = UOp.variable("c", 0, 0, dtypes.float32.vec(4))
    d = UOp.variable("e", 0, 0, dtypes.float32)
    # pack 1 from vector and 1 from scalar, moving 0th element to position 0 does nothing so only 1 vinsertps is generated
    n = self.isel_rewrite(UOp(Ops.VECTORIZE, dtypes.float32.vec(2), (a.gep(0), d)))
    self.assertTrue(n.op is X86Ops.VINSERTPS and n.src[0].op is X86Ops.DEFINE_REG)
    # interleaved shuffle between 2 vectors
    n = self.isel_rewrite(UOp(Ops.VECTORIZE, a.dtype, (a.gep(0), b.gep(1), a.gep(2), b.gep(3))))
    self.assertTrue(n.op is X86Ops.VINSERTPS)
    # shuffle between 4 sources
    n = self.isel_rewrite(UOp(Ops.VECTORIZE, a.dtype, (a.gep(3), b.gep(2), c.gep(1), d)))
    self.assertTrue(n.op is X86Ops.VINSERTPS)

  # complex address is [base + index*scale + displacement]
  def test_complex_address(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    load = UOp(Ops.PARAM, dtypes.int32.ptr(), arg=0).index(a + 1, ptr=True).load()
    n = self.isel_rewrite(load)
    # base is PARAM, index is "a"
    self.assertTrue(n.src[0].op is X86Ops.DEFINE_REG and n.src[1].op is X86Ops.DEFINE_REG)
    # displacement is the constant in "a" scaled to the buffer element size, dtype is int8 when the value fits otherwise int32
    self.assertTrue(n.src[2].op is X86Ops.IMM and n.src[2].dtype is dtypes.int8 and n.src[2].arg == 4)

  def test_fuse_load(self):
    load1 = UOp(Ops.PARAM, dtypes.int32.ptr(), arg=0).index(UOp.const(dtypes.int32, 0), ptr=True).load()
    load2 = UOp(Ops.PARAM, dtypes.int32.ptr(), arg=0).index(UOp.const(dtypes.int32, 1), ptr=True).load()
    n = self.isel_rewrite(load1 + load2)
    self.assertTrue(len(n.src) == 4)

  # don't fuse when used multiple times
  def test_dont_fuse_load_diff_users(self):
    load = UOp(Ops.PARAM, dtypes.int32.ptr(), arg=0).index(UOp.const(dtypes.int32, 0), ptr=True).load()
    add = load + 1
    n = self.isel_rewrite(add + load)
    self.assertTrue(len(n.src) == 2)

  def test_dont_fuse_load_same_user(self):
    load = UOp(Ops.PARAM, dtypes.int32.ptr(), arg=0).index(UOp.const(dtypes.int32, 0), ptr=True).load()
    n = self.isel_rewrite(load * load)
    self.assertTrue(len(n.src) == 2)

  # test noop has same reg as src, this is because noops aren't instructions but still need to be part of the graph
  # as they may have different dtype from src and the correct dtype is required to encode the correct instruction
  # by giving them the same reg as src we ensure they share the same live range
  @unittest.skip("hmmm")
  def test_noop(self):
    noop = UOp(Ops.NOOP, dtypes.int32, (UOp(Ops.PARAM, dtypes.int32.ptr(), arg=0),))
    n = self.isel_rewrite(noop)
    self.assertTrue(isinstance(n.arg, Register) and n.arg == n.src[0].arg)

  # TODO: might want to check that load isn't part of another range when fusing

if __name__ == "__main__":
  unittest.main()