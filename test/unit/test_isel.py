import unittest
from tinygrad.uop import X86Ops, Ops
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite
from tinygrad.renderer.x86 import X86Renderer
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.helpers import SPEC

@unittest.skipIf(SPEC > 1, "x86 spec not supported in full_spec")
class TestIselX86(unittest.TestCase):
  def isel_rewrite(self, x:UOp):
    x = graph_rewrite(x, X86Renderer().pre_isel_matcher)
    return graph_rewrite(x, X86Renderer().isel_matcher, IselContext(x), bottom_up=True)

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

  # the geps become part of the immediate in the instruction
  def test_vshufps_same_src(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    vec = UOp(Ops.VECTORIZE, a.dtype, (a.gep(3), a.gep(2), a.gep(1), a.gep(0)))
    n = self.isel_rewrite(vec)
    self.assertTrue(n.op is X86Ops.VSHUFPS and n.src[0] is a and n.src[1] is a and n.src[2].arg == 27)

  def test_vshufps_diff_src(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    b = UOp.variable("b", 0, 0, dtypes.float32)
    vec = UOp(Ops.VECTORIZE, a.dtype, (a.gep(2), a.gep(3), b, b))
    n = self.isel_rewrite(vec)
    self.assertTrue(n.op is X86Ops.VSHUFPS and n.src[0] is a and n.src[1] is b and n.src[2].arg == 14)

  def test_vinsertps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    b = UOp.variable("b", 0, 0, dtypes.float32.vec(4))
    c = UOp.variable("c", 0, 0, dtypes.float32.vec(4))
    d = UOp.variable("d", 0, 0, dtypes.float32)
    vec = UOp(Ops.VECTORIZE, dtypes.float32.vec(4), (a.gep(0), b.gep(0), c.gep(0), d))
    n = self.isel_rewrite(vec)
    self.assertTrue(n.op is X86Ops.VINSERTPS and len(n.src) == 3)
    self.assertTrue(n.src[0].op is X86Ops.VINSERTPS and n.src[1] is d and n.src[2].arg == 48)
    n = n.src[0]
    self.assertTrue(n.src[0].op is X86Ops.VINSERTPS and n.src[1] is c and n.src[2].arg == 32)
    n = n.src[0]
    # first gep is just moving the first element from a reg to another which does nothing
    self.assertTrue(n.src[0] is a and n.src[1] is b and n.src[2].arg == 16)

  # 8bit displacement should be used when possible
  def test_load_8bit_disp(self):
    offset = UOp.variable("a", 0, 0, dtypes.int32) + UOp.const(dtypes.int32, 1)
    index = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), arg=0).index(offset, ptr=True)
    load = index.load()
    n = self.isel_rewrite(load)
    self.assertTrue(n.src[2].op is X86Ops.IMM and n.src[2].dtype is dtypes.int8)

  def test_fuse_index(self):
    var = UOp.variable("a", 0, 0, dtypes.int32)
    offset = var + UOp.const(dtypes.int32, 1)
    index = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), arg=0).index(offset, ptr=True)
    load = index.load()
    n = self.isel_rewrite(load)
    self.assertTrue(n.src[1] is var)

  def test_fuse_load(self):
    offset = UOp.variable("a", 0, 0, dtypes.int32) + UOp.const(dtypes.int32, 1)
    index = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), arg=0).index(offset, ptr=True)
    load = index.load()
    add = offset + load
    n = self.isel_rewrite(add)
    self.assertTrue(len(n.src) == 4)

  # don't fuse when used multiple times
  def test_dont_fuse_load(self):
    offset = UOp.variable("a", 0, 0, dtypes.int32) + UOp.const(dtypes.int32, 1)
    index = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), arg=0).index(offset, ptr=True)
    load = index.load()
    add1 = offset + load
    add2 = add1 + load
    n = self.isel_rewrite(add2)
    self.assertTrue(len(n.src) == 2)

  def test_dont_fuse_load_same_user(self):
    offset = UOp.variable("a", 0, 0, dtypes.int32) + UOp.const(dtypes.int32, 1)
    index = UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), arg=0).index(offset, ptr=True)
    load = index.load()
    add = load + load
    n = self.isel_rewrite(add)
    self.assertTrue(len(n.src) == 2)

  # test noop has same reg as src, this is because noops aren't instructions but still need to be part of the graph
  # as they may have different dtype from src and the correct dtype is required to encode the correct instruction
  # by giving them the same reg as src we ensure they share the same live range
  @unittest.skip("hmmm")
  def test_noop(self):
    noop = UOp(Ops.NOOP, dtypes.int32, (UOp(Ops.DEFINE_GLOBAL, dtypes.int32.ptr(), arg=0),))
    n = self.isel_rewrite(noop)
    self.assertTrue(isinstance(n.arg, Register) and n.arg == n.src[0].arg)

  # TODO: don't use fmadd if uop used multiple times
  # TODO: might want to check that load isn't part of another range when fusing

if __name__ == "__main__":
  unittest.main()