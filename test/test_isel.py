import unittest
from tinygrad.device import Device
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite, Ops
from tinygrad.helpers import X86
from tinygrad.renderer.isa import X86Renderer

@unittest.skipUnless(Device.DEFAULT == "CPU" and X86, "only on x86 backend")
class TestIselX86(unittest.TestCase):
  def test_cmove(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    b = UOp.variable("b", 0, 0, dtypes.int32)
    c = (a < b).where(a, b)
    d = (a != b).where(a, b)
    f = c + d
    n = graph_rewrite(f, X86Renderer().isel_matcher, bottom_up=True)
    self.assertTrue(n.src[0].op is Ops.CMOVL and n.src[1].op is Ops.CMOVNE)

  def test_vshufps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32)
    vec = a.broadcast(4)
    n = graph_rewrite(vec, X86Renderer().isel_matcher, bottom_up=True)
    self.assertTrue(n.op is Ops.VSHUFPS and n.src[0] is a and n.src[1] is a and n.src[2].arg == 0)

  def test_vshufps_rm_geps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    vec = UOp(Ops.VECTORIZE, a.dtype, (a.gep(0), a.gep(1), a.gep(2), a.gep(3)))
    n = graph_rewrite(vec, X86Renderer().isel_matcher, bottom_up=True)
    self.assertTrue(n.op is Ops.VSHUFPS and n.src[0] is a and n.src[1] is a and n.src[2].arg == 228)

  def test_vshufps_diff_srcs(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    b = UOp.variable("b", 0, 0, dtypes.float32.vec(4))
    vec = UOp(Ops.VECTORIZE, a.dtype, (a.gep(0), a.gep(1), b.gep(2), b.gep(3)))
    n = graph_rewrite(vec, X86Renderer().isel_matcher, bottom_up=True)
    self.assertTrue(n.op is Ops.VSHUFPS and n.src[0] is a and n.src[1] is b and n.src[2].arg == 228)

  def test_vinsertps(self):
    a = UOp.variable("a", 0, 0, dtypes.float32.vec(4))
    b = UOp.variable("b", 0, 0, dtypes.float32.vec(4))
    c = UOp.variable("c", 0, 0, dtypes.float32.vec(4))
    d = UOp.variable("d", 0, 0, dtypes.float32.vec(4))
    vec = UOp(Ops.VECTORIZE, a.dtype, (a.gep(0), b.gep(0), c.gep(0), d.gep(0)))
    n = graph_rewrite(vec, X86Renderer().isel_matcher, bottom_up=True)
    self.assertTrue(n.op is Ops.VINSERTPS and len(n.src) == 3)
    self.assertTrue(n.src[0].op is Ops.VINSERTPS and n.src[1] is d and n.src[2].arg == 48)
    n = n.src[0]
    self.assertTrue(n.src[0].op is Ops.VINSERTPS and n.src[1] is c and n.src[2].arg == 32)
    n = n.src[0]
    self.assertTrue(n.src[0].op is Ops.VINSERTPS and n.src[1] is b and n.src[2].arg == 16)
    n = n.src[0]
    self.assertTrue(n.src[0] is a and n.src[1] is a and n.src[2].arg == 0)

  # this is an example of a dependency cycle, this requires rematerialization in the scheduler
  # the last where can't be scheduled until another cmp that overwrites the flags, meaning c1 must be scheduled twice
  # this is required to accurately model cmp/cmove as UOps
  @unittest.skip("")
  def test_cmove_cycle(self):
    a = UOp.variable("a", 0, 0, dtypes.int32)
    b = UOp.variable("b", 0, 0, dtypes.int32)
    c = UOp.variable("c", 0, 0, dtypes.int32)
    d = UOp.variable("d", 0, 0, dtypes.int32)
    e = UOp.variable("e", 0, 0, dtypes.int32)
    f = UOp.variable("f", 0, 0, dtypes.int32)
    g = UOp.variable("g", 0, 0, dtypes.int32)
    h = UOp.variable("h", 0, 0, dtypes.int32)
    c1 = (a > b)
    v1 = c1.where(c, d)
    v2 = (v1 > e).where(f, g)
    v3 = c1.where(v2, h)

if __name__ == '__main__':
  unittest.main()

    
		