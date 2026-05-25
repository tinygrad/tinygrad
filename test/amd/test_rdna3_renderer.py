#!/usr/bin/env python3
"""Test harness for the RDNA3 ISA renderer. """
import unittest
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, dtypes, graph_rewrite
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.isa.rdna3 import RDNA3Renderer

@unittest.skipIf(RDNA3Renderer is None, "RDNA3Renderer not yet created")
class TestRDNA3Isel(unittest.TestCase):
  """Test ISel rules: UOp in → Ops.INS out, check .arg is the right instruction."""

  @classmethod
  def setUpClass(cls):
    from tinygrad.helpers import Target
    cls.renderer = RDNA3Renderer(Target('AMD', 'gfx1100'))

  def isel(self, uop: UOp) -> UOp:
    """Run a UOp through pre_isel + isel and return the rewritten graph."""
    rewritten = graph_rewrite(uop, self.renderer.pre_isel_matcher, IselContext(uop), bottom_up=True) if self.renderer.pre_isel_matcher else uop
    return graph_rewrite(rewritten, self.renderer.isel_matcher, IselContext(rewritten), bottom_up=True)

  def test_add_f32(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import v_add_f32_e32
    a = UOp.variable("a", 0, 0, dtypes.float)
    b = UOp.variable("b", 0, 0, dtypes.float)
    c = a + b
    n = self.isel(c)
    print(n)
    self.assertEqual(n.src[0].arg, ("a", 0, 0))
    self.assertEqual(n.src[1].arg, ("b", 0, 0))
    self.assertEqual(n.arg, v_add_f32_e32)
    self.assertEqual(n.op, Ops.INS)
    self.assertEqual(n.src[0].op, Ops.DEFINE_VAR)

  def test_mul_f32(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import v_mul_f32_e32
    a = UOp.variable("a", 0, 0, dtypes.float)
    b = UOp.variable("b", 0, 0, dtypes.float)
    c = a * b
    n = self.isel(c)
    self.assertEqual(n.src[0].arg, ("a", 0, 0))
    self.assertEqual(n.src[1].arg, ("b", 0, 0))
    self.assertEqual(n.arg, v_mul_f32_e32)
    self.assertEqual(n.op, Ops.INS)
    self.assertEqual(n.src[0].op, Ops.DEFINE_VAR)

  def test_sink(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
    a = UOp(Ops.SINK, dtypes.void, src=())
    n = self.isel(a)
    print(n.src[0])
    print(n.src[0].arg)
    self.assertEqual(n.src[0].arg, s_endpgm)

  def test_global_store_b32(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import global_store_b32
    base = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
    offset = UOp.variable("offset", 0, 255, dtypes.int)
    val = UOp.variable("val", 0, 0, dtypes.float)
    store = base.index(offset, ptr=True).store(val)

    n = self.isel(store)
    print(n.src[0])
    print(n.src[1])
    print(n.src[2])

    self.assertEqual(n.src[0].arg, ("offset", 0, 255))
    self.assertEqual(n.src[0].dtype, dtypes.int)
    self.assertEqual(n.src[1].arg, ("val", 0, 0))
    self.assertEqual(n.src[1].dtype, dtypes.float)
    self.assertEqual(n.src[2].arg, 0)
    self.assertEqual(n.src[2].dtype, dtypes.float.ptr(256))


    self.assertEqual(n.src[0].op, Ops.DEFINE_VAR)
    self.assertEqual(n.src[1].op, Ops.DEFINE_VAR)
    self.assertEqual(n.src[2].op, Ops.PARAM)



  def test_full_add_one_kernel(self):
    from tinygrad.codegen import full_rewrite_to_sink
    from tinygrad.uop.ops import KernelInfo

    buf_in = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
    buf_out = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 1)
    tidx = UOp.special(256, 'lidx0')
    load = buf_in.index(tidx).load()
    add = load + UOp.const(dtypes.float, 1.0)
    store = buf_out.index(tidx).store(add)
    sink = store.sink(arg=KernelInfo('add_one'))

    lowered = full_rewrite_to_sink(sink, self.renderer)
    print(lowered)

if __name__ == "__main__":
  unittest.main()
