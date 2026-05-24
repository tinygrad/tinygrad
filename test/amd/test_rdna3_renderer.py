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

  def test_const_float(self):
    """Ops.CONST float32 → v_mov_b32_e32 (or inline constant)"""
    c = UOp.const(dtypes.float, 1.0)
    result = self.isel(c)
    self.assertEqual(result.op, Ops.INS)
    # TODO: check result.arg is the right instruction

  def test_const_int(self):
    """Ops.CONST int32 → s_mov_b32 (scalar constant)"""
    c = UOp.const(dtypes.int, 42)
    result = self.isel(c)
    self.assertEqual(result.op, Ops.INS)

  def test_sink(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
    a = UOp(Ops.SINK, dtypes.void, src=())
    n = self.isel(a)
    print(n.src[0])
    print(n.src[0].arg)
    self.assertEqual(n.src[0].arg, s_endpgm())

  
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
