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

  def test_global_load_b32(self):
    from tinygrad.runtime.autogen.amd.rdna3.ins import global_load_b32
    base = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
    offset = UOp.variable("offset", 0, 255, dtypes.int)
    load = base.index(offset, ptr=True).load()

    n = self.isel(load)
    print(n.src[0])
    print(n.src[1])

    self.assertEqual(n.src[0].arg, ("offset", 0, 255))
    self.assertEqual(n.src[0].dtype, dtypes.int)
    self.assertEqual(n.src[1].arg, 0)
    self.assertEqual(n.src[1].dtype, dtypes.float.ptr(256))


    self.assertEqual(n.src[0].op, Ops.DEFINE_VAR)
    self.assertEqual(n.src[1].op, Ops.PARAM)

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
      """
      UOp Graph                              Assembly
      ─────────                              ────────
      SINK                                   s_endpgm
      └── STORE                              global_store_b32 v0, v1, s[4:5]
          ├── INDEX                          v_lshlrev_b32 v0, 2, v0
          │   ├── PARAM(arg=1)               s_load_b64 s[4:5], s[0:1], 8
          │   └── SPECIAL('lidx0')           (v0, implicit)
          └── ADD                            v_add_f32 v1, v1, 1.0
              ├── LOAD                       global_load_b32 v1, v0, s[2:3]
              │   └── INDEX                  (shared with store index)
              │       ├── PARAM(arg=0)       s_load_b64 s[2:3], s[0:1], 0
              │       └── SPECIAL('lidx0')   (v0, implicit)
              └── CONST(1.0)                 (inline constant, no instruction)

      Wait instructions inserted between:
        s_load_b64s and global_load  → s_waitcnt lgkmcnt(0)
        global_load and v_add_f32    → s_waitcnt vmcnt(0)
      """
      from tinygrad.codegen import full_rewrite_to_sink
      from tinygrad.uop.ops import KernelInfo

      buf_in = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 0)
      buf_out = UOp(Ops.PARAM, dtypes.float.ptr(256), (), 1)
      tidx = UOp.special(256, 'lidx0')
      load = buf_in.index(tidx).load()
      add = load + UOp.const(dtypes.float, 1.0)
      store = buf_out.index(tidx).store(add)
      sink = store.sink(arg=KernelInfo('add_one'))

      n = full_rewrite_to_sink(sink, self.renderer)
      self.assertEqual(n.op, Ops.SINK)

      store = n.src[0]
      self.assertEqual(store.op, Ops.STORE)
      self.assertEqual(store.dtype, dtypes.void)

      index1 = store.src[0]
      self.assertEqual(index1.op, Ops.INDEX)
      self.assertEqual(index1.dtype, dtypes.float.ptr(256))

      add = store.src[1]
      self.assertEqual(add.op, Ops.ADD)
      self.assertEqual(add.dtype, dtypes.float)

      param = index1.src[0]
      self.assertEqual(param.op, Ops.PARAM)
      self.assertEqual(param.arg, 1)
      self.assertEqual(param.dtype, dtypes.float.ptr(256))

      special = index1.src[1]
      self.assertEqual(special.op, Ops.SPECIAL)
      self.assertEqual(special.arg, 'lidx0')
      self.assertEqual(special.dtype, dtypes.int)

      load = add.src[0]
      self.assertEqual(load.op, Ops.LOAD)
      self.assertEqual(load.dtype, dtypes.float)

      constexpr = add.src[1]
      self.assertEqual(constexpr.op, Ops.CONST)
      self.assertAlmostEqual(float(constexpr.arg), 1.0)
      self.assertEqual(constexpr.dtype, dtypes.float)

      index2 = load.src[0]
      self.assertEqual(index2.op, Ops.INDEX)
      self.assertEqual(index2.dtype, dtypes.float.ptr(256))

      p2 = index2.src[0]
      self.assertEqual(p2.op, Ops.PARAM)
      self.assertEqual(p2.arg, 0)
      self.assertEqual(p2.dtype, dtypes.float.ptr(256))

      special2 = index2.src[1]
      self.assertEqual(special2.op, Ops.SPECIAL)
      self.assertEqual(special2.arg, 'lidx0')
      self.assertIs(special2, special)  # shared reference, same UOp object

if __name__ == "__main__":
  unittest.main()
