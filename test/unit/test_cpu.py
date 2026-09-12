import unittest, io
from contextlib import redirect_stdout
from tinygrad import Tensor, Device
from tinygrad.helpers import Target
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.codegen import to_program

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestCPU(unittest.TestCase):
  def test_arch_feats(self):
    ast = (Tensor.empty(16) + Tensor.empty(16)).schedule_linear().src[-1].src[0]
    for ren in Device[Device.DEFAULT].renderers:
      for arch, expect_vmov in [("x86_64,x86-64,avx", True), ("x86_64,x86-64,-avx", False)]:
        with self.subTest(arch=arch):
          if ren is X86Renderer: continue # X86 requires avx support
          if ren is LVPRenderer: continue # LVP does not play nice with cross compilation
          r = ren(Target(device="CPU", arch=arch))
          p = to_program(ast, r)
          lib = r.compiler.compile(p.src[2].arg)
          out = io.StringIO()
          with redirect_stdout(out): r.compiler.disassemble(lib)
          self.assertEqual("vmov" in out.getvalue(), expect_vmov, out.getvalue())

  def test_x86_abi_interleaved_ordering(self):
    from tinygrad.renderer.isa import IselContext
    from tinygrad.dtype import AddrSpace, dtypes
    from tinygrad.uop.ops import UOp

    s0 = UOp.variable("s0", 1, 10)
    p0 = s0.param_like(0)  # ALU scalar in slot 0
    p1 = UOp.param(1, dtypes.float, (4,), AddrSpace.GLOBAL)  # Buffer in slot 1
    s2 = UOp.variable("s2", 1, 10)
    p2 = s2.param_like(2)  # ALU scalar in slot 2
    p3 = UOp.param(3, dtypes.float, (4,), AddrSpace.GLOBAL)  # Buffer in slot 3

    sink = UOp.sink(p1, p3, p0, p2)
    ctx = IselContext(sink)
    # CPUProgram passes [*bufs, *vals]. Buffers (non-ALU) must precede scalars (ALU)
    self.assertEqual(ctx.func_args, [p1, p3, p0, p2])

if __name__ == '__main__':
  unittest.main()
