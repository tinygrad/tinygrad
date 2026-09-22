import unittest, io, struct
from contextlib import redirect_stdout
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.engine.realize import get_runtime
from tinygrad.helpers import Target
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.codegen import to_program

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestCPU(unittest.TestCase):
  def test_index_bitcast_reshape(self):
    for ren in (ClangRenderer, CPULLVMRenderer):
      for optimize in (False, True):
        for gated in (False, True):
          with self.subTest(renderer=ren, optimize=optimize, gated=gated):
            src, dst = [UOp.param(i, dtypes.uint8, 16).bitcast(dtypes.float32) for i in (1, 0)]
            r = UOp.range(4, 0)
            idx = r.valid(r < 3) if gated else r
            # Mask both a load and a store through the reinterpreted pointer.
            store = dst.index(idx).store(src.index(idx) + 1)
            ast = store.end(r).sink(arg=KernelInfo(opts_to_apply=None if optimize else ()))
            p = to_program(ast, ren(Device["CPU"].renderer.target))
            self.assertFalse(any(u.op is Ops.RESHAPE for u in p.src[1].src))
            bufs = [Buffer("CPU", 16, dtypes.uint8, initial_value=struct.pack("4f", *vals))
                    for vals in ((-1.,)*4, (1., 2., 3., 4.))]
            get_runtime("CPU", p)(*[bufs[i]._buf for i in p.arg.globals], wait=True)
            self.assertEqual(struct.unpack("4f", bufs[0].as_memoryview()), (2., 3., 4., -1. if gated else 5.))

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

if __name__ == '__main__':
  unittest.main()
