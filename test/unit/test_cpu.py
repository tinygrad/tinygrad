import unittest, io
from contextlib import redirect_stdout
from tinygrad import Tensor, Device, dtypes, function
from tinygrad.helpers import Target
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.codegen import to_program

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestCPU(unittest.TestCase):
  def test_sliced_input(self):
    @function(precompile=True)
    def gray(x): return x ^ (x >> 1)
    x = Tensor(list(range(20)), dtype=dtypes.uint8).realize()
    self.assertEqual(gray(x[4:]).tolist(), [i ^ (i >> 1) for i in range(4, 20)])

  def test_unsigned_indices(self):
    x = Tensor(list(range(256)), dtype=dtypes.int32)
    indices = Tensor([0, 127, 128, 255], dtype=dtypes.uint8)
    self.assertEqual(x[indices].tolist(), [0, 127, 128, 255])

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
