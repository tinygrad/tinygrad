import unittest, io
from contextlib import redirect_stdout
from tinygrad import Tensor, Device
from tinygrad.helpers import Target
from tinygrad.engine.realize import get_program

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestCPU(unittest.TestCase):
  def test_arch_feats(self):
    ast = (Tensor.empty(16) + Tensor.empty(16)).schedule()[-1].ast
    for arch, expect_vmov in [("x86_64,x86-64,avx", True), ("x86_64,x86-64,-avx", False)]:
      with self.subTest(arch=arch):
        r = type(Device[Device.DEFAULT].renderer)(Target(device="CPU", arch=arch))
        p = get_program(ast, r)
        lib = r.compiler.compile(p.src)
        out = io.StringIO()
        with redirect_stdout(out): r.compiler.disassemble(lib)
        self.assertEqual("vmov" in out.getvalue(), expect_vmov, out.getvalue())

if __name__ == '__main__':
  unittest.main()
