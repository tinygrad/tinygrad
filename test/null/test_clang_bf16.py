import unittest
from tinygrad import Tensor
from tinygrad.codegen import to_program
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop import Ops


class TestClangBf16(unittest.TestCase):
  def test_uses_ushort_not_native_bf16(self):
    self.assertEqual(ClangRenderer.type_map[dtypes.bfloat16], "unsigned short")

  def test_bf16_cpu_compile(self):
    t = Tensor([1.0, 2.0], dtype=dtypes.bfloat16) * 2
    linear, _ = t.linear_with_vars()
    for si in linear.src:
      ast = si.src[0]
      if ast.op is Ops.SINK:
        src = to_program(ast, ClangRenderer(Target("CPU", "CLANG", "x86_64,native"))).src[3].arg
        self.assertIn("unsigned short", src)
        return
    self.fail("expected bf16 kernel")