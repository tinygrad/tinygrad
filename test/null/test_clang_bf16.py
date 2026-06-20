import unittest
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.renderer.cstyle import ClangRenderer


class TestClangBf16(unittest.TestCase):
  def test_uses_ushort_not_native_bf16(self):
    self.assertEqual(ClangRenderer.type_map[dtypes.bfloat16], "unsigned short")

  def test_bf16_cpu_compile(self):
    out = (Tensor([1.0, 2.0], dtype=dtypes.bfloat16) * 2).tolist()
    self.assertEqual(out, [2.0, 4.0])