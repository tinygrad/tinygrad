import unittest

from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels import amd_glm
from tinygrad.uop.ops import UOp


class TestGLMFP8RDNA3(unittest.TestCase):
  def test_fp8_fragment_widths(self):
    k = Tensor.zeros(64, 33, dtype=dtypes.uint32)  # 32 e4m3 words + one scale
    for rdna4, width in ((False, 16), (True, 8)):
      with self.subTest(rdna4=rdna4):
        frag = amd_glm._fp8_frag(k.uop, UOp.const(0), 0, UOp.const(0), rdna4)
        self.assertEqual(len(frag.src), width)


if __name__ == "__main__":
  unittest.main()
