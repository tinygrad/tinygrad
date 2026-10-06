import unittest
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Compiler
from tinygrad.helpers import WIN, Target
from tinygrad.renderer.cstyle import CUDARenderer


class TestCUDAFailures(unittest.TestCase):
  @unittest.skipUnless(WIN, "long long for u64/i64 (LLP64) is a windows thing")
  def test_windows_longlong(self):
    class CUDARendererMock(CUDARenderer):
      def __init__(self, target:Target, use_nvcc=False):
        super(CUDARenderer, self).__init__(target)
        self.compiler = Compiler(cachekey=None)
        self.tensor_cores = []

    t = Tensor([2], dtype=dtypes.long) + 1
    s = t.schedule_linear().src[-1]
    p = to_program(s.src[0], CUDARendererMock(Target.parse("NULL::sm_75")))
    self.assertIn("long long", p.src[-2].arg)

if __name__ == "__main__": unittest.main()
