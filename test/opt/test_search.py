import unittest

from tinygrad import UOp, dtypes
from tinygrad.codegen.opt.postrange import Scheduler
from tinygrad.codegen.opt.search import _try_compile
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.uop.ops import AxisType, KernelInfo


class TestSearch(unittest.TestCase):
  def test_compile_symbolic_kernel_candidate(self):
    out = UOp.param(0, dtypes.int, (4,))
    size = UOp.variable("size", 1, 4, dtype=dtypes.int)
    rng = UOp.range(4, 0, AxisType.LOOP, dtype=dtypes.int)
    ast = out.index(rng).store((rng < size).cast(dtypes.int)).end(rng).sink(arg=KernelInfo())
    _, compiled = _try_compile((0, Scheduler(ast, ClangRenderer(Target("CPU", arch="x86_64,x86-64")))))
    self.assertIsNotNone(compiled)


if __name__ == "__main__":
  unittest.main()
