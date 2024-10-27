import unittest, math
import numpy as np
from tinygrad import dtypes
from tinygrad.ops import UOp, UOps
from tinygrad.codegen.uopgraph import full_graph_rewrite
from tinygrad.codegen.transcendental import payne_hanek_reduction, cody_waite_reduction
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.runtime.ops_python import PythonProgram, PythonRenderer, PythonCompiler, PythonAllocator

class TestReduction(unittest.TestCase):
  def _run_uop(self, uop:UOp):
    g = UOp(UOps.DEFINE_GLOBAL, uop.dtype.ptr(), arg=0, src=())
    rw = full_graph_rewrite(UOp.store(g, UOp.const(dtypes.int, 0), uop).sink())
    prog = PythonProgram("run", PythonCompiler().compile(PythonRenderer().render("run", linearize_uop(rw))))
    buf = PythonAllocator().alloc(uop.dtype.itemsize)
    prog(buf)
    return buf.cast(uop.dtype.fmt).tolist()[0]

  def test_payne_hanek_reduction(self):
    r, q = (self._run_uop(u) for u in payne_hanek_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1)))
    # TODO: should r be in [0, pi/2) per doc?
    np.testing.assert_allclose(r, 0.1 - math.pi / 2)
    np.testing.assert_equal(q, 1)

  def test_cody_waite_reduction(self):
    r, q = (self._run_uop(u) for u in cody_waite_reduction(UOp.const(dtypes.float64, 12 * math.pi + 0.1)))
    np.testing.assert_allclose(r, 0.1)
    # TODO: should q be in [0, 1, 2, 3]?
    np.testing.assert_equal(q, 12)

if __name__ == '__main__':
  unittest.main()
