import unittest

from tinygrad.codegen.uopgraph import linearize_uop, full_graph_rewrite
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, UOps, BinaryOps

def render(image_shape, valid:UOp, idx:UOp) -> str:
  uops = linearize_uop(full_graph_rewrite(UOp(UOps.LOAD, dtypes.float.vec(4), (
    UOp(UOps.DEFINE_GLOBAL, dtypes.imagef(image_shape), arg=0),
    idx,
    UOp(UOps.VECTORIZE, dtypes.float.vec(4), src=(UOp.const(dtypes.float, 0),)*4),
    valid
  )).sink()))
  from tinygrad.renderer.cstyle import OpenCLRenderer
  class TestRenderer(OpenCLRenderer):
    code_for_op = {**OpenCLRenderer().code_for_op, BinaryOps.IDIV: lambda a,b,dtype: f"({a}//{b})"}
  fxn = TestRenderer().render("", uops)
  return fxn.split("float4 val0 = ")[1].split(";")[0]

def Variable(expr, nmax):
  return UOp(UOps.SPECIAL, dtypes.int, (), (expr, nmax))

class TestValidSimplification(unittest.TestCase):
  def test_idx_lt_0(self):
    # (idx1 * (-1) < 0) ? (..., idx1-1) : 0 can drop the valid
    gidx0 = Variable("gidx0", 32)
    gidx1 = Variable("gidx1", 32)
    self.assertEqual(
      render((10, 10, 4), (gidx1*(-1)).lt(0), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1-1))),
      "read_imagef(data0, smp, (int2)(gidx0,(gidx1+(-1))))"
    )

if __name__ == '__main__':
  unittest.main()
