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
  def test_idx_neg_lt_c(self):
    # (idx1 * (-1) < c) ? (..., idx1-1+c) : 0 can drop the valid
    gidx0 = Variable("gidx0", 32)
    gidx1 = Variable("gidx1", 32)
    self.assertEqual(render((10, 10, 4), (gidx1*(-1)).lt(0), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1-1))),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+(-1))))")
    self.assertEqual(render((10, 10, 4), (gidx1*(-1)).lt(-1), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1-2))),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+(-2))))")
    self.assertEqual(render((10, 10, 4), (gidx1*(-1)).lt(1), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1))),
                     "read_imagef(data0, smp, (int2)(gidx0,gidx1))")

    # should match any one of the AND clause and drop the matched statement from valid
    valid = (gidx1*(-1)).lt(0) and (gidx0*(-1)).lt(0)
    self.assertEqual(render((10, 10, 4), valid, UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1-1))),
                     "(((gidx0*(-1))<0)?read_imagef(data0, smp, (int2)(gidx0,(gidx1+(-1)))):(float4)(0.0f,0.0f,0.0f,0.0f))")

    valid = (gidx1*(-1)).lt(0) and (gidx1*(-1)).lt(0)
    self.assertEqual(render((10, 10, 4), valid, UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1-1))),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+(-1))))")

  def test_idx_lt_bound(self):
    # (idx1 < image_bound) ? (..., idx1) : 0 can drop the valid
    gidx0 = Variable("gidx0", 32)
    gidx1 = Variable("gidx1", 32)
    self.assertEqual(render((10, 10, 4), (gidx1).lt(10), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1))),
                     "read_imagef(data0, smp, (int2)(gidx0,gidx1))")
    # 10x20 image, not out of bound
    self.assertEqual(render((20, 10, 4), (gidx1).lt(10), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1))),
                     "((gidx1<10)?read_imagef(data0, smp, (int2)(gidx0,gidx1)):(float4)(0.0f,0.0f,0.0f,0.0f))")

  def test_generic_idx_lt_bound(self):
    # (idx1 < image_bound - c) ? (..., idx1 + c) : 0 can drop the valid
    gidx0 = Variable("gidx0", 32)
    gidx1 = Variable("gidx1", 32)
    self.assertEqual(render((10, 10, 4), (gidx1).lt(8), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1+2))),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+2)))")
    self.assertEqual(render((10, 10, 4), (gidx1).lt(5), UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0, gidx1+5))),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+5)))")

if __name__ == '__main__':
  unittest.main()
