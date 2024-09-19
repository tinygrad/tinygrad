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
  # print(fxn)
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

  def test_valid_empty_set(self):
    gidx0 = Variable("gidx0", 32)
    gidx1 = Variable("gidx1", 32)
    shape = (1, 2, 4)
    idx = UOp(UOps.VECTORIZE, dtypes.int.vec(2), (gidx0%2, gidx1+2))
    # not empty
    self.assertEqual(render(shape, (gidx0).lt(8) & (-gidx0).lt(-6), idx),
                     "(((gidx0<8)&((gidx0*(-1))<(-6)))?read_imagef(data0, smp, (int2)((gidx0%2),(gidx1+2))):(float4)(0.0f,0.0f,0.0f,0.0f))")

    # empty
    self.assertRaises(IndexError, lambda: render(shape, (gidx0).lt(8) & (-gidx0).lt(-7), idx))

  def test_simplify1(self):
    # idx has the form (A % m, A // m + k) and valid has (c0 < A) and (A < c1)
    gidx = Variable("gidx", 512)
    valid = gidx.lt(488) & (-gidx).lt(-479)
    idx = ((gidx*3+18)%26, (gidx*3+18)//26-56)
    # alu0 is ((gidx*3)+18)
    self.assertEqual(render((1, 26, 4), valid, UOp(UOps.VECTORIZE, dtypes.int.vec(2), idx)),
                     "read_imagef(data0, smp, (int2)(((gidx*3)+(-1438)),0))")

  def test_simplify2(self):
    # from GPU=1 DEBUG=4 FORWARD_ONLY=1 IMAGE=2 python3 test/test_ops.py TestOps.test_simple_padding_conv2d
    lidx = Variable("lidx", 4)
    valid = lidx.lt(3) & (-lidx).lt(0)
    idx = ((lidx+1)%2, (lidx+1)//2-1)
    self.assertEqual(render((1, 2, 4), valid, UOp(UOps.VECTORIZE, dtypes.int.vec(2), idx)),
                     "read_imagef(data0, smp, (int2)((lidx+(-1)),0))")

  def test_simplify3(self):
    # from openpilot
    idx0 = Variable("idx0", 265)
    valid = (-idx0).lt(-200)
    idx = ((idx0+55)%64, (idx0+55)//64-4)
    self.assertEqual(render((1, 64, 4), valid, UOp(UOps.VECTORIZE, dtypes.int.vec(2), idx)),
                     "read_imagef(data0, smp, (int2)((idx0+(-201)),0))")

  def test_simplify4(self):
    idx0 = Variable("idx0", 512)
    data1_shape = (4, 64, 4)
    alu2 = ((idx0*4+1)%32)
    alu3 = ((idx0*4+2)%32)
    alu4 = ((idx0*4+3)%32)
    alu5 = (idx0*4%32)
    alu8 = (idx0//8%32//4)
    alu9 = idx0.lt(256)

    # TODO: simplify these, manual parsing is not going to work
    # alu0 = (((idx0*4)+1)%32)
    self.assertEqual(render(data1_shape, alu9, UOp(UOps.VECTORIZE, dtypes.int.vec(2), (((alu8+(alu2*8))%64),(alu2//8)))),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((((idx0//8)%32)//4)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")
    # alu0 = (((idx0*4)+2)%32)
    self.assertEqual(render(data1_shape, alu9, UOp(UOps.VECTORIZE, dtypes.int.vec(2), (((alu8+(alu3*8))%64),(alu3//8)))),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((((idx0//8)%32)//4)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")
    # alu0 = (((idx0*4)+3)%32)
    self.assertEqual(render(data1_shape, alu9, UOp(UOps.VECTORIZE, dtypes.int.vec(2), (((alu8+(alu4*8))%64),(alu4//8)))),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((((idx0//8)%32)//4)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")
    # alu0 = ((idx0*4)%32)
    self.assertEqual(render(data1_shape, alu9, UOp(UOps.VECTORIZE, dtypes.int.vec(2), (((alu8+(alu5*8))%64),(alu5//8)))),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((((idx0//8)%32)//4)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")

if __name__ == '__main__':
  unittest.main()
