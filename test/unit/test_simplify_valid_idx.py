import unittest
from typing import Tuple

from tinygrad.codegen.linearize import linearize_uop
from tinygrad.codegen.uopgraph import full_graph_rewrite, is_increasing
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.ops import UOp, UOps, BinaryOps

def get_gated_load_uop(valid:UOp, idx:UOp):
  return UOp(UOps.LOAD, dtypes.float, (
    UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=0),
    idx,
    UOp.const(dtypes.float, 0.0),
    valid
  ))

def get_load_image_uop(image_shape:Tuple[int, ...], valid:UOp, idx:Tuple[UOp, UOp]):
  return UOp(UOps.LOAD, dtypes.float.vec(4), (
    UOp(UOps.DEFINE_GLOBAL, dtypes.imagef(image_shape), arg=0),
    UOp(UOps.VECTORIZE, dtypes.int.vec(2), idx),
    UOp(UOps.VECTORIZE, dtypes.float.vec(4), src=(UOp.const(dtypes.float, 0),)*4),
    valid
  ))

def render(uop:UOp) -> str:
  uops = linearize_uop(full_graph_rewrite(uop.sink()))
  from tinygrad.renderer.cstyle import OpenCLRenderer
  class TestRenderer(OpenCLRenderer):
    code_for_op = {**OpenCLRenderer.code_for_op, BinaryOps.IDIV: lambda a,b,dtype: f"({a}//{b})"}
  fxn = TestRenderer().render("", uops)
  # print(fxn)
  return fxn.split("val0 = ")[1].split(";")[0]

def Special(expr, nmax): return UOp(UOps.SPECIAL, dtypes.int, (), (expr, nmax))
def Variable(expr, nmin, nmax): return UOp.variable(expr, nmin, nmax)
def Range(n, nmax):
  return UOp(UOps.RANGE, dtypes.int, arg=(n, True), src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, nmax),))

class TestHelpers(unittest.TestCase):
  def test_is_increasing(self):
    idx1 = Special("idx1", 32)
    idx2 = Special("idx2", 64)
    ridx0 = Variable("ridx0", 0, 5)
    ridx1 = Variable("ridx1", 0, 2)
    ridx2 = Variable("ridx2", 0, 2)
    # (ridx0+(idx1*48)+(ridx2*6)+(-6)),((idx2*2)+ridx1+(-1)))
    f0 = ((idx1*24)+(ridx2*3)+ridx0+765)%768
    f1 = ridx0+(idx1*48)+(ridx2*6)+(-6)
    f2 = (idx2*2)+ridx1+((idx1+((ridx2+7)//8)+31)//32)+(-2)
    f3 = (idx2*2)+ridx1+(-1)

    self.assertFalse(is_increasing(f0))
    self.assertTrue(is_increasing(f1))
    self.assertTrue(is_increasing(f2))
    self.assertTrue(is_increasing(f3))

    rng = UOp(UOps.RANGE, dtypes.int, arg=(2, True), src=(UOp(UOps.CONST, dtypes.int, arg=0, src=()), UOp(UOps.CONST, dtypes.int, arg=5, src=()),))
    self.assertTrue(is_increasing(rng))
    self.assertTrue(is_increasing(rng+2))

class TestValidIdxSimplification(unittest.TestCase):
  def test_conv_backward(self):
    # DEBUG=4 python3 test/test_ops.py TestOps.test_simple_conv2d
    gidx0 = Special("gidx0", 3)
    gidx1 = Special("gidx1", 3)
    lidx0 = Special("lidx0", 4)
    lidx1 = Special("lidx1", 3)
    lidx2 = Special("lidx2", 3)
    ridx0 = Range(0, 4)
    alu0 = gidx0*3
    alu1 = (alu0+lidx2)
    alu2 = (gidx1*3)
    alu3 = (alu1+7)
    alu4 = (alu1+8)
    alu5 = (alu1+9)
    alu6 = ((gidx0+9)//10)
    alu7 = (alu3%10)
    alu8 = (alu4%10)
    alu9 = (alu5%10)
    alu10 = (gidx1+(ridx0*3))
    alu11 = (ridx0*9)
    alu12 = (alu2+lidx1+alu11)
    alu13 = ((alu6+alu2+lidx1+alu11)%10)
    alu14 = (alu12%10)
    alu15 = (((((alu10//10)+lidx0)%4)*441)+(((alu12//10)%3)*3)+(alu14*63))
    alu16 = alu12.lt(30)
    alu17 = alu16&(alu14.lt(7))

    # TODO: simplify these
    val0 = get_gated_load_uop(alu17&(alu9.lt(7)), alu15+(alu5//10)+(alu9*9))
    self.assertEqual(render(val0),
      "((((alu2<30)&(alu3<7))&(alu1<7))?data0[(((((gidx1+(ridx0*3))//10)+lidx0)%4)*441)+((alu2//10)*3)+(alu3*63)+(alu0//10)+(alu1*9)]:0.0f)")

    val1 = get_gated_load_uop(
      ((alu16&gidx0.lt(1))&alu13.lt(7))&alu7.lt(7),
      ((((((((((lidx1*10)+gidx0)//3)+3)//10)+alu10)//10)+lidx0)%4)*441)+((((alu6+alu12)//10)%3)*3)+(alu13*63)+(((alu3//10)+2)%3)+(alu7*9)
    )
    self.assertEqual(render(val1),
      "(((((alu2<30)&(gidx0<1))&(((((gidx0+9)//10)+alu0+lidx1+alu1)%10)<7))&((((gidx0*3)+lidx2+7)%10)<7))?data0[(lidx2*9)+(((((gidx1+(ridx0*3))//10)+lidx0)%4)*441)+((alu2//10)*3)+((alu2%10)*63)+65]:0.0f)")  # noqa: E501

    val2 = get_gated_load_uop(alu17&alu1.lt(7), alu15+(gidx0*27)+(lidx2*9))
    self.assertEqual(render(val2),
      "((((alu0<30)&(alu1<7))&(((gidx0*3)+lidx2)<7))?data0[(((((gidx1+(ridx0*3))//10)+lidx0)%4)*441)+((alu0//10)*3)+(alu1*63)+(gidx0*27)+(lidx2*9)]:0.0f)")  # noqa: E501

    val3 = get_gated_load_uop(alu17&alu8.lt(7), (alu4//10)+alu15+(alu8*9)+1)
    self.assertEqual(render(val3),
      "((((alu2<30)&(alu3<7))&(alu1<7))?data0[(alu0//10)+(((((gidx1+(ridx0*3))//10)+lidx0)%4)*441)+((alu2//10)*3)+(alu3*63)+(alu1*9)+1]:0.0f)")

  def test_cumsum(self):
    gidx0 = Special("gidx0", 5)
    lidx0 = Special("lidx0", 4)
    gate = (gidx0*4+lidx0).lt(19).ne(True)
    idx = gidx0*4+lidx0-19
    load = get_gated_load_uop(gate, idx)
    self.assertEqual(render(load), "(((((gidx0*4)+lidx0)<19)!=1)?data0[0]:0.0f)")

  def test_simplify_within_valid(self):
    ridx0 = Range(0, 4)
    ridx1 = Range(1, 4)
    ridx2 = Range(2, 4)
    ridx3 = Range(3, 4)
    valid = (ridx0*3+ridx1).lt(8) & (((ridx0*3+ridx1)//8+ridx2*3+ridx3)%4).lt(2)
    idx = ridx0+ridx1+ridx2+ridx3
    load = get_gated_load_uop(valid, idx)
    # TODO: simplify the valid
    # alu0 = ((ridx0*3)+ridx1)
    self.assertEqual(render(load), "(((alu0<8)&((((alu0//8)+(ridx2*3)+ridx3)%4)<2))?data0[ridx0+ridx1+ridx2+ridx3]:0.0f)")

class TestImageSimplification(unittest.TestCase):
  def test_idx_gt_c(self):
    # (idx1 < c+1).ne(True) ? (..., idx1-1+c) : 0 can drop the valid
    # (idx1 < c+1).ne(True) -> idx > c
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (10, 10, 4)
    load = get_load_image_uop(shape, (gidx1).lt(1).ne(True), (gidx0, gidx1-1))
    self.assertEqual(render(load), "read_imagef(data0, smp, (int2)(gidx0,(gidx1+-1)))")
    load = get_load_image_uop(shape, (gidx1).lt(1).ne(True), (gidx0, gidx1-2))
    self.assertEqual(render(load), "read_imagef(data0, smp, (int2)(gidx0,(gidx1+-2)))")

    # should match any one of the AND clause and drop the matched statement from valid
    valid = (gidx0).lt(1).ne(True) & (gidx1).lt(1).ne(True)
    load = get_load_image_uop(shape, valid, (gidx0+1, gidx1-1))
    self.assertEqual(render(load),
                     "(((gidx0<1)!=1)?read_imagef(data0, smp, (int2)((gidx0+1),(gidx1+-1))):(float4)(0.0f,0.0f,0.0f,0.0f))")

    valid = (gidx1).lt(1).ne(True) & (gidx1).lt(1).ne(True)
    load = get_load_image_uop(shape, valid, (gidx0, gidx1-1))
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+-1)))")

  def test_idx_lt_bound(self):
    # (idx1 < image_bound) ? (..., idx1) : 0 can drop the valid
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    load = get_load_image_uop((10, 10, 4), (gidx1).lt(10), (gidx0, gidx1))
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(gidx0,gidx1))")
    # same thing, valid has a div
    load = get_load_image_uop((10, 10, 4), (gidx1//2).lt(5), (gidx0, gidx1))
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(gidx0,gidx1))")
    # 10x20 image, not out of bound
    load = get_load_image_uop((20, 10, 4), (gidx1).lt(10), (gidx0, gidx1))
    self.assertEqual(render(load),
                     "((gidx1<10)?read_imagef(data0, smp, (int2)(gidx0,gidx1)):(float4)(0.0f,0.0f,0.0f,0.0f))")

  def test_generic_idx_lt_bound(self):
    # (idx1 < image_bound - c) ? (..., idx1 + c) : 0 can drop the valid
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (10, 10, 4)
    load = get_load_image_uop(shape, (gidx1).lt(8), (gidx0, gidx1+2))
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+2)))")
    load = get_load_image_uop(shape, (gidx1).lt(5), (gidx0, gidx1+5))
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(gidx0,(gidx1+5)))")

  def test_valid_empty_set(self):
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (32, 32, 4)
    idx = (gidx0%2, gidx1+2)
    # not empty
    load = get_load_image_uop(shape, (gidx0).lt(8), idx)
    self.assertEqual(render(load),
                     "((gidx0<8)?read_imagef(data0, smp, (int2)((gidx0%2),(gidx1+2))):(float4)(0.0f,0.0f,0.0f,0.0f))")

    # empty
    load = get_load_image_uop(shape, (gidx0).lt(8) & (gidx0).lt(8).ne(True), idx)
    self.assertRaises(IndexError, lambda: render(load))

  def test_openpilot_conv1(self):
    # first conv in openpilot
    # kernel in tinygrad ae5d1407ee844a97a52ad3756835d38e7e2b9e1b https://gist.github.com/chenyuxyz/39c2d4e9a076b46731c67d345ff066b6
    idx1 = Special("idx1", 32)
    idx2 = Special("idx2", 64)
    # ridx0 = Variable("ridx0", 0, 5)
    # ridx1 = Variable("ridx1", 0, 2)
    # ridx2 = Variable("ridx2", 0, 2)
    ridx0 = Range(0, 6)
    ridx1 = Range(1, 3)
    ridx2 = Range(2, 3)

    alu1 = ((idx2*2)+ridx1)
    alu4 = ((idx1*48)+(ridx2*6)+ridx0)

    valid = (((idx2*2)+(ridx1)).lt(1).ne(True))&(((idx1*8)+(ridx2)).lt(1).ne(True))
    shape = (128, 1536, 4)
    idx = ((alu4+1530)%1536, alu1+((idx1+((ridx2+7)//8)+31)//32)+(-2))

    load = get_load_image_uop(shape, valid, idx)
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(((idx1*48)+(ridx2*6)+ridx0+-6),((idx2*2)+ridx1+-1)))")

  def test_openpilot_conv2(self):
    # conv in test/external/external_test_valid_remove.py
    idx1 = Special("idx1", 32)
    idx2 = Special("idx2", 64)
    # ridx0 = Variable("ridx0", 0, 2)
    # ridx1 = Variable("ridx1", 0, 2)
    # ridx2 = Variable("ridx2", 0, 2)
    ridx0 = Range(0, 3)
    ridx1 = Range(1, 3)
    ridx2 = Range(2, 3)

    alu1 = ((idx2*2)+ridx1)
    alu3 = ((idx1*24)+(ridx2*3)+ridx0)

    valid = (((idx2*2)+ridx1).lt(1).ne(True))&(((idx1*8)+ridx2).lt(1).ne(True))
    shape = (128, 768, 4)
    idx = ((alu3+765)%768, alu1+((idx1+((ridx2+7)//8)+31)//32)+(-2))
    load = get_load_image_uop(shape, valid, idx)

    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(((idx1*24)+(ridx2*3)+ridx0+-3),((idx2*2)+ridx1+-1)))")

  def test_openpilot_conv3(self):
    # in openpilot 0.9.7
    idx0 = Special("idx0", 64)
    idx1 = Special("idx1", 2)
    idx2 = Special("idx2", 4)
    ridx0 = Range(0, 7)
    ridx1 = Range(1, 7)

    alu2 = ((idx2*2)+ridx0)
    alu4 = ((idx1*8)+ridx1)
    alu6 = ((idx1*512)+(ridx1*64)+idx0)

    valid = alu2.lt(11)&(alu4.lt(3).ne(True))
    shape = (8, 1024, 4)
    idx = (((alu6+832)%1024),(alu2+((idx1+((ridx1+5)//8)+1)//2)+(-4)))

    load = get_load_image_uop(shape, valid, idx)
    # TODO: simplify idx
    # alu0 = ((idx2*2)+ridx0)
    self.assertEqual(render(load),
      "(((alu0<11)&((((idx1*8)+ridx1)<3)!=1))?read_imagef(data0, smp, (int2)((((idx1*512)+(ridx1*64)+idx0+832)%1024),(alu0+((idx1+((ridx1+5)//8)+1)//2)+-4))):(float4)(0.0f,0.0f,0.0f,0.0f))")  # noqa: E501

  def test_simplify1(self):
    # idx has the form (A % m, A // m + k) and valid has (c0 < A) and (A < c1)
    gidx = Special("gidx", 512)
    valid = gidx.lt(488) & (gidx).lt(480).ne(True)
    idx = ((gidx*3+18)%26, (gidx*3+18)//26-56)
    load = get_load_image_uop((1, 26, 4), valid, idx)
    # alu0 is ((gidx*3)+18)
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)(((gidx*3)+-1438),0))")

  def test_simplify2(self):
    # from GPU=1 DEBUG=4 FORWARD_ONLY=1 IMAGE=2 python3 test/test_ops.py TestOps.test_simple_padding_conv2d
    lidx = Special("lidx", 4)
    valid = lidx.lt(3) & lidx.lt(1).ne(True)
    idx = ((lidx+1)%2, (lidx+1)//2-1)
    load = get_load_image_uop((1, 2, 4), valid, idx)
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)((lidx+-1),0))")

  def test_simplify3(self):
    # from openpilot
    idx0 = Special("idx0", 265)
    valid = idx0.lt(201).ne(True)
    idx = ((idx0+55)%64, (idx0+55)//64-4)
    load = get_load_image_uop((1, 64, 4), valid, idx)
    self.assertEqual(render(load),
                     "read_imagef(data0, smp, (int2)((idx0+-201),0))")

  def test_simplify4(self):
    idx0 = Special("idx0", 512)
    shape = (4, 64, 4)
    alu2 = ((idx0*4+1)%32)
    alu3 = ((idx0*4+2)%32)
    alu4 = ((idx0*4+3)%32)
    alu5 = (idx0*4%32)
    alu8 = (idx0//8%32//4)
    alu9 = idx0.lt(256)

    # TODO: can this be simplified further?
    load = get_load_image_uop(shape, alu9, (((alu8+(alu2*8))%64),(alu2//8)))
    # alu0 = (((idx0*4)+1)%32)
    self.assertEqual(render(load),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((idx0//32)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")

    load = get_load_image_uop(shape, alu9, (((alu8+(alu3*8))%64),(alu3//8)))
    # alu0 = (((idx0*4)+2)%32)
    self.assertEqual(render(load),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((idx0//32)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")

    load = get_load_image_uop(shape, alu9, (((alu8+(alu4*8))%64),(alu4//8)))
    # alu0 = (((idx0*4)+3)%32)
    self.assertEqual(render(load),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((idx0//32)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")

    load = get_load_image_uop(shape, alu9, (((alu8+(alu5*8))%64),(alu5//8)))
    # alu0 = ((idx0*4)%32)
    self.assertEqual(render(load),
                     "((idx0<256)?read_imagef(data0, smp, (int2)((((idx0//32)+(alu0*8))%64),(alu0//8))):(float4)(0.0f,0.0f,0.0f,0.0f))")

  def test_simplify5(self):
    # openpilot 0.9.7, chunk replacement to simplify
    shape = (10, 384, 4)
    idx0 = Special("idx0", 16)
    idx1 = Special("idx1", 24)
    alu0 = idx0*4
    alu1 = (idx1*256)+alu0
    alu2 = idx1//3
    alu3 = ((alu1+1)%768)
    idx = ((idx0+((((alu3//640)+alu2)%8)*16)+128),((alu3//64)%10))
    valid = alu3.lt(640)

    load = get_load_image_uop(shape, valid, idx)
    self.assertEqual(render(load),
                     "((alu0<640)?read_imagef(data0, smp, (int2)((idx0+((idx1//3)*16)+128),(alu0//64))):(float4)(0.0f,0.0f,0.0f,0.0f))")

if __name__ == '__main__':
  unittest.main()
