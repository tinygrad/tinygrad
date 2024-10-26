import unittest
from typing import Tuple

from tinygrad.codegen.uopgraph import full_graph_rewrite, is_increasing
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, UOps, simplify_valid

def get_gated_load_uop(valid:UOp, idx:UOp):
  return UOp(UOps.LOAD, dtypes.float, (
    UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0),
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

def Special(expr, nmax): return UOp(UOps.SPECIAL, dtypes.int, (), (expr, nmax))
def Variable(expr, nmin, nmax): return UOp.variable(expr, nmin, nmax)
def Range(n, nmax): return UOp(UOps.RANGE, dtypes.int, arg=(n, True), src=(UOp.const(dtypes.int, 0), UOp.const(dtypes.int, nmax),))

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
  def check(self, load, sidx, svalid):
    load = full_graph_rewrite(load.sink()).src[0]
    idx, valid = load.src[1], load.src[3]
    self.assertEqual(idx.render(simplify=False), sidx)
    self.assertEqual(valid.render(simplify=False), svalid)

  def test_cumsum(self):
    gidx0 = Special("gidx0", 5)
    lidx0 = Special("lidx0", 4)
    gate = (gidx0*4+lidx0).lt(19).ne(True)
    idx = gidx0*4+lidx0-19
    load = get_gated_load_uop(gate, idx)
    self.check(load,
      "0",
      "(((lidx0+(gidx0*4))<19)!=True)")

  def test_simplify_within_valid1(self):
    ridx0 = Range(0, 4)
    ridx1 = Range(1, 4)
    ridx2 = Range(2, 4)
    ridx3 = Range(3, 4)
    valid = (ridx0*3+ridx1).lt(8) & (((ridx0*3+ridx1)//8+ridx2*3+ridx3)%4).lt(2)
    idx = ridx0+ridx1+ridx2+ridx3
    load = get_gated_load_uop(valid, idx)
    self.check(load,
      "(((ridx0+ridx1)+ridx2)+ridx3)",
      "((((ridx0*3)+ridx1)<8)&((((ridx2*3)+ridx3)%4)<2))")

  def test_simplify_within_valid2(self):
    gidx0 = Special("gidx0", 56)
    ridx0 = Range(0, 3)
    alu0 = gidx0+ridx0
    valid = alu0.lt(57) & alu0.ge(1)
    self.assertIsNone(simplify_valid(valid))

class TestImageSimplification(unittest.TestCase):
  def check(self, load, svalid, sidx0, sidx1):
    load = full_graph_rewrite(load.sink()).src[0]
    idx = load.src[1]
    self.assertEqual(idx.op, UOps.VECTORIZE)
    self.assertEqual(len(idx.src), 2)
    idx0, idx1 = idx.src[0], idx.src[1]
    self.assertEqual(idx0.render(simplify=False), sidx0)
    self.assertEqual(idx1.render(simplify=False), sidx1)
    if svalid is not None: self.assertEqual(load.src[3].render(simplify=False), svalid)

  def test_idx_gt_c(self):
    # (idx1 < c+1).ne(True) ? (..., idx1-1+c) : 0 can drop the valid
    # (idx1 < c+1).ne(True) -> idx > c
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (10, 10, 4)
    load = get_load_image_uop(shape, (gidx1).lt(1).ne(True), (gidx0, gidx1-1))
    self.check(load, None, "gidx0", "(gidx1+-1)")
    load = get_load_image_uop(shape, (gidx1).lt(1).ne(True), (gidx0, gidx1-2))
    self.check(load, None, "gidx0", "(gidx1+-2)")

    # should match any one of the AND clause and drop the matched statement from valid
    valid = (gidx0).lt(1).ne(True) & (gidx1).lt(1).ne(True)
    load = get_load_image_uop(shape, valid, (gidx0+1, gidx1-1))
    self.check(load, "((gidx0<1)!=True)", "(gidx0+1)", "(gidx1+-1)")

    valid = (gidx1).lt(1).ne(True) & (gidx1).lt(1).ne(True)
    load = get_load_image_uop(shape, valid, (gidx0, gidx1-1))
    self.check(load, None, "gidx0", "(gidx1+-1)")

  def test_idx_lt_bound(self):
    # (idx1 < image_bound) ? (..., idx1) : 0 can drop the valid
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    load = get_load_image_uop((10, 10, 4), (gidx1).lt(10), (gidx0, gidx1))
    self.check(load, None, "gidx0", "gidx1")

    # same thing, valid has a div
    load = get_load_image_uop((10, 10, 4), (gidx1//2).lt(5), (gidx0, gidx1))
    self.check(load, None, "gidx0", "gidx1")

    # 10x20 image, not out of bound
    load = get_load_image_uop((20, 10, 4), (gidx1).lt(10), (gidx0, gidx1))
    self.check(load, "(gidx1<10)", "gidx0", "gidx1")

  def test_generic_idx_lt_bound(self):
    # (idx1 < image_bound - c) ? (..., idx1 + c) : 0 can drop the valid
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (10, 10, 4)
    load = get_load_image_uop(shape, (gidx1).lt(8), (gidx0, gidx1+2))
    self.check(load, None, "gidx0", "(gidx1+2)")

    load = get_load_image_uop(shape, (gidx1).lt(5), (gidx0, gidx1+5))
    self.check(load, None, "gidx0", "(gidx1+5)")

  def test_valid_empty_set(self):
    gidx0 = Special("gidx0", 32)
    gidx1 = Special("gidx1", 32)
    shape = (32, 32, 4)
    idx = (gidx0%2, gidx1+2)
    # not empty
    load = get_load_image_uop(shape, (gidx0).lt(8), idx)
    self.check(load, "(gidx0<8)", "(gidx0%2)", "(gidx1+2)")

    # empty -> invalid
    load = get_load_image_uop(shape, (gidx0).lt(8) & (gidx0).lt(8).ne(True), idx)
    load = full_graph_rewrite(load.sink()).src[0]
    self.assertEqual(load.op, UOps.VECTORIZE)
    self.assertEqual(load.dtype.count, 4)

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
    self.check(load, None, "((((idx1*48)+(ridx2*6))+ridx0)+-6)", "(((idx2*2)+ridx1)+-1)")

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

    self.check(load, None, "((((idx1*24)+(ridx2*3))+ridx0)+-3)", "(((idx2*2)+ridx1)+-1)")

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

    self.check(load,
               "((((idx2*2)+ridx0)<11)&((((idx1*8)+ridx1)<3)!=True))",
               "(((idx0+((idx1*512)+(ridx1*64)))+832)%1024)",
               "((((idx2*2)+ridx0)+(((idx1+((ridx1+5)//8))+1)//2))+-4)")

  def test_simplify1(self):
    # idx has the form (A % m, A // m + k) and valid has (c0 < A) and (A < c1)
    gidx = Special("gidx", 512)
    valid = gidx.lt(488) & (gidx).lt(480).ne(True)
    idx = ((gidx*3+18)%26, (gidx*3+18)//26-56)
    load = get_load_image_uop((1, 26, 4), valid, idx)
    self.check(load, None, "((gidx*3)+-1438)", "0")

  def test_simplify2(self):
    # from GPU=1 DEBUG=4 FORWARD_ONLY=1 IMAGE=2 python3 test/test_ops.py TestOps.test_simple_padding_conv2d
    lidx = Special("lidx", 4)
    valid = lidx.lt(3) & lidx.lt(1).ne(True)
    idx = ((lidx+1)%2, (lidx+1)//2-1)
    load = get_load_image_uop((1, 2, 4), valid, idx)
    self.check(load, None, "(lidx+-1)", "0")

  def test_simplify3(self):
    # from openpilot
    idx0 = Special("idx0", 265)
    valid = idx0.lt(201).ne(True)
    idx = ((idx0+55)%64, (idx0+55)//64-4)
    load = get_load_image_uop((1, 64, 4), valid, idx)
    self.check(load, None, "(idx0+-201)", "0")

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
    self.check(load, "(idx0<256)", "((((((idx0*4)+1)%32)*8)+(idx0//32))%64)", "((((idx0*4)+1)%32)//8)")

    load = get_load_image_uop(shape, alu9, (((alu8+(alu3*8))%64),(alu3//8)))
    self.check(load, "(idx0<256)", "((((((idx0*4)+2)%32)*8)+(idx0//32))%64)", "((((idx0*4)+2)%32)//8)")

    load = get_load_image_uop(shape, alu9, (((alu8+(alu4*8))%64),(alu4//8)))
    self.check(load, "(idx0<256)", "((((((idx0*4)+3)%32)*8)+(idx0//32))%64)", "((((idx0*4)+3)%32)//8)")

    load = get_load_image_uop(shape, alu9, (((alu8+(alu5*8))%64),(alu5//8)))
    self.check(load, "(idx0<256)", "(((((idx0*4)%32)*8)+(idx0//32))%64)", "(((idx0*4)%32)//8)")

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
    self.check(load, "(((((idx0*4)+(idx1*256))+1)%768)<640)", "((idx0+((idx1//3)*16))+128)", "(((((idx0*4)+(idx1*256))+1)%768)//64)")

if __name__ == '__main__':
  unittest.main()
