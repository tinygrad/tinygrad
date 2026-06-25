import unittest
from tinygrad import UOp, dtypes

class TestBitcastSpec(unittest.TestCase):
  def test_bitcast_no_shape_change(self):
    pl = UOp.placeholder((10,10), dtypes.int, 0)
    out = pl.bitcast(dtypes.float)
    self.assertEqual(out.shape, (10,10))

  def test_bitcast_increase_shape(self):
    pl = UOp.placeholder((10,10), dtypes.int, 0)
    out = pl.bitcast(dtypes.short)
    self.assertEqual(out.shape, (10,20))

  def test_bitcast_decrease_shape(self):
    pl = UOp.placeholder((10,10), dtypes.int, 0)
    out = pl.bitcast(dtypes.long)
    self.assertEqual(out.shape, (10,5))

  def test_bitcast_remove_ones(self):
    pl = UOp.placeholder((10,2), dtypes.int, 0)
    out = pl.bitcast(dtypes.long)
    self.assertEqual(out.shape, (10,))

  def test_bitcast_remove_ones_full(self):
    pl = UOp.placeholder((2,), dtypes.int, 0)
    out = pl.bitcast(dtypes.long)
    self.assertEqual(out.shape, ())

  def test_bitcast_add_ones_full(self):
    pl = UOp.placeholder((), dtypes.long, 0)
    out = pl.bitcast(dtypes.int)
    self.assertEqual(out.shape, (2,))

if __name__ == '__main__':
  unittest.main()
