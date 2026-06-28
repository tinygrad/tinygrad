import unittest
from tinygrad import UOp, Variable, dtypes
from tinygrad.uop.ops import shape_to_shape_arg, ParamArg, Ops, AddrSpace

def placeholder(shape, dtype, slot):
  return UOp(Ops.PARAM, dtype, (shape_to_shape_arg(shape),), arg=ParamArg(slot, AddrSpace.GLOBAL))

class TestBitcastSpec(unittest.TestCase):
  def test_bitcast_no_shape_change(self):
    pl = placeholder((10,10), dtypes.int, 0)
    out = pl.bitcast(dtypes.float)
    self.assertEqual(out.shape, (10,10))

  def test_bitcast_increase_shape(self):
    pl = placeholder((10,10), dtypes.int, 0)
    out = pl.bitcast(dtypes.short)
    self.assertEqual(out.shape, (10,20))

  def test_bitcast_decrease_shape(self):
    pl = placeholder((10,10), dtypes.int, 0)
    out = pl.bitcast(dtypes.long)
    self.assertEqual(out.shape, (10,5))

  def test_bitcast_remove_ones(self):
    pl = placeholder((10,2), dtypes.int, 0)
    out = pl.bitcast(dtypes.long)
    self.assertEqual(out.shape, (10,1))

  def test_bitcast_remove_ones_full(self):
    pl = placeholder((2,), dtypes.int, 0)
    out = pl.bitcast(dtypes.long)
    self.assertEqual(out.shape, (1,))

  def test_bitcast_add_ones_full(self):
    pl = placeholder((), dtypes.long, 0)
    out = pl.bitcast(dtypes.int)
    self.assertEqual(out.shape, (2,))

  def test_bitcast_add_ones_full_uchar(self):
    pl = placeholder((), dtypes.long, 0)
    out = pl.bitcast(dtypes.uchar)
    self.assertEqual(out.shape, (8,))

  def test_bitcast_scalar_widen_unsupported(self):
    with self.assertRaisesRegex(RuntimeError, "must be an expanding bitcast"):
      placeholder((), dtypes.int, 0).bitcast(dtypes.long).shape

  def test_bitcast_unsupported_size(self):
    with self.assertRaisesRegex(RuntimeError, "unsupported size in bitcast"):
      placeholder((3,), dtypes.uchar, 0).bitcast(dtypes.short).shape

  def test_bitcast_symbolic_supported_size(self):
    n = Variable("n", 1, 10)
    out = placeholder((n*2,), dtypes.uchar, 0).bitcast(dtypes.short)
    self.assertEqual(out.shape[0].render(), n.render())

  def test_bitcast_symbolic_unsupported_size(self):
    n = Variable("n", 1, 10)
    with self.assertRaisesRegex(RuntimeError, "unsupported size in bitcast"):
      placeholder((n,), dtypes.uchar, 0).bitcast(dtypes.short).shape

if __name__ == '__main__':
  unittest.main()
