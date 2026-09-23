import itertools, unittest
from tinygrad import Tensor, dtypes
from tinygrad.dtype import Invalid
from tinygrad.uop.ops import Ops, UOp

def bitcast(x:UOp, dtype): return x.alu(Ops.BITCAST, arg=dtype)

class TestBitcastShape(unittest.TestCase):
  def test_same_size(self):
    for shape in ((), (3,), (2, 3), (0,)):
      x = UOp.param(0, dtypes.uint32, shape)
      self.assertEqual(bitcast(x, dtypes.float32).shape, shape)

  def test_roundtrip(self):
    for big, small in itertools.combinations((dtypes.uint64, dtypes.uint32, dtypes.uint16, dtypes.uint8), 2):
      for shape in ((), (3,), (2, 3), (0,), (2, 0)):
        with self.subTest(big=big, small=small, shape=shape):
          x = UOp.param(0, big, shape)
          y = bitcast(x, small)
          self.assertEqual(y.shape, shape + (big.itemsize//small.itemsize,))
          self.assertEqual(bitcast(y, big).simplify().shape, shape)
          z = UOp.param(0, small, y.shape)
          self.assertEqual(bitcast(z, big).shape, shape)
          self.assertEqual(bitcast(bitcast(z, big), small).simplify().shape, z.shape)

  def test_widen_requires_exact_dimension(self):
    for shape in ((), (0,), (1,), (2,), (3,), (5,), (8,), (2, 8), (4, 1)):
      with self.subTest(shape=shape), self.assertRaises(RuntimeError):
        bitcast(UOp.param(0, dtypes.uint8, shape), dtypes.uint32).shape
    self.assertEqual(bitcast(UOp.param(0, dtypes.uint8, (2, 4)), dtypes.uint32).shape, (2,))

  def test_symbolic_shape(self):
    n = UOp.variable('n', 1, 8)
    x = UOp.param(0, dtypes.uint32, (n,))
    self.assertEqual(bitcast(x, dtypes.uint8).shape, (n, 4))
    self.assertEqual(bitcast(bitcast(x, dtypes.uint8), dtypes.uint32).shape, (n,))
    with self.assertRaises(RuntimeError): bitcast(UOp.param(0, dtypes.uint8, (n,)), dtypes.uint32).shape

  def test_chained_bitcasts_do_not_flatten_dimensions(self):
    x = UOp.param(0, dtypes.uint32, (3,))
    y = bitcast(bitcast(x, dtypes.uint16), dtypes.uint8)
    self.assertEqual(y.simplify().shape, (3, 2, 2))
    x = UOp.param(0, dtypes.uint8, (4,))
    y = bitcast(bitcast(x, dtypes.uint32), dtypes.uint16)
    self.assertEqual(y.simplify().shape, (2,))

  def test_invalid_gate(self):
    x = UOp.param(0, dtypes.uint32, (3,))
    cond = UOp.param(1, dtypes.bool, (3,))
    y = bitcast(cond.where(x, UOp.const(Invalid)), dtypes.uint8)
    self.assertEqual(y.simplify().shape, (3, 4))

  def test_public_api(self):
    x = Tensor.empty(2, 8, dtype=dtypes.uint8)
    y = x.bitcast(dtypes.uint32)
    self.assertEqual(y.shape, (2, 2))
    self.assertEqual(y.dtype, dtypes.uint32)
    z = y.bitcast(dtypes.uint8)
    self.assertEqual(z.shape, x.shape)
    self.assertEqual(z.dtype, dtypes.uint8)

if __name__ == '__main__': unittest.main()
