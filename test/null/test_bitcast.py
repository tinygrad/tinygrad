import itertools, unittest
from tinygrad import Tensor, dtypes
from tinygrad.dtype import Invalid
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.uop.symbolic import sym
from tinygrad.uop.spec import spec_program, test_pyrender as check_pyrender
from tinygrad.schedule.prepare import expand_bitcast


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
          self.assertEqual(bitcast(y, big).shape, shape)
          self.assertIs(graph_rewrite(bitcast(y, big), sym), x)
          z = UOp.param(0, small, y.shape)
          self.assertEqual(bitcast(z, big).shape, shape)
          self.assertEqual(bitcast(bitcast(z, big), small).shape, z.shape)
          self.assertIs(graph_rewrite(bitcast(bitcast(z, big), small), sym), z)
          check_pyrender(y)
          check_pyrender(bitcast(z, big))

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
    self.assertEqual(graph_rewrite(y, sym).shape, (3, 2, 2))
    x = UOp.param(0, dtypes.uint8, (4,))
    y = bitcast(bitcast(x, dtypes.uint32), dtypes.uint16)
    self.assertEqual(graph_rewrite(y, sym).shape, (2,))

  def test_invalid_gate(self):
    x = UOp.param(0, dtypes.uint32, (3,))
    cond = UOp.param(1, dtypes.bool, (3,))
    y = bitcast(cond.where(x, UOp.const(Invalid)), dtypes.uint8)
    self.assertEqual(y.simplify().shape, (3, 4))

  def test_lowering(self):
    for src, dst, shape in ((dtypes.uint32, dtypes.uint8, ()), (dtypes.uint32, dtypes.uint8, (2, 3)),
                            (dtypes.uint8, dtypes.uint32, (4,)), (dtypes.uint8, dtypes.uint32, (2, 3, 4))):
      x = bitcast(UOp.param(0, src, shape), dst)
      lowered = expand_bitcast(x)
      self.assertEqual(lowered.shape, x.shape)
      self.assertTrue(all(u.dtype.itemsize == u.src[0].dtype.itemsize for u in lowered.toposort() if u.op is Ops.BITCAST))

  def test_public_api(self):
    x = Tensor.empty(2, 8, dtype=dtypes.uint8)
    y = x.bitcast(dtypes.uint32)
    self.assertEqual(y.shape, (2, 2))
    self.assertEqual(y.uop.op, Ops.BITCAST)
    self.assertEqual(y.uop.src[0].shape, (2, 2, 4))
    z = y.bitcast(dtypes.uint8)
    self.assertEqual(z.shape, x.shape)
    self.assertEqual(z.uop.op, Ops.RESHAPE)
    self.assertEqual(z.uop.src[0].shape, (2, 2, 4))

class TestBitcastRenderer(unittest.TestCase):
  def setUp(self):
    self.renderer = CStyleLanguage(Target())
    self.x = UOp.param(0, dtypes.uint8, 8)
    self.reshape = self.x.reshape((2, 4))
    self.bitcast = bitcast(self.reshape, dtypes.uint32)

  def test_bitcast_reshape(self):
    zero = UOp.const(0, dtypes.int32)
    index = self.bitcast.index(zero)
    code = self.renderer.render([self.x, self.reshape, self.bitcast, *zero.toposort(), index, index.load()])
    self.assertIn("unsigned int val0", code)
    self.assertNotIn("unsigned_char8", code)

  def test_shrink_bitcast_reshape(self):
    offset, size = UOp.const(0, dtypes.int32), UOp.const(1, dtypes.int32)
    shrink = UOp(Ops.SHRINK, src=(self.bitcast, offset, size))
    self.assertIs(spec_program.rewrite(shrink), True)
    code = self.renderer.render([self.x, self.reshape, self.bitcast, *offset.toposort(), *size.toposort(), shrink, shrink.load()])
    self.assertIn("unsigned int val0", code)

  def test_reject_arbitrary_reshape(self):
    with self.assertRaisesRegex(AssertionError, "failed to render Ops.RESHAPE"):
      self.renderer.render([self.x, self.reshape, self.reshape.sink()])

  def test_reject_reshape_with_non_bitcast_user(self):
    with self.assertRaisesRegex(AssertionError, "failed to render Ops.RESHAPE"):
      self.renderer.render([self.x, self.reshape, self.bitcast, UOp.sink(self.reshape, self.bitcast)])

if __name__ == '__main__': unittest.main()
