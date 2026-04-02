# test cases are modified from pytorch test_indexing.py

import unittest

from tinygrad import Tensor
from tinygrad.codegen.simplify import flatten_range
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp, Ops, AxisType
from tinygrad.schedule.indexing import apply_movement_op

class TestIndexing(unittest.TestCase):
  def test_single_int(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[4].shape, (7, 3))

  def test_multiple_int(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[4].shape, (7, 3))
    self.assertEqual(v[4, :, 1].shape, (7,))

  def test_none(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[None].shape, (1, 5, 7, 3))
    self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
    self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
    self.assertEqual(v[..., None].shape, (5, 7, 3, 1))

  def test_int_indices(self):
    v = Tensor.randn(5, 7, 3)
    self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
    self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
    self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

  def test_index_src_datatype(self):
    src = Tensor.ones(3, 2, 4)
    # test index
    res = src[[0, 2, 1], :, :]
    self.assertEqual(res.shape, src.shape)

  def test_empty_slice(self):
    x = Tensor.randn(2, 3, 4, 5)
    y = x[:, :, :, 1]
    z = y[:, 1:1, :]
    self.assertEqual((2, 0, 4), z.shape)

  def test_invalid_index(self):
    x = Tensor.arange(0, 16).reshape(4, 4)
    self.assertRaises(TypeError, lambda: x["0":"1"])

  def test_out_of_bound_index(self):
    x = Tensor.arange(0, 100).reshape(2, 5, 10)
    self.assertRaises(IndexError, lambda: x[0, 5])
    self.assertRaises(IndexError, lambda: x[4, 5])
    self.assertRaises(IndexError, lambda: x[0, 1, 15])
    self.assertRaises(IndexError, lambda: x[:, :, 12])

class TestNumpy(unittest.TestCase):
  def test_index_no_floats(self):
    a = Tensor([[[5.]]])

    self.assertRaises(IndexError, lambda: a[0.0])
    self.assertRaises(IndexError, lambda: a[0, 0.0])
    self.assertRaises(IndexError, lambda: a[0.0, 0])
    self.assertRaises(IndexError, lambda: a[0.0, :])
    self.assertRaises(IndexError, lambda: a[:, 0.0])
    self.assertRaises(IndexError, lambda: a[:, 0.0, :])
    self.assertRaises(IndexError, lambda: a[0.0, :, :])
    self.assertRaises(IndexError, lambda: a[0, 0, 0.0])
    self.assertRaises(IndexError, lambda: a[0.0, 0, 0])
    self.assertRaises(IndexError, lambda: a[0, 0.0, 0])
    self.assertRaises(IndexError, lambda: a[-1.4])
    self.assertRaises(IndexError, lambda: a[0, -1.4])
    self.assertRaises(IndexError, lambda: a[-1.4, 0])
    self.assertRaises(IndexError, lambda: a[-1.4, :])
    self.assertRaises(IndexError, lambda: a[:, -1.4])
    self.assertRaises(IndexError, lambda: a[:, -1.4, :])
    self.assertRaises(IndexError, lambda: a[-1.4, :, :])
    self.assertRaises(IndexError, lambda: a[0, 0, -1.4])
    self.assertRaises(IndexError, lambda: a[-1.4, 0, 0])
    self.assertRaises(IndexError, lambda: a[0, -1.4, 0])
    # these two trigger slice internal type verification first
    self.assertRaises(TypeError, lambda: a[0.0:, 0.0])
    self.assertRaises(TypeError, lambda: a[0.0:, 0.0,:])

  def test_none_index(self):
    # `None` index adds newaxis
    a = Tensor([1, 2, 3])
    self.assertEqual(a[None].ndim, a.ndim+1)

  def test_everything_returns_views(self):
    # Before `...` would return a itself.
    a = Tensor([5])

    self.assertIs(a, a[()])
    self.assertIs(a, a[...])
    self.assertIs(a, a[:])

  def test_broaderrors_indexing(self):
    a = Tensor.zeros(5, 5)
    self.assertRaises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
    self.assertRaises(IndexError, a.contiguous().__setitem__, ([0, 1], [0, 1, 2]), 0)

class TestRangeifyMovementFastpaths(unittest.TestCase):
  def _eval_expr(self, expr:UOp, values:dict[UOp, int]) -> int:
    subs = {k: UOp.const(dtypes.weakint, v) for k,v in values.items()}
    ret = expr.substitute(subs).ssimplify()
    self.assertIsInstance(ret, int)
    return ret

  def test_flatten_range_preserves_range_order(self):
    r0, r1, r2 = UOp.range(4, 0, AxisType.LOOP), UOp.range(5, 1, AxisType.LOOP), UOp.range(6, 2, AxisType.LOOP)
    body = UOp.const(dtypes.int, 0)
    ret = flatten_range(body.end((r0 + r1).simplify(), (r1 + r2).simplify()))
    self.assertIsNotNone(ret)
    self.assertEqual(ret.src[1:], (r0, r1, r2))

  def test_reshape_remove_insert_ones(self):
    r0, r1 = UOp.range(77, 0, AxisType.LOOP), UOp.range(768, 1, AxisType.LOOP)
    z = UOp.const(dtypes.weakint, 0)

    removed = apply_movement_op(Ops.RESHAPE, (77, 768), (1, 77, 768), (z, r0, r1))
    self.assertEqual(removed, (r0, r1))

    inserted = apply_movement_op(Ops.RESHAPE, (1, 77, 768, 1), (1, 77, 768), (z, r0, r1))
    self.assertEqual(inserted, (z, r0, r1, z))

  def test_reshape_single_dim_split(self):
    r0, r1 = UOp.range(77, 0, AxisType.LOOP), UOp.range(768, 1, AxisType.LOOP)
    ret = apply_movement_op(Ops.RESHAPE, (59136,), (77, 768), (r0, r1))
    self.assertEqual(len(ret), 1)
    self.assertEqual(ret[0].render(), (r0*768 + r1).render())

  def test_reshape_single_dim_split_with_shared_prefix(self):
    z, r0, r1 = UOp.const(dtypes.weakint, 0), UOp.range(77, 0, AxisType.LOOP), UOp.range(768, 1, AxisType.LOOP)
    r2, r3 = UOp.range(12, 2, AxisType.LOOP), UOp.range(64, 3, AxisType.LOOP)

    split = apply_movement_op(Ops.RESHAPE, (1, 77, 768), (1, 77, 12, 64), (z, r0, r2, r3))
    self.assertEqual(len(split), 3)
    self.assertEqual(split[0], z)
    self.assertEqual(split[1], r0)
    self.assertEqual(split[2].render(), (r2*64 + r3).render())

    merged = apply_movement_op(Ops.RESHAPE, (1, 77, 12, 64), (1, 77, 768), (z, r0, r1))
    self.assertEqual(len(merged), 4)
    self.assertEqual(merged[0], z)
    self.assertEqual(merged[1], r0)
    self.assertEqual(merged[2].render(), (r1//64).render())
    self.assertEqual(merged[3].render(), (r1%64).render())

  def test_reshape_multi_dim_static_split(self):
    r0, r1 = UOp.range(4, 0, AxisType.LOOP), UOp.range(6, 1, AxisType.LOOP)
    ret = apply_movement_op(Ops.RESHAPE, (2, 3, 4), (4, 6), (r0, r1))
    self.assertEqual(len(ret), 3)
    for i in range(4):
      for j in range(6):
        linear = i*6 + j
        values = {r0:i, r1:j}
        self.assertEqual(self._eval_expr(ret[0], values), linear//12)
        self.assertEqual(self._eval_expr(ret[1], values), (linear//4)%3)
        self.assertEqual(self._eval_expr(ret[2], values), linear%4)

  def test_reshape_multi_dim_static_merge_with_prefix_suffix(self):
    z = UOp.const(dtypes.weakint, 0)
    r0, r1, r2 = UOp.range(5, 0, AxisType.LOOP), UOp.range(4, 1, AxisType.LOOP), UOp.range(6, 2, AxisType.LOOP)
    ret = apply_movement_op(Ops.RESHAPE, (1, 5, 2, 3, 4, 1), (1, 5, 4, 6, 1), (z, r0, r1, r2, z))
    self.assertEqual(len(ret), 6)
    self.assertEqual(ret[0], z)
    self.assertEqual(ret[1], r0)
    self.assertEqual(ret[5], z)
    for i in range(5):
      for j in range(4):
        for k in range(6):
          linear = j*6 + k
          values = {r0:i, r1:j, r2:k}
          self.assertEqual(self._eval_expr(ret[2], values), linear//12)
          self.assertEqual(self._eval_expr(ret[3], values), (linear//4)%3)
          self.assertEqual(self._eval_expr(ret[4], values), linear%4)

if __name__ == '__main__':
  unittest.main()
