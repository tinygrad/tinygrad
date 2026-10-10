import unittest
import numpy as np
from tinygrad import Tensor, Variable, dtypes
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import graph_rewrite
from tinygrad.codegen.simplify import pm_load_collapse


class TestFlipDecomposition(unittest.TestCase):
  def test_axes(self):
    data = np.arange(30, dtype=np.int32).reshape(2, 3, 5)
    for axes in ((), (0,), (1,), (2,), (-1,), (0, 2), (2, 0), (0, 1, 2)):
      with self.subTest(axes=axes):
        np.testing.assert_array_equal(Tensor(data).flip(axes).numpy(), np.flip(data, axes))
        np.testing.assert_array_equal(Tensor(Tensor(data).uop.flip(axes)).numpy(), np.flip(data, axes))

  def test_empty_and_singleton(self):
    for shape in ((0,), (0, 3), (2, 0, 3), (1,), (1, 3, 1)):
      with self.subTest(shape=shape):
        data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        np.testing.assert_array_equal(Tensor(data).flip(tuple(range(len(shape)))).numpy(), np.flip(data))

  def test_float_bits(self):
    data = np.array([np.nan, np.inf, -0.0, 0.0, -np.inf], dtype=np.float32)
    out = Tensor(data).flip(0).numpy()
    np.testing.assert_array_equal(out.view(np.uint32), data[::-1].view(np.uint32))

  def test_composed_views(self):
    data = np.arange(60, dtype=np.int32).reshape(3, 4, 5)
    x = Tensor(data).permute(2, 0, 1).shrink(((1, 5), None, (1, 4)))
    ref = data.transpose(2, 0, 1)[1:5, :, 1:4]
    np.testing.assert_array_equal(x.flip((0, 2)).numpy(), np.flip(ref, (0, 2)))
    np.testing.assert_array_equal(x.flip((0, 2)).flip((0, 2)).numpy(), ref)

  def test_gradient(self):
    data = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    weights = data + 1
    for axes in ((0,), (1,), (2,), (0, 2)):
      with self.subTest(axes=axes):
        x = Tensor(data)
        grad = (x.flip(axes) * Tensor(weights)).sum().gradient(x)[0]
        np.testing.assert_array_equal(grad.numpy(), np.flip(weights, axes))

  def test_gradient_no_reduction(self):
    for shape, axes in (((32,), (0,)), ((4096,), (0,)), ((2, 3, 5), (0, 2)), ((5, 7), (0, 1)), ((7, 5, 3), (2, 0, 1))):
      with self.subTest(shape=shape, axes=axes):
        x = Tensor.empty(*shape).realize()
        dy = Tensor.empty(*shape).realize()
        for call in x.flip(axes).gradient(x, gradient=dy)[0].schedule_linear().src:
          if call.body.op is Ops.SINK:
            ast = graph_rewrite(call.body, pm_load_collapse)
            self.assertNotIn(Ops.REDUCE, [u.op for u in ast.toposort()])

  def test_gradient_large_and_composed(self):
    for shape, axes in (((127,), (0,)), ((5, 7), (0, 1)), ((7, 5, 3), (2, 0, 1))):
      with self.subTest(shape=shape, axes=axes):
        x = Tensor.empty(*shape)
        weights = np.random.default_rng(0).normal(size=shape).astype(np.float32)
        grad = x.flip(axes).gradient(x, gradient=Tensor(weights))[0]
        np.testing.assert_array_equal(grad.numpy(), np.flip(weights, axes))

  def test_symbolic_zero_and_singleton(self):
    data = np.arange(1, 9, dtype=np.int32).reshape(2, 4)
    weights = np.array([[1, 10, 100, 1000], [2, 20, 200, 2000]], dtype=np.int32)
    x, w = Tensor(data).realize(), Tensor(weights).realize()
    for size in (0, 1, 2, 4):
      with self.subTest(size=size):
        n = Variable('flip_size', 0, 4).bind(size)
        # Unequal weights check order, unlike an unweighted sum of a reversed tensor.
        out = (x[:, :n].flip(1).contiguous() * w[:, :n]).sum().item()
        self.assertEqual(out, int((data[:, :size][:, ::-1] * weights[:, :size]).sum()))

  def test_symbolic_gradient(self):
    weights = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
    x, dy = Tensor.zeros(2, 8).contiguous().realize(), Tensor(weights).realize()
    for lower, sizes in ((0, (0, 1, 2, 5, 8)), (2, (2, 5, 8))):
      for size in sizes:
        with self.subTest(lower=lower, size=size):
          n = Variable(f'flip_grad_size_{lower}', lower, 8).bind(size)
          grad = x[:, :n].flip(1).gradient(x, gradient=dy[:, :n])[0]
          expected = np.zeros_like(weights)
          expected[:, :size] = weights[:, :size][:, ::-1]
          np.testing.assert_array_equal(grad.numpy(), expected)

  def test_assign_self(self):
    x = Tensor([1, 2, 3, 4, 5], dtype=dtypes.int32).realize()
    x.assign(x.flip(0))
    self.assertEqual(x.tolist(), [5, 4, 3, 2, 1])

  def test_partial_store_gradient(self):
    x = Tensor([1., 2., 3., 4.])
    y, value = x.clone(), Tensor([10., 20.])
    y.flip(0)[:2].assign(value)
    gx, gv = (y * Tensor([1., 2., 3., 4.])).sum().gradient(x, value)
    self.assertEqual(gx.tolist(), [1., 2., 0., 0.])
    self.assertEqual(gv.tolist(), [4., 3.])

  def test_constant_graph_size(self):
    self.assertNotIn('FLIP', Ops.__members__)
    counts = []
    for n in (7, 127, 4096):
      x = Tensor.empty(n).uop
      added = set(x.flip(0).toposort()) - set(x.toposort())
      movements = [u.op for u in added if u.op in GroupOp.Movement]
      self.assertLessEqual(set(movements), {Ops.EXPAND, Ops.RESHAPE, Ops.SHRINK, Ops.PERMUTE})
      counts.append(len(movements))
    self.assertEqual(counts, [counts[0]] * len(counts))
    self.assertLessEqual(counts[0], 8)


if __name__ == '__main__':
  unittest.main()
