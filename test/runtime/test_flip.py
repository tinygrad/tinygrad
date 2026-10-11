import unittest
import numpy as np
from tinygrad import Tensor, Variable, GlobalCounters
from tinygrad.engine.realize import run_linear

class TestFlip(unittest.TestCase):
  def test_gradient(self):
    for shape, axes in (((32,), (0,)), ((127,), (0,)), ((4096,), (0,)), ((2, 3, 5), (0, 2)), ((5, 7), (0, 1)), ((7, 5, 3), (2, 0, 1))):
      with self.subTest(shape=shape, axes=axes):
        x = Tensor.empty(*shape).realize()
        weights = np.random.default_rng(0).normal(size=shape).astype(np.float32)
        dy = Tensor(weights).realize()
        GlobalCounters.reset()
        grad = x.flip(axes).gradient(x, gradient=dy)[0]
        np.testing.assert_array_equal(grad.numpy(), np.flip(weights, axes))
        # Neither arithmetic nor memory traffic should grow quadratically.
        self.assertLessEqual(GlobalCounters.global_ops, 16*weights.size)
        self.assertLessEqual(GlobalCounters.global_mem, 4*weights.nbytes)

  def test_symbolic_zero_and_singleton(self):
    data = np.arange(1, 9, dtype=np.int32).reshape(2, 4)
    weights = np.array([[1, 10, 100, 1000], [2, 20, 200, 2000]], dtype=np.int32)
    x, w = Tensor(data).realize(), Tensor(weights).realize()
    for size in (0, 1, 2, 4):
      with self.subTest(size=size):
        n = Variable('flip_size', 0, 4).bind(size)
        # Nonconstant weights check order; this also catches WebGPU's unsupported i64 index regression.
        out = (x[:, :n].flip(1).contiguous() * w[:, :n]).sum().item()
        self.assertEqual(out, int((data[:, :size][:, ::-1] * weights[:, :size]).sum()))

  def test_symbolic_gradient(self):
    weights = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
    x, dy = Tensor.zeros(2, 8).contiguous().realize(), Tensor(weights).realize()
    for axes in ((0,), (1,), (0, 1)):
      for lower, sizes in ((0, (0, 1, 2, 5, 8)), (2, (2, 5, 8))):
        n = Variable(f'flip_grad_size_{lower}', lower, 8).bind(sizes[0])
        grad = x[:, :n].flip(axes).gradient(x, gradient=dy[:, :n])[0].contiguous()
        # The graph depends on the bounds, not the binding. Schedule once and run every size through it.
        linear, var_vals = grad.linear_with_vars()
        for size in sizes:
          with self.subTest(axes=axes, lower=lower, size=size):
            run_linear(linear, var_vals | {n.expr:size})
            expected = np.zeros_like(weights)
            expected[:, :size] = np.flip(weights[:, :size], axes)
            np.testing.assert_array_equal(grad.numpy(), expected)

  def test_symbolic_partial_store(self):
    data, values = [1., 2., 3., 4.], [10., 20., 30., 40.]
    for size in (0, 1, 2, 4):
      with self.subTest(size=size):
        x = Tensor(data).realize()
        n = Variable('flip_store_size', 0, 4).bind(size)
        x[:n].flip(0).assign(Tensor(values)[:n])
        self.assertEqual(x.tolist(), values[:size][::-1] + data[size:])

  def test_partial_store_gradient(self):
    x = Tensor([1., 2., 3., 4.])
    y, value = x.clone(), Tensor([10., 20.])
    y.flip(0)[:2].assign(value)
    gx, gv = (y * Tensor([1., 2., 3., 4.])).sum().gradient(x, value)
    self.assertEqual(gx.tolist(), [1., 2., 0., 0.])
    self.assertEqual(gv.tolist(), [4., 3.])

if __name__ == '__main__': unittest.main()
