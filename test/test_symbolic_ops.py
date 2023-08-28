import unittest
from tinygrad.shape.symbolic import Variable
from tinygrad.helpers import getenv, CI
from tinygrad.tensor import Tensor, Device
import numpy as np

@unittest.skipIf(getenv("ARM64") or getenv("PTX"), "ARM64 and PTX are not supported")
@unittest.skipUnless(Device.DEFAULT in ["GPU", "METAL", "CLANG", "CUDA"], f"{Device.DEFAULT} is not supported")
class TestSymbolicOps(unittest.TestCase):
  def test_plus1(self):
    def f(a): return (a+1).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(3, i)
      symbolic = f(a.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_add(self):
    def f(a, b): return (a+b).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, i)
      symbolic = f(a.reshape(3, vi), b.reshape(3, vi)).reshape(3, i).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(3, i)
      b = Tensor.rand(i, 5)
      symbolic = f(a.reshape(3, vi), b.reshape(vi, 5)).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_matmul_same_var_different_val(self):
    def f(a, b): return (a@b).realize()
    vi = Variable("i", 1, 10)
    a = Tensor.rand(3, 4)
    b = Tensor.rand(7, 5)
    with self.assertRaises(AssertionError):
      f(a.reshape(3, vi), b.reshape(vi, 5)).numpy()

  def test_attention(self, dropout_p=0.0):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      q = Tensor.rand(2, 1, 4, 8)
      k = Tensor.rand(2, i, 4, 8)
      v = Tensor.rand(2, i, 4, 8)
      symbolic = f(q, k.reshape(2, vi, 4, 8), v.reshape(2, vi, 4, 8)).reshape(2, 4, 1, 8).numpy()
      expected = f(q, k, v).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_attention_training(self):
    Tensor.training = True
    self.test_attention(dropout_p=0.0)
    with self.assertRaises(TypeError):
      # symbolic shape dropout is not supported
      self.test_attention(dropout_p=0.5)

  def test_cat_dim0(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(i, 3)
      b = Tensor.rand(2, 3)
      symbolic = f(a.reshape(vi, 3), b).reshape(i+2, 3).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim1(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(3, i)
      b = Tensor.rand(3, 2)
      symbolic = f(a.reshape(3, vi), b).reshape(3, i+2).numpy()
      expected = f(a, b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim0_two_vars(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    vi = Variable("i", 1, 10)
    vj = Variable("j", 1, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        a = Tensor.rand(i, 3)
        b = Tensor.rand(j, 3)
        symbolic = f(a.reshape(vi, 3), b.reshape(vj, 3)).reshape(i+j, 3).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim1_two_vars(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    vi = Variable("i", 1, 10)
    vj = Variable("j", 1, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        a = Tensor.rand(3, i)
        b = Tensor.rand(3, j)
        symbolic = f(a.reshape(3, vi), b.reshape(3, vj)).reshape(3, i+j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_two_vars_plus1(self):
    def f(a, b): return (a@b+1).realize()
    vi = Variable("i", 1, 10)
    vj = Variable("j", 1, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        a = Tensor.rand(i, 3)
        b = Tensor.rand(3, j)
        symbolic = f(a.reshape(vi, 3), b.reshape(3, vj)).reshape(i, j).numpy()
        expected = f(a, b).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_shrink(self):
    vi = Variable("i", 1, 10)
    for i in range(1, 5):
      a = Tensor.rand(7, 11)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic.lazydata.st.var_vals[vi] = i
      symbolic = symbolic.numpy()
      expected = a.shrink(((3,5),(i,i+2))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
  unittest.main()