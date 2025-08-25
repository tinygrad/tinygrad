import unittest
from tinygrad import Tensor, Variable
from tinygrad.shape.shapetracker import View
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import sym_infer
from tinygrad.dtype import dtypes
from tinygrad.device import Device
from examples.gpt2 import Attention
import numpy as np

class TestSymbolicOps(unittest.TestCase):
  def test_plus1(self):
    def f(a): return (a+1).realize()
    a = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      # Pad to max size then shrink down to remove symbolic shape
      symbolic = f(a[:, :vi]).pad((0,10-vi))[:, :i].numpy()
      expected = f(a[:, :i]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_add(self):
    def f(a, b): return (a+b).realize()
    a = Tensor.rand(3, 10)
    b = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = f(a[:, :vi], b[:, :vi]).pad((0,10-vi))[:, :i].numpy()
      expected = f(a[:, :i], b[:, :i]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_matmul(self):
    def f(a, b): return (a@b).realize()
    a = Tensor.rand(3, 10)
    b = Tensor.rand(10, 5)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = f(a[:, :vi], b[:vi, :]).numpy()
      expected = f(a[:, :i], b[:i, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_attention(self, dropout_p=0.0, imin=1, imax=5, use_symbolic=True):
    def f(q, k, v): return Tensor.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p).realize()
    q = Tensor.rand(2, 1, 4, 8)
    k = Tensor.rand(2, 10, 4, 8)
    v = Tensor.rand(2, 10, 4, 8)
    for i in range(imin, imax):
      vi = Variable("i", 1, 10).bind(i) if use_symbolic else i
      Tensor.realize(q, k, v)
      GlobalCounters.reset()
      symbolic = f(q, k[:, :vi, :, :], v[:, :vi, :, :]).reshape(2, 4, 1, 8).numpy()
      expected = f(q, k[:, :i, :, :], v[:, :i, :, :]).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_attention_cmp_symbolic(self):
    # symbolic isn't seeing if i == i, so it's not putting them on the same axis
    self.test_attention(imin=4, imax=5, use_symbolic=False)
    self.test_attention(imin=4, imax=5, use_symbolic=True)

  # until this works, symbolic single kernel softmax won't
  @unittest.expectedFailure
  def test_attention_simple_view(self):
    i = Variable("i", 2, 10)
    v1 = View.create((2,4,1,i,i), ((i*4),i,0,0,1))
    v2 = View.create((2,4,1,i,i,i), (((i*i)*4),(i*i),0,0,i,1))
    self.assertIsNotNone(v1+v2)

  def test_attention_training(self):
    with Tensor.train():
      self.test_attention(dropout_p=0.0)
      with self.assertRaises(ValueError):
        # symbolic shape dropout is not supported
        self.test_attention(dropout_p=0.5)

  def test_attention_pos_0_sz_0(self):
    Attention(128, 8)(Tensor.ones(1, 0, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_attention_pos_0_sz_1(self):
    Attention(128, 8)(Tensor.ones(1, 1, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_attention_pos_0_sz_2(self):
    Attention(128, 8)(Tensor.ones(1, 2, 128), Variable("start_pos", 0, 128).bind(0), None)

  def test_cat_dim0(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    a = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      b = Tensor.rand(2, 3)
      symbolic = f(a[:vi, :], b).pad(((0,12-vi), (0,0)))[:i+2].numpy()
      expected = f(a[:i, :], b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim1(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    a = Tensor.rand(3, 10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      b = Tensor.rand(3, 2)
      symbolic = f(a[:, :vi], b).pad(((0,0), (0,12-vi)))[:, :i+2].numpy()
      expected = f(a[:, :i], b).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim0_two_vars(self):
    def f(a, b): return a.cat(b, dim=0).realize()
    a = Tensor.rand(10, 3)
    b = Tensor.rand(10, 3)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = f(a[:vi, :], b[:vj, :])
        symbolic = symbolic.pad(((0,10-vi),(0,0))).pad(((0,10-vj),(0,0)))[:i+j].numpy()
        expected = f(a[:i, :], b[:j, :]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_cat_dim1_two_vars(self):
    def f(a, b): return a.cat(b, dim=1).realize()
    a = Tensor.rand(3, 10)
    b = Tensor.rand(3, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = f(a[:, :vi], b[:, :vj])
        symbolic = symbolic.pad(((0,0),(0,10-vi))).pad(((0,0),(0,10-vj)))[:, :i+j].numpy()
        expected = f(a[:, :i], b[:, :j]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_two_vars_plus1_ij(self):
    def f(a, b): return (a@b+1).realize()
    a = Tensor.rand(10, 3)
    b = Tensor.rand(3, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = f(a[:vi, :], b[:, :vj]).pad(((0,10-vi),(0,10-vj)))[:i, :j].numpy()
        expected = f(a[:i, :], b[:, :j]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_two_vars_plus1_ji(self):
    # reverse the order of variables
    def f(a, b): return (a@b+1).realize()
    a = Tensor.rand(10, 3)
    b = Tensor.rand(3, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        symbolic = f(a[:vj, :], b[:, :vi]).pad(((0,10-vj),(0,10-vi)))[:j, :i].numpy()
        expected = f(a[:j, :], b[:, :i]).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_reshape_from_symbolic(self):
    a = Tensor.rand(30)
    for i in range(3, 5):
      vi = Variable("i", 3, 10).bind(i)
      symbolic = a[:vi*3].reshape((3, 3))
      symbolic = symbolic.numpy()
      # reshaping from symbolic can be a shrink - should it?
      expected = a[:i*3].shrink(((0, 9),)).reshape((3, 3)).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_invalid_reshape_from_symbolic_simple(self):
    a = Tensor.rand(30)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = a[:vi*3]
      with self.assertRaises(ValueError): symbolic.reshape((3, 3))

  def test_invalid_symbolic_reshape(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(30)
      # Reshaping to introduce symbolic shape now asserts
      with self.assertRaises(AssertionError): a.reshape((3, vi))

  def test_shrink(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(7, 11)
      symbolic = a.shrink(((3,5),(vi,vi+2)))
      symbolic = symbolic.numpy()
      expected = a.shrink(((3,5),(i,i+2))).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_slice(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.rand(7, 11)
      symbolic = a[3:5, vi:vi+2]
      symbolic = symbolic.numpy()
      expected = a[3:5, i:i+2].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_slice_no_start(self):
    a = Tensor.rand(7, 11)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = a[3:5, :vi:1]
      symbolic = symbolic.pad(((0,0),(0,10-vi)))[:, :i].numpy()
      expected = a[3:5, :i:1].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_expand_padded(self):
    a = Tensor(1).unsqueeze(0).pad((0, 1)).unsqueeze(0)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = a.expand(vi, 2)
      symbolic = symbolic.pad(((0,10-vi),(0,0)))[:i].numpy()
      expected = a.expand(i, 2).numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_slice_var_shape(self):
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      a = Tensor.ones(vi, 11).contiguous()
      symbolic = a[:, 1:2]
      symbolic = symbolic.pad(((0,10-vi),(0,0)))[:i].numpy()
      expected = a.pad(((0,10-vi),(0,0)))[:i]  # no-op
      expected = expected[:, 1:2].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_ones_sum(self):
    t = Tensor.ones(10)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      symbolic = t[:vi].sum().item()
      expected = t[:i].sum().item()
      np.testing.assert_equal(symbolic, expected)

  def test_mean(self):
    a = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      for axis in [None, 0]:
        expected = a[:i].mean(axis).numpy()
        symbolic = a[:vi].mean(axis).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_mean_dim1(self):
    a = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      expected = a[:i].mean(1).numpy()
      symbolic = a[:vi].mean(1).pad((0,10-vi))[:i].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_mean_2d(self):
    a = Tensor.rand(10, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        expected = a[:i, :j].mean().numpy()
        symbolic = a[:vi, :vj].mean().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_mean_2d_dim0(self):
    a = Tensor.rand(10, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        expected = a[:i, :j].mean(0).numpy()
        symbolic = a[:vi, :vj].mean(0).pad((0,10-vj))[:j].numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_mean_2d_dim1(self):
    a = Tensor.rand(10, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        expected = a[:i, :j].mean(1).numpy()
        symbolic = a[:vi, :vj].mean(1).pad((0,10-vi))[:i].numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var(self):
    a = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      for axis in [None, 0]:
        expected = a[:i].var(axis).numpy()
        symbolic = a[:vi].var(axis).numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var_dim1(self):
    a = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      expected = a[:i].var(1).numpy()
      symbolic = a[:vi].var(1).pad((0,10-vi))[:i].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var_2d(self):
    a = Tensor.rand(10, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        expected = a[:i, :j].var().numpy()
        symbolic = a[:vi, :vj].var().numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var_2d_dim0(self):
    a = Tensor.rand(10, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        expected = a[:i, :j].var(0).numpy()
        symbolic = a[:vi, :vj].var(0).pad((0,10-vj))[:j].numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_var_2d_dim1(self):
    a = Tensor.rand(10, 10)
    for i in range(1, 5):
      for j in range(1, 5):
        vi = Variable("i", 1, 10).bind(i)
        vj = Variable("j", 1, 10).bind(j)
        expected = a[:i, :j].var(1).numpy()
        symbolic = a[:vi, :vj].var(1).pad((0,10-vi))[:i].numpy()
        np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=1e-6)

  def test_bitcast_down(self):
    a = Tensor.rand(10, 3)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      expected = a[:i].bitcast(dtypes.uint8).numpy()
      symbolic = a[:vi].bitcast(dtypes.uint8).pad(((0,10-vi),(0,0)))[:i].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=0)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "no uint64")
  def test_bitcast_up(self):
    a = Tensor.rand(10, 4)
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      expected = a[:i].bitcast(dtypes.uint64).numpy()
      symbolic = a[:vi].bitcast(dtypes.uint64).pad(((0,10-vi),(0,0)))[:i].numpy()
      np.testing.assert_allclose(symbolic, expected, atol=1e-6, rtol=0)

  @unittest.expectedFailure
  def test_conv2d_ceildiv_edge_case(self):
    v = Variable('v', 11, 50_000)
    val = 39601
    x = Tensor.randn(1, 22, 50_000)[:, :, :v.bind(val)]
    weight = Tensor.randn(256, 22, 12)

    result = x.conv2d(weight=weight, groups=1, stride=6, dilation=1, padding=(3, 3))
    var_val = {v: val}
    shape = tuple(sym_infer(s, var_val) for s in result.shape)
    self.assertEqual(shape, (1, 256, 6600))  # TODO: fails if ceildiv is incorrect
    # TODO: test output is correct

if __name__ == '__main__':
  unittest.main()
