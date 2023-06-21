import dataclasses
import numpy as np
import torch
import unittest
import itertools
from tinygrad.tensor import Tensor, Device
from tinygrad.helpers import dtypes
from extra.gradcheck import numerical_jacobian, jacobian, gradcheck

x_init = np.random.randn(1,3).astype(np.float32)
U_init = np.random.randn(3,3).astype(np.float32)
V_init = np.random.randn(3,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTinygrad(unittest.TestCase):
  def test_zerodim_initialization(self):
    a = Tensor(55)
    b = Tensor(3.14)

    self.assertEqual(a.shape, ())
    self.assertEqual(b.shape, ())

  def test_plus_equals(self):
    a = Tensor.randn(10,10)
    b = Tensor.randn(10,10)
    c = a + b
    val1 = c.numpy()
    a += b
    val2 = a.numpy()
    np.testing.assert_allclose(val1, val2)

  def test_backward_pass(self):
    def test_tinygrad():
      x = Tensor(x_init, requires_grad=True)
      W = Tensor(W_init, requires_grad=True)
      m = Tensor(m_init)
      out = x.dot(W).relu()
      out = out.log_softmax()
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.cpu().numpy(), x.grad.cpu().numpy(), W.grad.cpu().numpy()

    def test_pytorch():
      x = torch.tensor(x_init, requires_grad=True)
      W = torch.tensor(W_init, requires_grad=True)
      m = torch.tensor(m_init)
      out = x.matmul(W).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.detach().numpy(), x.grad, W.grad

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_backward_pass_diamond_model(self):
    def test_tinygrad():
      u = Tensor(U_init, requires_grad=True)
      v = Tensor(V_init, requires_grad=True)
      w = Tensor(W_init, requires_grad=True)
      x = u.mul(v).relu()
      y = u.mul(w).relu()
      out = x.add(y).mul(y).relu()
      out = out.log_softmax()
      out = out.sum()
      out.backward()
      return out.cpu().numpy(), u.cpu().grad.numpy(), v.cpu().grad.numpy(), w.cpu().grad.numpy()

    def test_pytorch():
      u = torch.tensor(U_init, requires_grad=True)
      v = torch.tensor(V_init, requires_grad=True)
      w = torch.tensor(W_init, requires_grad=True)
      x = u.mul(v).relu()
      y = u.mul(w).relu()
      out = x.add(y).mul(y).relu()
      out = torch.nn.functional.log_softmax(out, dim=1)
      out = out.sum()
      out.backward()
      return out.detach().numpy(), u.grad, v.grad, w.grad

    for x,y in zip(test_tinygrad(), test_pytorch()):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_nograd(self):
    x = Tensor(x_init, requires_grad=False)
    m = Tensor(m_init, requires_grad=False)
    W = Tensor(W_init, requires_grad=True)
    tmp = x.mul(m)
    mm = tmp.matmul(W)
    out = mm.relu()
    out = out.sum()
    out.backward()
    assert x.grad is None
    assert m.grad is None
    assert tmp.grad is None
    assert mm.grad is not None
    assert W.grad is not None

  def test_dropout(self):
    Tensor.training = True
    n, rate = 1_000_000, 0.1
    w = Tensor.ones(n).dropout(rate)
    non_zeros = np.count_nonzero(w.cpu().numpy())
    expected = n * (1 - rate)
    np.testing.assert_allclose(non_zeros, expected, rtol=2e-3)

  def test_onehot(self):
    np.random.seed(1337)
    n, classes = 100, 20
    x = np.random.randint(0, classes, n).astype(np.int64)
    y = Tensor(x).onehot(classes)
    y_torch = torch.nn.functional.one_hot(torch.tensor(x), num_classes=classes)
    np.testing.assert_allclose(y.cpu().numpy(), y_torch.cpu().numpy())

  def test_cross_entropy(self):
    np.random.seed(1337)
    n, classes = 100, 10
    label = np.random.randint(0, classes, n, dtype=np.int64)
    label_mask = np.random.choice((0, 1), size=n, p=(0.3, 0.7)).astype(np.int32)
    weights = np.random.choice((0, 1, 2, 3, 4), size=classes, p=(0.9, 0.05, 0.025, 0.015, 0.01)).astype(np.float32)
    x = np.random.uniform(0, 10, (n, classes)).astype(np.float32)
    
    # test cases
    ignore_index = -100
    reduction_choices = ['mean', 'sum', 'none']
    mask_labels_choices = [False, True]
    apply_weights_choices = [False, True]
    
    for reduction_choice, mask_label_choice, apply_weight_choice in itertools.product(reduction_choices, mask_labels_choices, apply_weights_choices):
      l = label.copy() if mask_label_choice else label
      if mask_label_choice:
        l[label_mask == 0] = ignore_index

      # tinygrad
      X = Tensor(x)
      out = X.cross_entropy(
        Tensor(l), 
        weight=Tensor(weights) if apply_weight_choice else None, 
        reduction=reduction_choice,
      )
      
      # torch
      X_torch = torch.tensor(x)
      X_torch = torch.nn.functional.cross_entropy(
        X_torch, 
        torch.tensor(l),
        weight=torch.tensor(weights) if apply_weight_choice else None,
        reduction=reduction_choice,
      )

      # cmp
      np.testing.assert_allclose(out.cpu().numpy(), X_torch.cpu().numpy(), atol=1e-6, rtol=1e-6)

  #@unittest.skipUnless(Device.DEFAULT == Device.CPU, "float64 not supported on GPU")
  @unittest.skip("float64 support broken")
  def test_jacobian(self):
    W = np.random.RandomState(1337).random((10, 5))
    x = np.random.RandomState(7331).random((1, 10)) - 0.5

    torch_x = torch.tensor(x, requires_grad=True)
    torch_W = torch.tensor(W, requires_grad=True)
    torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)
    PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()

    tiny_x = Tensor(x)
    tiny_W = Tensor(W)
    tiny_func = lambda x: x.dot(tiny_W).relu().log_softmax()
    J = jacobian(tiny_func, tiny_x)
    NJ = numerical_jacobian(tiny_func, tiny_x)

    np.testing.assert_allclose(PJ, J, atol = 1e-5)
    np.testing.assert_allclose(PJ, NJ, atol = 1e-5)

  #@unittest.skipUnless(Device.DEFAULT == Device.CPU, "float64 not supported on GPU")
  @unittest.skip("float64 support broken")
  def test_gradcheck(self):
    W = np.random.RandomState(1337).random((10, 5))
    x = np.random.RandomState(7331).random((1, 10)) - 0.5

    tiny_x = Tensor(x)
    tiny_W = Tensor(W)
    tiny_func = lambda x: x.dot(tiny_W).relu().log_softmax()

    self.assertTrue(gradcheck(tiny_func, tiny_x))

    # coarse approx. since a "big" eps and the non-linearities of the model
    self.assertFalse(gradcheck(tiny_func, tiny_x, eps = 0.1))

  def test_random_fns_are_deterministic_with_seed(self):
    for random_fn in [Tensor.randn, Tensor.uniform, Tensor.scaled_uniform, Tensor.glorot_uniform]:
      with self.subTest(msg=f"Tensor.{random_fn.__name__}"):
        Tensor.manual_seed(1337)
        a = random_fn(10,10).realize()
        Tensor.manual_seed(1337)
        b = random_fn(10,10).realize()
        np.testing.assert_allclose(a.numpy(), b.numpy())

  def test_zeros_like_has_same_dtype(self):
    for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
      a = Tensor([1, 2, 3], dtype=datatype)
      b = Tensor.zeros_like(a)
      assert a.dtype == b.dtype, f"a.dtype and b.dtype should be {datatype}"
      assert a.shape == b.shape, f"shape mismatch (Tensor.zeros_like){a.shape} != (torch){b.shape}"

    a = Tensor([1, 2, 3])
    b = Tensor.zeros_like(a, dtype=dtypes.int8)
    assert a.dtype != b.dtype and a.dtype == dtypes.float32 and b.dtype == dtypes.int8, "a.dtype should be float and b.dtype should be char"
    assert a.shape == b.shape, f"shape mismatch (Tensor.zeros_like){a.shape} != (torch){b.shape}"

  def test_ones_like_has_same_dtype_and_shape(self):
    for datatype in [dtypes.float16, dtypes.float32, dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8]:
      a = Tensor([1, 2, 3], dtype=datatype)
      b = Tensor.ones_like(a)
      assert a.dtype == b.dtype, f"a.dtype and b.dtype should be {datatype}"
      assert a.shape == b.shape, f"shape mismatch (Tensor.ones_like){a.shape} != (torch){b.shape}"

    a = Tensor([1, 2, 3])
    b = Tensor.ones_like(a, dtype=dtypes.int8)
    assert a.dtype != b.dtype and a.dtype == dtypes.float32 and b.dtype == dtypes.int8, "a.dtype should be float and b.dtype should be char"
    assert a.shape == b.shape, f"shape mismatch (Tensor.ones_like){a.shape} != (torch){b.shape}"

  def test_ndim(self):
    assert Tensor.randn(1).ndim == 1
    assert Tensor.randn(2,2,2).ndim == 3
    assert Tensor.randn(1,1,1,1,1,1).ndim == 6

  def test_numel(self):
    assert Tensor.randn(10, 10).numel() == 100
    assert Tensor.randn(1,2,5).numel() == 10
    assert Tensor.randn(1,1,1,1,1,1).numel() == 1
    assert Tensor([]).numel() == 0
    # assert Tensor.randn(1,0,2,5) == 0 # TODO: fix empty tensors

  def test_element_size(self):
    for f in dataclasses.fields(dtypes):
      dtype = f.default
      assert dtype.itemsize == Tensor.randn(3, dtype=dtype).element_size(), f"Tensor.element_size() not matching Tensor.dtype.itemsize for {dtype}"

if __name__ == '__main__':
  unittest.main()
