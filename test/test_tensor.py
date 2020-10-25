import numpy as np
import torch
import unittest
from tinygrad.tensor import Tensor
from tinygrad.gradcheck import numerical_jacobian, jacobian, gradcheck

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTinygrad(unittest.TestCase):
  def test_backward_pass(self):
    def test_tinygrad():
      x = Tensor(x_init)
      W = Tensor(W_init)
      m = Tensor(m_init)
      out = x.dot(W).relu()
      out = out.logsoftmax()
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.data, x.grad, W.grad

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

  def test_jacobian(self):
    W = np.random.RandomState(1337).random((10, 5))
    x = np.random.RandomState(7331).random((1, 10)) - 0.5

    torch_x = torch.tensor(x, requires_grad=True)
    torch_W = torch.tensor(W, requires_grad=True)
    torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)
    PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()

    tiny_x = Tensor(x)
    tiny_W = Tensor(W)
    tiny_func = lambda x: x.dot(tiny_W).relu().logsoftmax()
    J = jacobian(tiny_func, tiny_x)
    NJ = numerical_jacobian(tiny_func, tiny_x)

    np.testing.assert_allclose(PJ, J, atol = 1e-5)
    np.testing.assert_allclose(PJ, NJ, atol = 1e-5)

  def test_gradcheck(self):
    W = np.random.RandomState(1337).random((10, 5))
    x = np.random.RandomState(7331).random((1, 10)) - 0.5

    tiny_x = Tensor(x)
    tiny_W = Tensor(W)
    tiny_func = lambda x: x.dot(tiny_W).relu().logsoftmax()

    self.assertTrue(gradcheck(tiny_func, tiny_x))

    # coarse approx. since a "big" eps and the non-linearities of the model
    self.assertFalse(gradcheck(tiny_func, tiny_x, eps = 0.1))

class TestOps(unittest.TestCase):
  def test_conv2d(self):
    for cin in [1,2,3]:
      for H in [2,3,5]:
        for W in [2,3,5]:
          x = torch.randn((5,cin,10,7), requires_grad=True)
          w = torch.randn((4,cin,H,W), requires_grad=True)
          xt = Tensor(x.detach().numpy())
          wt = Tensor(w.detach().numpy())

          out = torch.nn.functional.conv2d(x,w)
          ret = Tensor.conv2d(xt, wt)
          # TODO: why so inaccurate?
          np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=1e-5)

          out.relu().mean().backward()
          ret.relu().mean().backward()

          np.testing.assert_allclose(w.grad, wt.grad, atol=1e-7)
          np.testing.assert_allclose(x.grad, xt.grad, atol=1e-7)

  def test_maxpool2x2(self):
    x = torch.randn((5,2,10,8), requires_grad=True)
    xt = Tensor(x.detach().numpy())

    # in tinygrad
    ret = xt.maxpool2x2()
    assert ret.shape == (5,2,10//2,8//2)
    ret.mean().backward()

    # in torch
    out = torch.nn.MaxPool2d((2,2))(x)
    out.mean().backward()

    # forward and backward the same
    np.testing.assert_allclose(ret.data, out.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(x.grad, xt.grad, atol=1e-5)

if __name__ == '__main__':
  unittest.main()

