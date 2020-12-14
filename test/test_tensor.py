import numpy as np
import torch
import unittest
from tinygrad.tensor import Tensor, GPU
from extra.gradcheck import numerical_jacobian, jacobian, gradcheck

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestTinygrad(unittest.TestCase):
  gpu = False

  def setUp(self):
    self.data = np.random.RandomState(1337).random(100).astype(np.float32)
    self.W = np.random.RandomState(1337).random((10, 5))
    self.x = np.random.RandomState(7331).random((1, 10)) - 0.5
    self._idx_end = np.random.RandomState(1330).randint(0,99, size=(1))[0]
    self._idx_start = np.random.RandomState(1330).randint(0,self._idx_end, size=(1))[0]

  def test_backward_pass(self):

    def test_tinygrad():
      x = Tensor(x_init, gpu=self.gpu)
      W = Tensor(W_init, gpu=self.gpu)
      m = Tensor(m_init, gpu=self.gpu)
      out = x.dot(W).relu()
      out = out.logsoftmax()
      out = out.mul(m).add(m).sum()
      out.backward()
      return out.cpu().data, x.grad.cpu().data, W.grad.cpu().data

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
    torch_x = torch.tensor(self.x, requires_grad=True)
    torch_W = torch.tensor(self.W, requires_grad=True)
    torch_func = lambda x: torch.nn.functional.log_softmax(x.matmul(torch_W).relu(), dim=1)
    PJ = torch.autograd.functional.jacobian(torch_func, torch_x).squeeze().numpy()

    tiny_x = Tensor(self.x, gpu=self.gpu)
    tiny_W = Tensor(self.W, gpu=self.gpu)
    tiny_func = lambda x: x.dot(tiny_W).relu().logsoftmax()
    J = jacobian(tiny_func, tiny_x)
    NJ = numerical_jacobian(tiny_func, tiny_x)

    np.testing.assert_allclose(PJ, J, atol = 1e-5)
    np.testing.assert_allclose(PJ, NJ, atol = 1e-5)

  def test_gradcheck(self):
    tiny_x = Tensor(self.x, gpu=self.gpu)
    tiny_W = Tensor(self.W, gpu=self.gpu)
    tiny_func = lambda x: x.dot(tiny_W).relu().logsoftmax()

    self.assertTrue(gradcheck(tiny_func, tiny_x))

    # coarse approx. since a "big" eps and the non-linearities of the model
    self.assertFalse(gradcheck(tiny_func, tiny_x, eps = 0.1))

  def test_multi_indexing(self):
    shape = np.random.RandomState(1813).randint(1,5, size=5)
    data = np.random.RandomState(1330).random(shape)
    x = Tensor(data ,gpu=self.gpu)
    idx = tuple([np.random.RandomState(1399).randint(0, ix, size=1)[0] for ix in shape])
    self.assertTrue(x[idx].cpu().data, data[idx])

  def test_slicing(self):
    x = Tensor(self.data ,gpu=self.gpu)
    self.assertListEqual(x[self._idx_start:self._idx_end].cpu().data.tolist(), self.data[self._idx_start:self._idx_end].tolist())

  def test_single_index(self):
    t = Tensor(self.data, gpu=self.gpu)
    self.assertTrue(t[int(self._idx_start)].cpu().data, self.data[self._idx_start])

  def test_invalid_indices(self):
    invalid_indices = ['invalid', self, [slice(2,4), slice(9,10)], [(1,2)], range(1), [1,2,3]]
    t = Tensor(self.data , gpu=self.gpu)
    with self.assertRaises(IndexError):
      [t[idx] for idx in invalid_indices]

  def tearDown(self):
    if self.gpu:
      pass
      # may be release gpu buffer once testing is done?

@unittest.skipUnless(GPU, "Requires GPU")
class TestTinygradGPU(TestTinygrad):
  gpu = True

  @unittest.skip("float64 not supported on GPU")
  def test_jacobian(self): pass

  @unittest.skip("float64 not supported on GPU")
  def test_gradcheck(self): pass


if __name__ == '__main__':
  unittest.main()
