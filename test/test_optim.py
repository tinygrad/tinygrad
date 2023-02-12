import numpy as np
import torch
import unittest
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam, SGD, RMSprop
from extra.utils import get_parameters

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

def walk_tinygrad(optim, steps, kwargs={}):
  net = TinyNet()
  optim = optim([net.x, net.W], **kwargs)
  for _ in range(steps):
    out = net.forward()
    out.data
    out.backward()
    optim.step()
  return net.x.cpu().data, net.W.cpu().data

def walk_pytorch(optim, steps, kwargs={}):
  net = TorchNet()
  optim = optim([net.x, net.W], **kwargs)
  for _ in range(steps):
    out = net.forward()
    out.backward()
    optim.step()
  return net.x.detach().numpy(), net.W.detach().numpy()


class TinyNet():
  def __init__(self):
    self.x = Tensor(x_init.copy(), requires_grad=True)
    self.W = Tensor(W_init.copy(), requires_grad=True)
    self.m = Tensor(m_init.copy())

  def forward(self):
    out = self.x.dot(self.W).relu()
    out = out.logsoftmax()
    out = out.mul(self.m).add(self.m).sum()
    return out


class TorchNet():
  def __init__(self):
    self.x = torch.tensor(x_init.copy(), requires_grad=True)
    self.W = torch.tensor(W_init.copy(), requires_grad=True)
    self.m = torch.tensor(m_init.copy())

  def forward(self):
    out = self.x.matmul(self.W).relu()
    out = torch.nn.functional.log_softmax(out, dim=1)
    out = out.mul(self.m).add(self.m).sum()
    return out


class TestOptim(unittest.TestCase):

  global steps
  steps = 20

  def test_adam(self):
    for x,y in zip(walk_tinygrad(Adam, steps),
                   walk_pytorch(torch.optim.Adam, steps)):
      np.testing.assert_allclose(x, y, atol=1e-4)

  def test_sgd(self):
    for x,y in zip(walk_tinygrad(SGD, steps, kwargs={'lr': 0.001}),
                   walk_pytorch(torch.optim.SGD, steps, kwargs={'lr': 0.001})):
      np.testing.assert_allclose(x, y, atol=1e-5)
  
  def test_sgd_momentum(self):
    for x,y in zip(walk_tinygrad(SGD, steps, kwargs={'lr': 0.001, 'momentum': 0.9}),
                   walk_pytorch(torch.optim.SGD, steps, kwargs={'lr': 0.001, 'momentum': 0.9})):
      np.testing.assert_allclose(x, y, atol=1e-5)

  def test_rmsprop(self):
    for x,y in zip(walk_tinygrad(RMSprop, steps, kwargs={'lr': 0.001, 'decay': 0.99}),
                   walk_pytorch(torch.optim.RMSprop, steps,
                                kwargs={'lr': 0.001, 'alpha': 0.99})):
      np.testing.assert_allclose(x, y, atol=1e-5)

if __name__ == '__main__':
  unittest.main()
