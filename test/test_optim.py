import numpy as np
import torch
import unittest
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam, SGD, RMSprop, get_parameters

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

def step(net, optim, kwargs={}):
    optim = optim([net.x, net.W], **kwargs)
    out = net.forward()
    out.backward()
    optim.step()
    return net.x.detach().numpy(), net.W.detach().numpy()

def test_steps(steps, tinyoptim, torchoptim, tinyoptim_args={}, torchoptim_args={}, atol=1e-4):
    tinynet = TinyNet()
    torchnet = TorchNet()
    for _ in range(steps):
        x1, w1 = step(tinynet, tinyoptim, tinyoptim_args)
        x2, w2 = step(torchnet, torchoptim, torchoptim_args)
        np.testing.assert_allclose(x1, x2, atol=atol)
        np.testing.assert_allclose(w1, w2, atol=atol)

class TinyNet():
  def __init__(self):
    self.x = Tensor(x_init.copy())
    self.W = Tensor(W_init.copy())
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
  STEPS = 10

  def test_adam(self):
    test_steps(self.STEPS, Adam, torch.optim.Adam)

  def test_sgd(self):
    optim_args = {'lr': 0.001}
    test_steps(self.STEPS, SGD, torch.optim.SGD, optim_args, optim_args, 1e-5)

  def test_sgd_momentum(self):
    optim_args = {'lr': 0.001, 'momentum': 0.9}
    test_steps(self.STEPS, SGD, torch.optim.SGD, optim_args, optim_args, 1e-5)

  def test_sgd_momentum_nesterov(self):
    optim_args = {'lr': 0.001, 'momentum': 0.9, "nesterov": True}
    test_steps(self.STEPS, SGD, torch.optim.SGD, optim_args, optim_args, 1e-5)

  def test_rmsprop(self):
    test_steps(self.STEPS, RMSprop, torch.optim.RMSprop, {'lr': 0.001, 'decay': 0.99}, {'lr': 0.001, 'alpha': 0.99}, 1e-5)

if __name__ == '__main__':
  unittest.main()
