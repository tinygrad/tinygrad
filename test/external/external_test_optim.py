#!/usr/bin/env python
import unittest
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import LAMB

np.random.seed(1337)
x_init = np.random.randn(1,4).astype(np.float32)
W_init = np.random.randn(4,4).astype(np.float32)
m_init = np.random.randn(1,4).astype(np.float32)

class TinyNet:
  def __init__(self):
    self.x = Tensor(x_init.copy(), requires_grad=True)
    self.W = Tensor(W_init.copy(), requires_grad=True)
    self.m = Tensor(m_init.copy())

  def forward(self):
    out = self.x.matmul(self.W).relu()
    out = out.log_softmax(1)
    out = out.mul(self.m).add(self.m).sum()
    return out

class TinyNetTF:
  def __init__(self):
    self.x = tf.Variable(x_init.copy(), trainable=True)
    self.W = tf.Variable(W_init.copy(), trainable=True)
    self.m = tf.constant(m_init.copy())

  def forward(self):
    out = tf.matmul(self.x, self.W)
    out = tf.nn.relu(out)
    out = tf.nn.log_softmax(out, axis=1)
    out = tf.multiply(out, self.m) + self.m
    out = tf.reduce_sum(out)
    return out

def step(optim, steps=1, kwargs={}):
  net = TinyNet()
  optim = optim([net.x, net.W], **kwargs)
  for _ in range(steps):
    out = net.forward()
    optim.zero_grad()
    out.backward()
    optim.step()
  return net.x.detach().numpy(), net.W.detach().numpy()

def step_tf(optim, steps=1, kwargs={}):
  net = TinyNetTF()
  optim = optim(**kwargs)
  for _ in range(steps):
    with tf.GradientTape() as tape:
      out = net.forward()
    grads = tape.gradient(out, [net.x, net.W])
    optim.apply_gradients(zip(grads, [net.x, net.W]))
  return net.x.numpy(), net.W.numpy()

class ExternalTestOptim(unittest.TestCase):
  def _test_optim(self, tinygrad_optim, tensorflow_optim, steps, opts, atol, rtol):
    for x,y in zip(step(tinygrad_optim, steps, kwargs=opts),
                   step_tf(tensorflow_optim, steps, kwargs=opts)):
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

  def _test_lamb(self, steps, opts, atol, rtol): self._test_optim(LAMB, tfa.optimizers.LAMB, steps, opts, atol, rtol)

  def test_lamb(self): self._test_lamb(1, {'lr': 0.001}, 1e-5, 0)
  def test_lamb_high_lr(self): self._test_lamb(1, {'lr': 10}, 1e-5, 1e-5)

  def test_multistep_lamb(self): self._test_lamb(10, {'lr': 0.001}, 1e-5, 0)
  def test_multistep_lamb_high_lr(self): self._test_lamb(10, {'lr': 10}, 1e-5, 3e-4)

if __name__ == '__main__':
  unittest.main()
