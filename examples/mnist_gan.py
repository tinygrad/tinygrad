#!/usr/bin/env python
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

from tinygrad.tensor import Tensor, Function, register
from tinygrad.utils import get_parameters
import tinygrad.optim as optim

GPU = os.getenv("GPU") is not None

class Dropout(Function):
  @staticmethod
  def forward(ctx, input, p=0.5):
    _dead_ns = np.random.binomial(1, 1-p, size=input.shape)
    ctx.save_for_backward(input, _dead_ns)
    return input*_dead_ns

  @staticmethod
  def backward(ctx, grad_output):
    input, _dead_ns = ctx.saved_tensors
    grad_input = grad_output * _dead_ns
    return grad_input
register('dropout', Dropout)

class LinearGen:
  def __init__(self):
    lv = 128
    self.l1 = Tensor.uniform(128, 256)
    self.l2 = Tensor.uniform(256, 512)
    self.l3 = Tensor.uniform(512, 1024)
    self.l4 = Tensor.uniform(1024, 784)

  def forward(self, x):
    x = x.dot(self.l1).leakyrelu(0.2)
    x = x.dot(self.l2).leakyrelu(0.2)
    x = x.dot(self.l3).leakyrelu(0.2)
    x = x.dot(self.l4).tanh()
    x = x.reshape(shape=(-1,1,28,28))
    return x

class LinearDisc:
  def __init__(self):
    in_sh = 784
    self.l1 = Tensor.uniform(784, 1024)
    self.l2 = Tensor.uniform(1024, 512)
    self.l3 = Tensor.uniform(512, 256)
    self.l4 = Tensor.uniform(256, 1)

  def forward(self, x, train=False):
    x = x.dot(self.l1).leakyrelu(0.2)
    if train:
        x = x.dropout(0.3)
    x = x.dot(self.l2).leakyrelu(0.2)
    if train:
        x = x.dropout(0.3)
    x = x.dot(self.l3).leakyrelu(0.2)
    if train:
        x = x.dropout(0.3)
    x = x.dot(self.l4).sigmoid()
    return x
if __name__ == "__main__":
  generator = LinearGen()
  discriminator = LinearDisc()
  generator_params = get_parameters(generator)
  discriminator_params = get_parameters(discriminator)
  GPU = 0
  if GPU:
    [x.cuda_() for x in generator_params+discriminator_params]
  # optimizers
  optim_g = optim.Adam(generator_params, lr=0.001)
  optim_d = optim.Adam(discriminator_params, lr=0.001)
  import numpy as np
  x = discriminator.forward(Tensor(np.random.uniform(1, size=(784))), train=True)
  breakpoint()
