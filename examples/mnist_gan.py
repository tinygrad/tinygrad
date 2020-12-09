#!/usr/bin/env python
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'test'))

from tinygrad.tensor import Tensor, Function, register
from tinygrad.utils import get_parameters
import tinygrad.optim as optim

GPU = os.getenv("GPU") is not None

class LeakyReLU(Function):
  @staticmethod
  def forward(ctx, input, leaky_slope=0.1):
    ctx.save_for_backward(input, leaky_slope)
    return np.maximum(input*leaky_slope, 0)

  @staticmethod
  def backward(ctx, grad_output):
    input,leaky_slope = ctx.saved_tensors
    grad_input = np.zeros_like(input)
    grad_input[input<=0.0] = leaky_slope
    grad_input[input>0.0] = grad_output
    return grad_input
register('leakyrelu', LeakyReLU)

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

  def forward(self, x):
    x = x.dot(self.l1).leakyrelu(0.2)
    x = x.dot(self.l2).leakyrelu(0.2)
    x = x.dot(self.l3).leakyrelu(0.2)
    x = x.dot(self.l4).sigmoid()
    return x
if __name__ == "__main__":
  generator = LinearGen()
  discriminator = LinearDisc()
  generator_params = get_parameters(generator)
  discriminator_params = get_parameters(discriminator)
  if GPU:
    [x.cuda_() for x in params]
  # optimizers
  optim_g = optim.Adam(generator_params, lr=0.001)
  optim_d = optim.Adam(discriminator_params, lr=0.001)
  import numpy as np
  x = generator.forward(Tensor(np.random.uniform(64, size=(10, 128))))
  breakpoint()
