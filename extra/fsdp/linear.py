import os
os.environ["TRACEMETA"] = "0"
from tinygrad import Tensor
from tinygrad.device import Device
import tinygrad.nn as nn
import math
from dataclasses import dataclass
from extra.fsdp.mlb import print_lb


GPUS = [f"{Device.DEFAULT}:{i}" for i in range(4)]
print(GPUS)

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.empty(out_features, in_features)
    self.bias = Tensor.empty(out_features) if bias else None

  def __call__(self, x:Tensor):
    return x.linear(self.weight.transpose(), self.bias)


class Optimizer:
  def __init__(self, params):
    for p in params:
      if p.requires_grad is None:
        p.requires_grad = True
    self.params = [x for x in params if x.requires_grad]

  def zero_grad(self):
    for x in self.params:
      x.grad = None

  def step(self):
    for x in self.params:
      x.assign(x.detach() - x.grad.all_gather_())
    Tensor.realize(*self.params)

class Model:
  def __init__(self):
    self.lin = Linear(8, 12, bias=False)

  def __call__(self, x):
    return self.lin(x)

x = Tensor.empty(2, 8)
model = Model()

x.shard_(GPUS)
for p in nn.state.get_parameters(model):
  p.shard_(GPUS, axis=0)
  p.realize()

opt = Optimizer(nn.state.get_parameters(model))

def train():
  y = model(x)
  loss = y.sum(0).sum(0)
  loss.backward()
  opt.step()
  opt.zero_grad()

with Tensor.train():
  for i in range(3):
    train()
