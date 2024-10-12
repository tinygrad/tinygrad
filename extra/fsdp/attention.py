import os
os.environ["TRACEMETA"] = "0"
from tinygrad import Tensor
from tinygrad.device import Device
import tinygrad.nn as nn
import math
from dataclasses import dataclass
from extra.mlb import print_lb
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(2)]
print(GPUS)
class Optimizer:
  def __init__(self, params):
    for p in params:
      if p.requires_grad is None:
        p.requires_grad = True
    self.params = [x for x in params if x.requires_grad]

  def zero(self):
    for x in self.params:
      x.grad = None

  def step(self):
    for x in self.params:
      x.assign(x.detach() - x.grad)
    Tensor.realize(*self.params)

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.empty(out_features, in_features)
    self.bias = Tensor.empty(out_features) if bias else None

  def __call__(self, x:Tensor):
    return x.linear(self.weight.transpose(), self.bias)

class Model:
  def __init__(self):
    self.q = Linear(6, 6)
    self.k = Linear(6, 6)
    self.v = Linear(6, 6)
    self.bias = Tensor.ones(1, 12, 12).tril()
    self.bias.requires_grad = False

  def __call__(self, x):
    k = self.k(x)
    q = self.q(x)
    v = self.v(x)
    k = k.view(3, 2, 3).transpose(0, 1)
    q = q.view(3, 2, 3).transpose(0, 1)
    v = v.view(3, 2, 3).transpose(0, 1)

    att = (q @ k.transpose(1, 2))
    att = att.masked_fill(self.bias[:, :3, :3] == 0, float('-inf'))
    att = att.softmax()
    y = att @v
    y = y.transpose(0, 1).view(3, 6)
    return y

model = Model()
x = Tensor.empty(3, 6)
x.shard_(GPUS)
for k, p in nn.state.get_state_dict(model).items():
  p.shard_(GPUS, axis=None if k == "bias" else 0)
  p.realize()

opt = Optimizer(nn.state.get_parameters(model))

def train():
  y = model(x)
  loss = y.sum(0).sum(0)
  loss.backward()
  opt.step()
  opt.zero()

for i in range(3):
  train()

y = model(x)