import os
os.environ["TRACEMETA"] = "0"
from tinygrad import Tensor
from tinygrad.device import Device
import tinygrad.nn as nn
import math
from dataclasses import dataclass
from extra.fsdp.utils import print_size
from tinygrad.multi import MultiLazyBuffer


GPUS = [f"{Device.DEFAULT}:{i}" for i in range(2)]

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
      if isinstance(x.grad.lazydata, MultiLazyBuffer) and (axis:=x.grad.lazydata.axis) is not None and axis != x.lazydata.axis:
        print("ALL GATHER")
        x.grad.all_gather_()
      x.assign(x.detach() - x.grad)
    Tensor.realize(*self.params)

class Model:
  def __init__(self):
    self.layers = [
      Linear(8, 12, bias=False),
      Linear(12, 10, bias=False),
    ]

  def __call__(self, x):
    x = self.layers[0](x)
    x = self.layers[1](x)
    return x

x = Tensor.empty(2, 8)
model = Model()
opt = Optimizer(nn.state.get_parameters(model))
print_size("model", *nn.state.get_parameters(model))
print_size("model with optimizer", *nn.state.get_parameters(opt))

if os.environ.get("SHARD"):
  print("SHARDING ON", GPUS)
  x.shard_(GPUS)
  for p in nn.state.get_parameters(opt):
    p.shard_(GPUS, axis=0)
    p.realize()
else:
  for p in nn.state.get_parameters(opt):
    p.realize()

def train():
  y = model(x)
  print(f"{y.shape=}")
  loss = y.sum(0).sum(0)
  loss.backward()
  opt.step()
  opt.zero_grad()

with Tensor.train():
  for i in range(1):
    train()
