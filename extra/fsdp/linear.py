import os
os.environ["TRACEMETA"] = "0"
from tinygrad import Tensor, TinyJit
from tinygrad.device import Device
import tinygrad.nn as nn
import math
from dataclasses import dataclass
from extra.fsdp.utils import print_size, print_lb
from tinygrad.multi import MultiLazyBuffer
from tinygrad.helpers import prod



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
    for t in self.params:
      if isinstance(t.grad.lazydata, MultiLazyBuffer) and isinstance(t.lazydata, MultiLazyBuffer) \
        and t.lazydata.axis is not None and t.grad.lazydata.axis != t.lazydata.axis:
        if t.lazydata.axis is None:
          print("Gather")
          t.grad.gather_()
        else:
          print("Reshard")
          t.grad.reshard_(t.lazydata.axis)
      t.assign(t.detach() - t.grad)
    Tensor.realize(*self.params)

class Model:
  def __init__(self):
    self.layers = [
      nn.Linear(8, 12, bias=False),
      nn.Linear(12, 10, bias=False),
    ]

  def __call__(self, x):
    x = self.layers[0](x)
    x = self.layers[1](x)
    return x

x = Tensor.empty(2, 8)
model = Model()
opt = nn.optim.SGD(nn.state.get_parameters(model))
print_size("model", *nn.state.get_parameters(model))
print_size("model with optimizer", *nn.state.get_parameters(opt))

SHARD = int(os.environ.get("SHARD", 0))
if SHARD > 1:
  GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
  print("SHARDING ON", GPUS)
  x.shard_(GPUS)
  for k, p in nn.state.get_state_dict(opt).items():
    p.shard_(GPUS, axis=None if prod(p.shape) <= 1 else 0)
    p.realize()
else:
  print("NO SHARD")
  for p in nn.state.get_parameters(opt):
    p.realize()

# @TinyJit
def train():
  y = model(x)
  loss = y.sum(0).sum(0)
  loss.backward()
  opt.step()
  opt.zero_grad()

with Tensor.train():
  for i in range(10):
    train()
