import os
os.environ["TRACEMETA"] = "0"
from tinygrad import Tensor
from tinygrad.device import Device
import tinygrad.nn as nn
import math
from dataclasses import dataclass
from extra.fsdp.utils import print_size, print_lb
from tinygrad.helpers import prod
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
    mask = self.bias[:, :3, :3]
    mask_zero = mask == 0
    att = att.masked_fill(mask_zero, float('-inf'))
    att = att.softmax()
    y = att @v
    y = y.transpose(0, 1).view(3, 6)
    return y

model = Model()
opt = Optimizer(nn.state.get_parameters(model))
x = Tensor.empty(3, 6)

print_size("model", *nn.state.get_parameters(model))
print_size("model with optimizer", *nn.state.get_parameters(opt))
SHARD = int(os.environ.get("SHARD"))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
if SHARD > 1:
  print("SHARDING ON", GPUS)
  x.shard_(GPUS)
  seen = set()
  for k, p in nn.state.get_state_dict(opt).items():
    if p in seen: continue
    seen.add(p)
    p.shard_(GPUS, axis=None if prod(p.shape) <= 1 else 0)
    p.realize()
  for k, p in nn.state.get_state_dict(model).items():
    if p in seen: continue
    seen.add(p)
    if k == "bias":
      p.shard_(GPUS)
    else:
      p.shard_(GPUS, axis=None if prod(p.shape) <= 1 else 0)
    p.realize()
else:
  print("NO SHARD")
  for p in nn.state.get_parameters(opt):
    p.realize()


def train():
  y = model(x)
  loss = y.sum(0).sum(0)
  loss.backward()
  opt.step()
  opt.zero()

for i in range(10):
  train()
