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

SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
def reset_mem_high():
  for gpu in GPUS:
    Device[gpu].allocator.reset_mem_high()


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
        and t.grad.lazydata.axis is not None and t.grad.lazydata.axis != t.lazydata.axis:
        if t.lazydata.axis is None:
          print("gather")
          t.grad.gather_()
        else:
          print("reshard")
          t.grad.reshard_(t.lazydata.axis)
      t.assign(t.detach() - t.grad)
    Tensor.realize(*self.params)

def shape(x):
  print(f"{x.shape=}")
  return x
class Model:
  def __init__(self):
    self.layers = [
      nn.Conv2d(1, 32, 3),
      nn.Conv2d(32, 64, 3),
      lambda x: x.flatten(1)
    ]

  def __call__(self, x):
    return x.sequential(self.layers)

x = Tensor.empty(4, 1, 28, 28)
x.realize()
model = Model()
opt = Optimizer(nn.state.get_parameters(model))
print(list(nn.state.get_parameters(opt)))
print_size("model", *nn.state.get_parameters(model))
print_size("model with optimizer", *nn.state.get_parameters(opt))

SHARD = int(os.environ.get("SHARD", 0))
if SHARD > 1:
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
  loss.realize()
  loss.backward()
  opt.step()
  opt.zero_grad()

with Tensor.train():
  reset_mem_high()
  for i in range(1):
    train()
