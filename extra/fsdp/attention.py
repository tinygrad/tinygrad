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

n_embd = 48
block_size = 12
n_head = 2
class Model:
  def __init__(self):
    self.q = Linear(n_embd, n_embd)
    self.k = Linear(n_embd, n_embd)
    self.v = Linear(n_embd, n_embd)
    self.bias = Tensor.ones(1, block_size, block_size).tril()
    self.bias.requires_grad = False

  def __call__(self, x):
    B, T, C = x.shape
    k = self.k(x)
    q = self.q(x)
    v = self.v(x)
    k = k.view(B, T, 1, C).transpose(1, 2)
    q = q.view(B, T, 1, C).transpose(1, 2)
    v = v.view(B, T, 1, C).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) # B C T @ B 
    mask = self.bias[:, :T, :T]
    mask_zero = mask == 0
    att = att.masked_fill(mask_zero, float('-inf'))
    att = att.softmax()
    y = att @v
    y = y.transpose(1, 2).view(B, T, C)
    return y
  
SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
def reset_mem_high():
  for gpu in GPUS:
    Device[gpu].allocator.reset_mem_high()

def shard_model(model, opt):
  seen = set()
  for k, p in nn.state.get_state_dict(model).items():
    print(f"{k=}")
    if p in seen: continue
    seen.add(p)
    axis = 0
    print(f"{k=}")
    p.shard_(GPUS, axis)
  for k, p in nn.state.get_state_dict(model).items():
    if p in seen: continue
    seen.add(p)
    p.shard_(GPUS, axis=None if prod(p.shape) <= 1 else 0)
  for p in seen:
    p.realize()

model = Model()
opt = Optimizer(nn.state.get_parameters(model))
x = Tensor.empty(4, 3, n_embd)

if SHARD > 1:
  print("SHARDING ON", GPUS)
  x.shard_(GPUS)
  shard_model(model, opt)
else:
  print("NO SHARD")
  for p in nn.state.get_parameters(opt):
    p.realize()

print_size("model", *nn.state.get_parameters(model))
print_size("model with optimizer", *nn.state.get_parameters(opt))

reset_mem_high()
def train():
  y = model(x)
  loss = y.sum([0,1,2])
  loss.backward()
  opt.step()
  opt.zero()

for i in range(10):
  train()
