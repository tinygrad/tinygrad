import os, math, time
os.environ["TRACEMETA"] = "0"
import numpy as np
from tinygrad import Tensor, nn, fetch, Device, TinyJit, GlobalCounters
from dataclasses import dataclass
from typing import List
import pathlib
from tinygrad.multi import MultiLazyBuffer
import tiktoken, argparse

SHARD = os.getenv("SHARD", False)


class GPT:
  def __init__(self):
    self.linear = nn.Linear(768, 50257)

  def __call__(self, idx, targets):
    x = idx.reshape((4, 64, 1))
    x = x.expand((4, 64, 768))
    x = self.linear(x)
    loss = x.sparse_categorical_crossentropy(targets)
    return x, loss

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
tokens_bin = pathlib.Path("/root/tinygrad/tmp/tiny_shakespeare_val.bin")
assert os.path.isfile(tokens_bin)
print(f"loading cached tokens in {tokens_bin}")
with open(tokens_bin, "rb") as f:
  f.seek(0x400)
  tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
tokens = Tensor(tokens)

B, T = 4, 64
GPUS = [f'{Device.DEFAULT}:{i}' for i in range(2)]
def get_batch():
  assert B*T+1 <= len(tokens), "not enough tokens"
  # for 338,025 tokens. E.g. with B=8 T=1024, this will yield 41 batches before looping
  i = 0
  while True:
    x = tokens[i:i+B*T].view(B, T)
    y = tokens[i+1:i+B*T+1].view(B, T)
    if SHARD:
      x.shard_(GPUS, axis=0)
      y.shard_(GPUS, axis=0)
    yield x, y
    i += B*T
    if i + B*T + 1 >= len(tokens):
      i = 0 # in prod we'd want to randomize the start point a bit

data_iter = iter(get_batch())
x, y = next(data_iter)

model = GPT()
if SHARD:
  seen = set()
  for p in nn.state.get_state_dict(model).values():
    if p in seen: continue
    seen.add(p)
    p.shard_(GPUS, axis=0)

optimizer = nn.optim.AdamW(nn.state.get_parameters(model), shard_axis=0)

@TinyJit
def step(x, y):
  _, loss = model(x, y)
  optimizer.zero_grad()
  loss.backward()
  return loss.realize(*optimizer.schedule_step())

with Tensor.train():
  for i in range(3):
    GlobalCounters.reset()
    t0 = time.time()
    loss = step(x.contiguous(), y.contiguous())
    Device[Device.DEFAULT].synchronize()
    t1 = time.time()
    print(f"iteration {i}, loss: {loss.item():.6f}, time: {(t1-t0)*1000:.3f}ms, {int(B*T/(t1-t0))} tok/s")
