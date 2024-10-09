import os, math, time
os.environ["TRACEMETA"] = "0"
import numpy as np
from tinygrad import Tensor, nn, fetch, Device, TinyJit, GlobalCounters
from dataclasses import dataclass
from typing import List
import pathlib
from tinygrad.multi import MultiLazyBuffer
import tiktoken, argparse

SHARD = os.getenv("SHARD", True)

n_embed = 768
n_head = 12
block_size = 1024

class CausalSelfAttention:
  def __init__(self):
    # key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(n_embed, 3 * n_embed)
    # output projection
    self.c_proj = nn.Linear(n_embed, n_embed)
    # regularization
    self.n_head = n_head
    self.n_embed = n_embed
    # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
    self.bias = Tensor.ones(1, 1, block_size, block_size).tril()
    self.bias.requires_grad = False

  def __call__(self, x:Tensor):
    B, T, C = x.shape
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embed, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    att = att.softmax()
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).view(B, T, C) # re-assemble all head outputs side by side
    # output projection
    y = self.c_proj(y)
    return y

class Block:
  def __init__(self):
    self.attn = CausalSelfAttention()
    pass
  def __call__(self, x):
    return self.attn(x)
class GPT:
  def __init__(self):
    self.wte = nn.Embedding(50304, n_embed)
    self.wpe = nn.Embedding(block_size, n_embed)
    self.linear = nn.Linear(768, 50257)
    self.b0 = Block()

  def __call__(self, idx, targets):
    b, t = idx.shape
    tok_emb = self.wte(idx)
    pos = Tensor.arange(0, t)
    if SHARD:
      pos.shard_(GPUS, axis=0)
    pos_emb = self.wpe(pos)
    x = tok_emb + pos_emb
    x = self.b0(x)
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
      x = x.realize()
      y = y.realize()
      x.shard_(GPUS)
      y.shard_(GPUS)
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

optimizer = nn.optim.Adam(nn.state.get_parameters(model), shard_axis=0)

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
