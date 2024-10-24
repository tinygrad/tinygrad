import os
os.environ["TRACEMETA"] = "0"
from pathlib import Path
from typing import List
import json, argparse, random, time
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters, TinyJit
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm, prod
from extra.fsdp.utils import print_size
import numpy as np
import math
import re
from tinygrad.multi import MultiLazyBuffer

SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
GPU_NAME = Device.DEFAULT
# if len(GPUS) > 1:
#   Device.DEFAULT = "PYTHON"

def shard_model(model, opt):
  seen = set()
  for k, p in nn.state.get_state_dict(model).items():
    if p in seen: continue
    seen.add(p)
    axis = 0
    if p.shape[0] == 1:
      axis = None
    p.shard_(GPUS, axis)
  for k, p in nn.state.get_state_dict(opt).items():
    if p in seen: continue
    seen.add(p)
    p.shard_(GPUS, axis=None if prod(p.shape) <= 1 else 0)
  for p in seen:
    p.realize()


MODEL_PARAMS = {
  "sm": {
    "args": {"dim": 24, "n_heads": 1, "n_kv_heads": 1, "n_layers": 1, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 48},
    "files": 1
  },
  "8B": {
    "args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336},
    "files": 1
  },
  "70B": {
    "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256,  "hidden_dim": 28672},
    "files": 8
  }
}

model_size = "sm"
linear = nn.Linear
model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True)
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-4, weight_decay=0)
print_size("model", *nn.state.get_parameters(model))
print_size("adamW", *nn.state.get_parameters(opt))
if len(GPUS) > 1:
  shard_model(model, opt)

for k, p in nn.state.get_state_dict(model).items():
  print(k, p.shape, f"Axis {p.lazydata.axis}" if isinstance(p.lazydata, MultiLazyBuffer) else "")

for k, p in nn.state.get_state_dict(opt).items():
  print(f"{k=}")
class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [
      f"<|reserved_special_token_{i}|>"
      for i in range(5, 256 - 5)
    ]
    self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

    self.model = tiktoken.Encoding(name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens)
  
  @property
  def bos_id(self): return self.special_tokens["<|begin_of_text|>"]
  @property
  def stop_tokens(self): return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

  def decode(self, toks): return self.model.decode([t for t in toks if t < self.num_base_tokens])
  def encode(self, text, allow_special=False):
    return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())


tokenizer = Tokenizer("tmp/tokenizer.model")
def encode_role(role: str):
  return [tokenizer.special_tokens["<|start_header_id|>"]] + tokenizer.encode(role) + [tokenizer.special_tokens["<|end_header_id|>"]] + tokenizer.encode("\n\n")
def encode_message(role: str, content: str):
  return encode_role(role) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]

def tokenize_data():
  with open("tmp/tiny_shakespeare.txt") as f:
    text = f.read()
    length = len(text)
    split = math.floor(length * 0.8)
    train = tokenizer.encode(text[:split])
    val = tokenizer.encode(text[split:])
    return Tensor(train), Tensor(val)

train, val = tokenize_data()
tokens = val
print(f"{tokens.shape=}")
B = 4
T = 16
# lightweight dataloader
def get_batch():
  i = 0
  while True:
    x = tokens[i:i+B*T].view(B, T)
    y = tokens[i+1:i+B*T+1].view(B, T)
    if len(GPUS) > 1:
      x.shard_(GPUS)
      y.shard_(GPUS)
    yield x, y
    i += B*T
    if i + B*T + 1 >= len(tokens):
      i = 0 # in prod we'd want to randomize the start point a bit

data_iter = iter(get_batch())
x, y = next(data_iter)

@TinyJit
def step(x, y):
  print(f"{x.shape=} {y.shape=}")
  loss = model(x, 0, target=y)
  opt.zero_grad()
  loss.backward()
  return loss.realize(*opt.schedule_step())

with Tensor.train():
  Device.DEFAULT = GPU_NAME
  for i in range(2):
  
    GlobalCounters.reset()
    t0 = time.time()
    loss = step(x.contiguous(), y.contiguous())
    Device[Device.DEFAULT].synchronize()
    t1 = time.time()
    print(f"iteration {i}, loss: {loss.item():.6f}, time: {(t1-t0)*1000:.3f}ms, {int(B*T/(t1-t0))} tok/s")
