import os
os.environ["TRACEMETA"] = "0"
from typing import List
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters, TinyJit
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm, prod
import numpy as np
import math
import re
from tinygrad.multi import MultiLazyBuffer

def get_size(tensors: List[Tensor]): return sum([t.nbytes() if isinstance(t, Tensor) else t.size for t in tensors])

def print_size(name, *tensors: Tensor):
    size = get_size(tensors)
    for unit in ['bytes', 'KB', 'MB', 'GB']:
        if size < 1000 or unit == 'GB': break
        size /= 1000
    print(f'{name} size: {size:.2f} {unit}')

Tensor.manual_seed(2)
SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
print(GPUS)
GPU_NAME = Device.DEFAULT
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
    axis = None if prod(p.shape) <= 1 else 0
    if p.shape[0] == 1:
      axis = None
    p.shard_(GPUS, axis)

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
    ret = self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())
    return ret


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
tokens = train
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

vocab_size = 128256
dim = 4
n_heads = 1

class Attention:
  def __init__(self):
    self.wq = nn.Linear(dim, dim)
    self.wk = nn.Linear(dim, dim)
    self.wv = nn.Linear(dim, dim)
  
  def __call__(self, x: Tensor):
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    attn = xq.scaled_dot_product_attention(xk, xv)
    return attn

class Model:
  def __init__(self):
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.attention = Attention()
    self.out = nn.Linear(dim, vocab_size, bias=False)

  def __call__(self, x: Tensor, target: Tensor = None):
    x = self.tok_embeddings(x)
    x = self.attention(x)
    x = self.out(x)

    if target is not None:
      loss = x.sparse_categorical_crossentropy(target)
      return x, loss
    return x, None
  
  def generate(self):
    tokens = Tensor([tokenizer.encode("<|begin_of_text|>", allow_special=True)])
    if len(GPUS) > 1:
      tokens.shard_(GPUS)
    for _ in range(10):
      logits, _ = self(tokens)
      logits = logits[:, -1, :]
      idx_next = logits.softmax().multinomial()
      tokens = tokens.cat(idx_next, dim=1)
    print(tokenizer.decode(tokens.tolist()[0]))

model = Model()
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=0.1)
print_size("model", *nn.state.get_parameters(model))
print_size("opt", *nn.state.get_parameters(opt))
x, y = next(data_iter)
if len(GPUS) > 1:
  shard_model(model, opt)


def step():
  opt.zero_grad()
  logits, loss = model(x, y)
  loss.backward()
  loss.realize(*opt.schedule_step())
  print(f"Loss {loss.tolist()}")

Device.DEFAULT = GPU_NAME
with Tensor.train():
  for i in range(3): step()
# model.generate()


def size_unit(size: str):
  for unit in ['bytes', 'KB', 'MB', 'GB']:
    if size < 1000 or unit == 'GB': break
    size /= 1000
  return float(size), unit

for device in GPUS:
  device = Device[device].allocator
  highest, unit = size_unit(device.mem_high)
  print(f"{device.name} highest: {highest:.2f} {unit}")