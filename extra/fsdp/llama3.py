import os
os.environ["TRACEMETA"] = "0"
from typing import List, Tuple
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters
from tinygrad import Tensor, dtypes, nn, Context, Device, GlobalCounters, TinyJit
from tinygrad.helpers import Profiling, Timing, DEBUG, colored, fetch, tqdm, prod, trange
import numpy as np
import math
import re
from tinygrad.multi import MultiLazyBuffer
from extra.models.llama import sample, Transformer
from extra.fsdp.utils import get_size
Tensor.manual_seed(2)

SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
GPU_NAME = Device.DEFAULT
if len(GPUS) > 1:
  Device.DEFAULT = "CLANG"
B = 4
T = 16
vocab_size = 128256
dim = 64
n_layers = 1
n_heads = 32
n_kv_heads = 8
max_context = 8192
rope_theta=50000
hidden_dim = 32
epoch = 0
lr = 1e-4
weight_decay=0
generate_tokens = 5
assert dim % n_heads == 0 and dim % SHARD == 0
head_dim = dim // n_heads
shard_dim = dim // SHARD
norm_eps = 1e-5
TEMPERATURE = 0.95
TOP_K = 0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

model = Transformer(
  dim=dim,
  hidden_dim=hidden_dim,
  n_heads=n_heads,
  n_layers=n_layers,
  norm_eps=norm_eps,
  vocab_size=vocab_size,
  n_kv_heads=n_kv_heads,
  max_context=max_context,
  jit=False
)
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, weight_decay=weight_decay)
model_size, model_size_unit = get_size(nn.state.get_parameters(model))
opt_size, opt_size_unit = get_size(nn.state.get_parameters(opt))
print(f"Model {model_size:.4f} {model_size_unit} Opt: {opt_size:.4f} {opt_size_unit}")

def shard_model(model, opt):
  seen = set()
  for k, p in nn.state.get_state_dict(model).items():
    if p in seen: continue
    seen.add(p)
    axis = 0
    if k == 'tok_embeddings.weight':
      axis = 1
    elif k == 'freqs_cis':
      axis = None
    elif p.shape[0] == 1:
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

if len(GPUS) > 1:
  shard_model(model, opt)


def generate():
  opt.zero_grad()
  for p in nn.state.get_parameters(model):
    p.requires_grad = False
  tokens = Tensor([tokenizer.encode("<|begin_of_text|>", allow_special=True)])
  if len(GPUS) > 1:
    tokens.shard_(GPUS, axis=None)
  for start_pos in range(generate_tokens):
    idx_next = model(tokens, start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).unsqueeze(0)
    if len(GPUS) > 1:
      idx_next.shard_(GPUS, axis=None)
    tokens = tokens.cat(idx_next, dim=1)
  return tokenizer.decode(tokens.tolist()[0])

print(generate())