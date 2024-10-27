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

SHARD = int(os.environ.get("SHARD", 1))
GPUS = [f"{Device.DEFAULT}:{i}" for i in range(SHARD)]
GPU_NAME = Device.DEFAULT
if len(GPUS) > 1:
  Device.DEFAULT = "CLANG"
B = 4
T = 16
vocab_size = 128256
dim = 4096
n_layers = 32
n_heads = 32
max_context = 8192
rope_theta=50000
hidden_dim = 14336
epoch = 30
lr = 1e-4
weight_decay=0
generate_tokens = 5
assert dim % n_heads == 0 and dim % SHARD == 0
s_head_dim = dim // n_heads
shard_dim = dim // SHARD
norm_eps = 1e-5


# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=dtypes.half) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  # TODO: move dtype outside this
  ret = Tensor.stack(freqs.cos().cast(dtype), freqs.sin().cast(dtype), dim=-1)
  ret = ret.reshape(1, end, 1, dim//2, 2)
  ret.requires_grad = False
  return ret

# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

def get_size(tensors: List[Tensor]):
  size = sum([t.nbytes() if isinstance(t, Tensor) else t.size for t in tensors])
  for unit in ['bytes', 'KB', 'MB', 'GB']:
    if size < 1000 or unit == 'GB': break
    size /= 1000
  return size, unit

Tensor.manual_seed(2)

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

def to_device_0(model):
  for k, p in nn.state.get_state_dict(model).items():
    p.to_(GPUS[0])

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


class Attention:
  def __init__(self):
    self.wq = nn.Linear(dim, dim, bias=False)
    self.wk = nn.Linear(dim, dim, bias=False)
    self.wv = nn.Linear(dim, dim, bias=False)
    self.wo = nn.Linear(dim, dim, bias=False)
  
  def __call__(self, x: Tensor, freqs_cis: Tensor, mask: Tensor):
    _B, _T, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk, xv = map(lambda w: w.reshape(_B, _T, n_heads, s_head_dim), [xq, xk, xv])
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    xq, xk, xv = map(lambda w: w.transpose(1, 2), [xq, xk, xv])
    attn = xq.scaled_dot_product_attention(xk, xv, mask)
    attn = attn.transpose(1,2).reshape(_B, _T, dim)
    attn = self.wo(attn)
    return attn

class FeedForward:
  def __init__(self):
    self.w1 = nn.Linear(dim, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, hidden_dim, bias=False) # the gate in Gated Linear Unit

  def __call__(self, x:Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x)) # SwiGLU [arxiv/2002.05202, eq (5)]

class TransformerBlock:
  def __init__(self):
    self.attention = Attention()
    self.feed_forward = FeedForward()
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)
  
  def __call__(self, x: Tensor, freqs_cis: Tensor, mask: Tensor):
    h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
    return (h + self.feed_forward(self.ffn_norm(h)))

class Transformer:
  def __init__(self):
    self.layers = [TransformerBlock() for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = nn.Embedding(vocab_size, dim)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(s_head_dim, max_context * 2, rope_theta).contiguous()

  def __call__(self, x: Tensor, target: Tensor = None):
    _B, _T = x.shape
    x = self.tok_embeddings(x)
    freqs_cis = self.freqs_cis.shrink((None, (0, _T),None,None,None))
    mask = Tensor.full((1, 1, _T, _T), float("-inf"), dtype=x.dtype, device=x.device).triu(1).realize()
    for layer in self.layers:
      x = layer(x, freqs_cis, mask)
    x = self.norm(x)
    x = self.output(x)
    if target is not None:
      loss = x.sparse_categorical_crossentropy(target)
      return x, loss
    return x, None
  
  def generate(self):
    to_device_0(model)
    tokens = Tensor([tokenizer.encode("<|begin_of_text|>", allow_special=True)])
    for _ in range(generate_tokens):
      logits, _ = self(tokens)
      logits = logits[:, -1, :]
      idx_next = logits.softmax().multinomial()
      tokens = tokens.cat(idx_next, dim=1)
    return tokenizer.decode(tokens.tolist()[0])

model = Transformer()
opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=lr, weight_decay=weight_decay)
model_size, model_size_unit = get_size(nn.state.get_parameters(model))
opt_size, opt_size_unit = get_size(nn.state.get_parameters(opt))
print(f"Model {model_size:.4f} {model_size_unit} Opt: {opt_size:.4f} {opt_size_unit}")

if len(GPUS) > 1:
  shard_model(model, opt)

train, val = tokenize_data()

# lightweight dataloader
def get_batch(tokens, batch):
  i = 0
  while True:
    x = tokens[i:i+batch*T].view(batch, T)
    y = tokens[i+1:i+batch*T+1].view(batch, T)
    if len(GPUS) > 1:
      x.shard_(GPUS)
      y.shard_(GPUS)
    yield x, y
    i += batch*T
    if i + batch*T + 1 >= len(tokens):
      i = 0 # in prod we'd want to randomize the start point a bit

train_data = iter(get_batch(train, B))
x_test, y_test = next(iter(get_batch(val, 32)))


@TinyJit
@Tensor.train()
def train_step(x: Tensor, y: Tensor):
  logits, loss = model(x, y)
  opt.zero_grad()
  loss.backward()
  loss.realize(*opt.schedule_step())
  return loss

@TinyJit
@Tensor.test()
def test_step() -> Tensor:
  _, loss = model(x_test, y_test)
  return loss


losses = []
Device.DEFAULT = GPU_NAME
test_loss = float('nan')
for i in (t:= trange(epoch)):
  x, y = next(train_data)
  loss = train_step(x.contiguous(), y.contiguous())
  if i % 10 == 9: test_loss = test_step().item()
  t.set_description(f"loss: {loss.item():6.2f} test_loss: {test_loss:5.2f}")


def size_unit(size: str):
  for unit in ['bytes', 'KB', 'MB', 'GB']:
    if size < 1000 or unit == 'GB': break
    size /= 1000
  return float(size), unit

mem_usage = []
for device in GPUS:
  device = Device[device].allocator
  highest, unit = size_unit(device.mem_high)
  mem_usage.append(f"{device.name}: {highest:.2f} {unit}")

print("Training peak mem", mem_usage)


text = model.generate()
print("Inference:", text)
