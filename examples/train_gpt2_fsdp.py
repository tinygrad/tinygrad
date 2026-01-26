#!/usr/bin/env/ python3
import contextlib, time
with contextlib.suppress(ImportError): import tiktoken
from tinygrad import Tensor, TinyJit, Device, GlobalCounters
from tinygrad.helpers import fetch, colored
from tinygrad.nn import Embedding, Linear, LayerNorm
from tinygrad.nn.state import get_state_dict, get_parameters
from tinygrad.nn.distributed import FSDP
from tinygrad.nn.optim import AdamW
import numpy as np


N_GPUS = 2
DEVICES = tuple(f"{Device.DEFAULT}:{i}" if i != 0 else f"{Device.DEFAULT}" for i in range(N_GPUS))

NUM_ITERS = 50
BATCH_SIZE = 8
SEQ_LEN = 256
DIM = 768
N_HEADS = 12
N_LAYERS = 12
VOCAB_SIZE = 50257

class Attention:
  def __init__(self, dim, n_heads):
    self.c_attn = Linear(dim, 3*dim)
    self.c_proj = Linear(dim, dim)
    self.n_heads, self.head_dim = n_heads, dim // n_heads

  def __call__(self, x, mask):
    xqkv = self.c_attn(x).reshape(None, None, 3, self.n_heads, self.head_dim)
    xq, xk, xv = [xqkv[:, :, i, :, :] for i in range(3)]
    return self.c_proj(xq.transpose(1, 2).scaled_dot_product_attention(xk.transpose(1, 2), xv.transpose(1, 2), mask).transpose(1, 2).flatten(2))

class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps):
    self.attn = Attention(dim, n_heads)
    self.c_fc, self.c_proj = Linear(dim, 4*dim), Linear(4*dim, dim)
    self.ln_1, self.ln_2 = LayerNorm(dim, norm_eps), LayerNorm(dim, norm_eps)

  def __call__(self, x, mask):
    x = x + self.attn(self.ln_1(x), mask)
    return x + self.c_proj(self.c_fc(self.ln_2(x)).gelu())

class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
    self.wte, self.wpe = Embedding(vocab_size, dim), Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
    self.ln_f = LayerNorm(dim, norm_eps)
    self.lm_head = Linear(dim, vocab_size, bias=False)

    for name, p in get_state_dict(self).items():
      p.requires_grad = True
      if "weight" in name and len(p.shape) >= 2:
        p.assign(Tensor.normal(p.shape, mean=0, std=0.02, device=p.device))
      elif "bias" in name:
        p.assign(Tensor.zeros(p.shape, device=p.device))

    self.mask = Tensor.full((max_seq_len, max_seq_len), float("-inf"), device=DEVICES[0], requires_grad=False).triu(1).realize()

  def __call__(self, tokens):
    bsz, seqlen = tokens.shape
    mask = self.mask[:seqlen, :seqlen].to(tokens.device[0]).shard(tokens.device, axis=None)
    pos = Tensor.arange(seqlen, device=tokens.device[0]).reshape(1, seqlen).shard(tokens.device, axis=None)

    x = self.wte(tokens) + self.wpe(pos)
    for block in self.h: x = block(x, mask)
    return self.lm_head(self.ln_f(x))

def load_data():
  try:
    fn = fetch("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    text = open(fn, "r").read()
    return np.array(tiktoken.get_encoding("gpt2").encode(text), dtype=np.int32)
  except: return np.random.randint(0, VOCAB_SIZE, (20000,), dtype=np.int32)

def get_batch(data_np, bs, seq_len, devices):
  ix = np.random.randint(0, len(data_np) - seq_len - 1, (bs,))
  x = np.stack([data_np[i:i+seq_len] for i in ix])
  y = np.stack([data_np[i+1:i+seq_len+1] for i in ix])
  return Tensor(x).shard(devices, axis=0).realize(), Tensor(y).shard(devices, axis=0).realize()

if __name__ == "__main__":
  data = load_data()
  print(f"Running on {len(DEVICES)} devices: {DEVICES}")

  base_model = Transformer(DIM, N_HEADS, N_LAYERS, 1e-5, VOCAB_SIZE, max_seq_len=SEQ_LEN)

  if len(DEVICES) > 1:
    model = FSDP(base_model, DEVICES, axis=0)
  else:
    model = base_model

  if hasattr(model, "sharded_params"):
    params = [p for p in model.sharded_params.values() if p.requires_grad]
  else:
    params = [p for p in get_parameters(model) if p.requires_grad]

  optimizer = AdamW(params, lr=1e-4)

  @TinyJit
  def train_step(x, y):
    for p in optimizer.params:
      if p.grad is not None: p.grad.assign(p.grad * 0)

    logits = model(x)
    loss = logits.sparse_categorical_crossentropy(y)

    loss.backward()

    if hasattr(model, "sync_grad"): model.sync_grad()

    grads = [p.grad for p in optimizer.params if p.grad is not None]
    if grads:
      norm = sum([p.square().sum() for p in grads]).sqrt()
      for p in optimizer.params:
        if p.grad is not None:
          p.grad.assign(p.grad * (1.0 / (norm + 1e-6)).clip(0, 1.0))

    optimizer.step()
    return loss.realize()

  print(colored("Starting Training...", "green"))

  with Tensor.train():
    for i in range(NUM_ITERS):
      GlobalCounters.reset()
      st = time.perf_counter()

      x, y = get_batch(data, BATCH_SIZE, SEQ_LEN, DEVICES)
      loss = train_step(x, y)

      et = time.perf_counter()
      eff_toks = (BATCH_SIZE * SEQ_LEN) / (et - st)

      print(f"iteration {i}, loss: {loss.item():.6f}, time: {1000*(et - st):.3f}ms, {eff_toks:.0f} tok/s")

