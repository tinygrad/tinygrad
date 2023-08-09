#!/usr/bin/env python3
# pip3 install tiktoken

import functools, math
import numpy as np
from tqdm import trange
np.set_printoptions(linewidth=200)
from typing import Optional, Tuple

from tinygrad.helpers import getenv, dtypes
from tinygrad.lazy import Device
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.jit import TinyJit

from examples.llama import sample

class LayerNorm:
  def __init__(self, dim, eps=1e-5):
    self.eps = eps
    self.weight = Tensor.ones(dim)
    self.bias = Tensor.zeros(dim)

  def __call__(self, x:Tensor):
    return (x.layernorm(eps=self.eps)) * self.weight + self.bias

class Attention:
  def __init__(self, dim, n_heads, linear=Linear):
    self.c_attn = linear(dim, 3*dim, bias=True)
    self.c_proj = linear(dim, dim, bias=True)
    self.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads

  def prepare_attention(self, x:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xqkv = self.c_attn(x)
    xq, xk, xv = [xqkv.slice([None, None, (i*self.dim, (i+1)*self.dim)]) for i in range(3)]
    xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim) for x in (xq, xk, xv)]
    return xq, xk, xv

  def inner_attention(self, xq:Tensor, xk:Tensor, xv:Tensor, start_pos:int, mask:Optional[Tensor]) -> Tensor:
    bsz, seqlen, _, _ = xq.shape
    # kv caching!
    if start_pos == 0:
      keys, values = xk, xv
    else:
      assert hasattr(self, 'cache_k'), "no cache"
      assert start_pos == self.cache_k.shape[1] and start_pos == self.cache_v.shape[1], "cache is wrong shape"
      assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
      keys, values = self.cache_k.cat(xk, dim=1), self.cache_v.cat(xv, dim=1)

    # save the cache
    self.cache_k, self.cache_v = keys.realize(), values.realize()

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = xq.matmul(keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask
    scores = scores.softmax()  # this is casted to float
    return scores.matmul(values).transpose(1, 2).reshape(bsz, seqlen, -1)

  # NOTE: this is not called
  def __call__(self, x:Tensor, start_pos:int, mask:Optional[Tensor]) -> Tensor:
    xq, xk, xv = self.prepare_attention(x)
    output = self.inner_attention(xq, xk, xv, start_pos, mask)
    return self.c_proj(output)

class FeedForward:
  def __init__(self, dim, hidden_dim, linear=Linear):
    self.c_fc = linear(dim, hidden_dim, bias=True)
    self.c_proj = linear(hidden_dim, dim, bias=True)

  def __call__(self, x:Tensor) -> Tensor:
    return self.c_proj(self.c_fc(x).gelu())

class TransformerBlock:
  def __init__(self, dim, n_heads, norm_eps, linear=Linear):
    self.attn = Attention(dim, n_heads, linear)
    self.mlp = FeedForward(dim, 4*dim, linear)
    self.ln_1 = LayerNorm(dim, norm_eps)
    self.ln_2 = LayerNorm(dim, norm_eps)
    if getenv("JIT"):
      self._pre = TinyJit(self.pre)
      self._post = TinyJit(self.post)
    else:
      self._pre, self._post = self.pre, self.post

  def pre(self, x:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xq, xk, xv = self.attn.prepare_attention(self.ln_1(x))
    return xq.realize(), xk.realize(), xv.realize()

  def post(self, x:Tensor, output:Tensor) -> Tensor:
    h = x + self.attn.c_proj(output)
    return (h + self.mlp(self.ln_2(h))).realize()

  def __call__(self, x:Tensor, start_pos:int, mask:Optional[Tensor]):
    xq, xk, xv = self._pre(x)
    # inner_attention can't be jitted because it's dynamic based on start_pos
    output = self.attn.inner_attention(xq, xk, xv, start_pos, mask)
    return self._post(x, output)

class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps=1e-5, vocab_size=50257, linear=Linear, max_seq_len=1024):
    self.wte = Embedding(vocab_size, dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps, linear) for _ in range(n_layers)]
    self.ln_f = LayerNorm(dim, norm_eps)
    self.lm_head = linear(dim, vocab_size, bias=False)

  def __call__(self, tokens:Tensor, start_pos:int):
    _bsz, seqlen = tokens.shape
    tok_emb = self.wte(tokens)
    pos = Tensor.arange(start_pos, start_pos + seqlen).reshape(1, -1)
    pos_emb = self.wpe(pos)
    h = tok_emb + pos_emb

    # get only the part we are using. making it contiguous avoids more kernel calls
    mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=dtypes.float32).triu(start_pos+1).realize() if seqlen > 1 else None
    h = h.sequential([functools.partial(layer, start_pos=start_pos, mask=mask) for layer in self.h])
    h = self.ln_f(h)
    return self.lm_head(h)

# **** files and arguments ****

MODEL_PARAMS = {
  'gpt2':         dict(n_layers=12, n_heads=12, dim=768),   # 124M params
  'gpt2-medium':  dict(n_layers=24, n_heads=16, dim=1024),  # 350M params
  'gpt2-large':   dict(n_layers=36, n_heads=20, dim=1280),  # 774M params
  'gpt2-xl':      dict(n_layers=48, n_heads=25, dim=1600),  # 1558M params
}

def get_url(model_size): return f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'

class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    import tiktoken
    from tinygrad.state import torch_load, load_state_dict
    from extra.utils import fetch_as_file
    tokenizer = tiktoken.get_encoding("gpt2")

    params = MODEL_PARAMS[model_size]
    model = Transformer(**params)
    weights = torch_load(fetch_as_file(get_url(model_size)))
    # special treatment for the Conv1D weights we need to transpose
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    for k in weights.keys():
      if any(k.endswith(w) for w in transposed):
        weights[k] = Tensor(weights[k].numpy().T)
    # lm head and wte are tied
    weights['lm_head.weight'] = Tensor(weights['wte.weight'].numpy())

    load_state_dict(model, weights)
    return GPT2(model, tokenizer)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def greedy_until(self, prompt:str, max_length, temperature):
    toks = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    start_pos = 0
    for _ in trange(max_length):
      logits = self.model(Tensor([toks[start_pos:]]), start_pos).realize()[:, -1, :]
      tok = sample(logits, temperature)
      start_pos = len(toks)
      toks.append(tok)
      output = self.tokenizer.decode(toks)
    return output

# **** main code ****

if __name__ == "__main__":
  Tensor.no_grad = True
  model_size = "gpt2-medium"
  print(f"using {Device.DEFAULT} backend")
  print(f"using {model_size}")

  gpt2 = GPT2.build(model_size)
  print('Generating text...')
  y = gpt2.greedy_until('What is the answer to life, the universe, and everything?', 100, 1)
  print(y)
