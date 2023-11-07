#!/usr/bin/env python3
# pip3 install tiktoken

import argparse
import numpy as np
from tqdm import trange
np.set_printoptions(linewidth=200)
from typing import Optional, Dict

from tinygrad.helpers import Timing, getenv, dtypes, DEBUG
from tinygrad.helpers import GlobalCounters
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.jit import TinyJit
from tinygrad.shape.symbolic import Variable

MAX_CONTEXT = 128

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

  def __call__(self, x:Tensor, cache_k:Optional[Tensor], cache_v:Optional[Tensor], start_pos:int, mask:Optional[Tensor]) -> Tensor:
    xqkv = self.c_attn(x)
    xq, xk, xv = [xqkv.slice([None, None, (i*self.dim, (i+1)*self.dim)]) for i in range(3)]
    xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim) for x in (xq, xk, xv)]

    bsz, seqlen, _, _ = xq.shape
    # kv caching!
    if start_pos == 0:
      keys, values = xk, xv
    else:
      assert cache_k, "no cache"
      #assert start_pos == cache_k.shape[1] and start_pos == cache_v.shape[1], "cache is wrong shape"
      assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
      keys, values = cache_k.cat(xk, dim=1), cache_v.cat(xv, dim=1)

    # save the cache
    cache_k, cache_v = keys.realize(), values.realize()
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    output = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.c_proj(output), cache_k, cache_v

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

  def __call__(self, x:Tensor, cache_k:Optional[Tensor], cache_v:Optional[Tensor], start_pos:int, mask:Optional[Tensor]):
    output, cache_k, cache_v = self.attn(self.ln_1(x), cache_k, cache_v, start_pos, mask)
    h = x + output
    h = (h + self.mlp(self.ln_2(h)))
    return h, cache_k, cache_v

class Transformer:
  def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_seq_len=1024):
    self.wte = Embedding(vocab_size, dim)
    self.wpe = Embedding(max_seq_len, dim)
    self.h = [TransformerBlock(dim, n_heads, norm_eps, linear) for _ in range(n_layers)]
    self.kv_caches = None
    self.n_layers = n_layers
    self.ln_f = LayerNorm(dim, norm_eps)
    self.lm_head = linear(dim, vocab_size, bias=False)

  def embed(self, tokens:Tensor, pos:Tensor):
    tok_emb = self.wte(tokens)
    pos_emb = self.wpe(pos)
    h = tok_emb + pos_emb
    if getenv("FP16"): h = h.half()
    return h

  def postprocess(self, x, temperature:Optional[float]):
    logits = self.lm_head(self.ln_f(x))
    if temperature is not None: return (logits[:, -1, :] / (temperature+1e-10)).softmax().flatten()
    return logits

  @TinyJit
  def run_all_layers(self, tokens:Tensor, pos:Tensor, start_pos:int, temperature:float, **kv_cache):
    h = self.embed(tokens, pos)

    for i, hi in enumerate(self.h):
      h, kv_cache[f'cache_k{i}'], kv_cache[f'cache_v{i}'] = hi(h, kv_cache[f'cache_k{i}'], kv_cache[f'cache_v{i}'], start_pos=start_pos, mask=None)

    # don't realize until here
    for v in kv_cache.values(): v.realize()
    return self.postprocess(h, temperature).realize(), kv_cache

  def __call__(self, tokens:Tensor, start_pos:int, temperature:Optional[float]=None):
    _bsz, seqlen = tokens.shape
    if seqlen == 1 and start_pos > 0 and getenv("JIT"):
      start_pos_var = Variable("start_pos", 1, MAX_CONTEXT).bind(start_pos)
      pos = self.allpos.shrink(((0, self.allpos.shape[0]), (start_pos_var, start_pos_var + seqlen)))
      for k,v in self.kv_caches.items():
        self.kv_caches[k] = v.reshape(v.shape[0], start_pos_var, v.shape[2], v.shape[3])
      logit_or_softmax, self.kv_caches = self.run_all_layers(tokens, pos, start_pos=start_pos, temperature=temperature, **self.kv_caches)
      return logit_or_softmax
    else:
      if start_pos == 0:
        if not hasattr(self, 'allpos'): self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
        self.kv_caches = {**{f'cache_k{i}':None  for i in range(self.n_layers)}, **{f'cache_v{i}':None for i in range(self.n_layers)}}

      pos = self.allpos.shrink(((0, self.allpos.shape[0]), (start_pos, start_pos+seqlen)))
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=dtypes.float16 if getenv('FP16') else dtypes.float32).triu(start_pos+1).realize()
      h = self.embed(tokens, pos)
      for i, hi in enumerate(self.h):
        h, self.kv_caches[f'cache_k{i}'], self.kv_caches[f'cache_v{i}'] = hi(h, self.kv_caches[f'cache_k{i}'], self.kv_caches[f'cache_v{i}'], start_pos=start_pos, mask=mask)
      for v in self.kv_caches.values(): v.realize()
      return self.postprocess(h, temperature).realize()

# **** files and arguments ****

VOCAB_SIZE = 50257
MODEL_PARAMS = {
  'gpt2':         dict(n_layers=12, n_heads=12, dim=768, norm_eps=1e-5, vocab_size=VOCAB_SIZE),   # 124M params
  'gpt2-medium':  dict(n_layers=24, n_heads=16, dim=1024, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 350M params
  'gpt2-large':   dict(n_layers=36, n_heads=20, dim=1280, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 774M params
  'gpt2-xl':      dict(n_layers=48, n_heads=25, dim=1600, norm_eps=1e-5, vocab_size=VOCAB_SIZE),  # 1558M params
}

def get_url(model_size): return f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'

class GPT2:
  @staticmethod
  def build(model_size="gpt2"):
    import tiktoken
    from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
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

    if getenv("FP16"):
      for k, v in weights.items():
        weights[k] = v.cpu().half().realize()
    load_state_dict(model, weights)
    return GPT2(model, tokenizer)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def greedy_until(self, prompt:str, max_length:int, temperature:float, timing:bool=False):
    toks = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    start_pos = 0
    for _ in trange(max_length, disable=(timing==True)):
      GlobalCounters.reset()
      if timing: print("")
      st = GlobalCounters.time_sum_s
      with Timing("total ", enabled=timing):
        with Timing("ran model in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                    f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                    (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=timing):
          probs = self.model(Tensor([toks[start_pos:]]), start_pos, temperature)
        probs_np = probs.numpy()
        tok = int(np.random.choice(len(probs_np), p=probs_np))
      start_pos = len(toks)
      toks.append(tok)
      output = self.tokenizer.decode(toks)
    return output

# **** main code ****

if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run GPT2 in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--prompt', type=str, default="What is the answer to life, the universe, and everything?", help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Max number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.8, help="Temperature in the softmax")
  parser.add_argument('--model_size', type=str, default="gpt2-medium", help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--seed', type=int, help="Set the random seed")
  args = parser.parse_args()

  if args.seed is not None:
    Tensor._seed = args.seed
    np.random.seed(args.seed)

  print(f"using {args.model_size}")
  gpt2 = GPT2.build(args.model_size)
  print('Generating text...')
  y = gpt2.greedy_until(args.prompt, args.count, args.temperature, timing=args.timing)
  print(y)