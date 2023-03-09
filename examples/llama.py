#!/usr/bin/env python3
import os
os.environ["METAL"] = "1"  # metal is best choice for llama

import math
import numpy as np
from typing import Optional
from extra.helpers import Timing
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    # TODO: convert to float?
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight

class Attention:
  def __init__(self, dim, n_heads):
    self.wq, self.wk, self.wv, self.wo = [Linear(dim, dim, bias=False) for _ in range(4)]
    self.n_heads = n_heads
    self.head_dim = dim // n_heads

  def __call__(self, x, start_pos, freqs_cis, mask):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk, xv = [x.reshape(bsz, seqlen, self.n_heads, self.head_dim) for x in (xq, xk, xv)]

    # TODO: need this
    #xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # TODO: add cache
    keys, values = xk, xv

    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = xq.matmul(keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None: scores = scores + mask
    scores = scores.softmax()  # this is casted to float
    output = scores.matmul(values).transpose(1,2).reshape(bsz, seqlen, -1)

    return self.wo(output)

class FeedForward:
  def __init__(self, dim, hidden_dim, multiple_of):
    # TODO: what is this?
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)

  def __call__(self, x):
    return self.w2(self.w1(x).silu() * self.w3(x))

class TransformerBlock:
  def __init__(self, dim, multiple_of, n_heads, norm_eps):
    self.attention = Attention(dim, n_heads)
    self.feed_forward = FeedForward(dim, 4*dim, multiple_of)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, start_pos:int, freqs_cis:Tensor, mask:Optional[Tensor]):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out

class Transformer:
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, norm_eps) for i in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    self.tok_embeddings = {"weight": Tensor.zeros(vocab_size, dim)}
    self.output = Linear(dim, vocab_size)

  def __call__(self, tokens:Tensor, start_pos:int):
    h = tokens @ self.tok_embeddings['weight']

    # TODO: write this
    freqs_cis = None
    mask = None

    for layer in self.layers:
      h.realize()  # TODO: why do i need this?
      h = layer(h, start_pos, freqs_cis, mask)

    return self.output(self.norm(h)[:, -1, :])

VOCAB_SIZE = 32000
args_small = {"dim": 512, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE}
args_7B = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}

# TODO: use pathlib
WEIGHTS_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/7B/consolidated.00.pth")
TOKENIZER_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/tokenizer.model")

if __name__ == "__main__":
  from sentencepiece import SentencePieceProcessor
  sp_model = SentencePieceProcessor(model_file=TOKENIZER_FILENAME)
  assert sp_model.vocab_size() == VOCAB_SIZE

  toks = sp_model.encode("Why did the chicken ")

  if getenv("SMALL"):
    model = Transformer(**args_small)
  else:
    model = Transformer(**args_7B)

    from extra.utils import fake_torch_load_zipped, get_child
    weights = fake_torch_load_zipped(open(WEIGHTS_FILENAME, "rb"), load_weights=getenv("WEIGHTS", 1), base_name="consolidated")
    for k,v in weights.items():
      if '.inner_attention.rope.freqs' in k: continue  # no rope today
      mv = get_child(model, k)
      assert mv.shape == v.shape, f"shape mismatch in {k}"
      mv.lazydata.realized = v

  for _ in range(3):
    onehot = np.zeros((1, len(toks), VOCAB_SIZE))
    onehot[0,range(len(toks)),toks] = 1

    with Timing("ran model in "):
      out = model(Tensor(onehot), 0).numpy()
      tok = int(out.argmax(axis=-1)[-1])
      toks.append(tok)

  print(sp_model.decode(toks))

