#!/usr/bin/env python3
import os
os.environ["METAL"] = "1"  # metal is best choice for llama

import sys, argparse, math
import numpy as np
np.set_printoptions(linewidth=200)
from hexdump import hexdump
from typing import Optional
from extra.helpers import Timing
from tinygrad.helpers import getenv, DEBUG
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.ops import GlobalCounters

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = np.outer(np.arange(end), freqs)
  return np.stack([np.cos(freqs), np.sin(freqs)], axis=-1).reshape(1, end, 1, dim//2, 2)

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    # TODO: convert to float?
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight

# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, B):
  assert len(A.shape) == 5 and len(B.shape) == 5
  a,b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
  c,d = B[:, :, :, :, 0:1], B[:, :, :, :, 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis):
  assert freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[1], "freqs_cis shape mismatch"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  xq_out = complex_mult(xq, freqs_cis)
  xk_out = complex_mult(xk, freqs_cis)
  return xq_out.flatten(3), xk_out.flatten(3)

class Attention:
  def __init__(self, dim, n_heads):
    self.wq, self.wk, self.wv, self.wo = [Linear(dim, dim, bias=False) for _ in range(4)]
    self.n_heads = n_heads
    self.head_dim = dim // n_heads

  def __call__(self, x, start_pos, freqs_cis, mask):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk, xv = [x.reshape(bsz, seqlen, self.n_heads, self.head_dim) for x in (xq, xk, xv)]
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

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
    output = scores.matmul(values).transpose(1, 2).reshape(bsz, seqlen, -1)
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
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, max_batch_size=32, max_seq_len=1024):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, norm_eps) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    self.tok_embeddings = {"weight": Tensor.zeros(vocab_size, dim)}
    self.output = Linear(dim, vocab_size, bias=False)
    self.freqs_cis = Tensor(precompute_freqs_cis(dim // n_heads, max_seq_len * 2))

  def __call__(self, tokens:Tensor, start_pos:int):
    _bsz, seqlen, _ = tokens.shape
    h = tokens @ self.tok_embeddings['weight']

    # get only the part we are using
    freqs_cis = self.freqs_cis[:, start_pos:start_pos+seqlen]

    # WTF!!! This changes the output, and fixes the kv caching. Most serious tinygrad bug in a while.
    # It is not fixed by disabling the method cache.
    # TODO: P0. Fix this bug. An offset is likely getting lost somewhere.
    freqs_cis.realize()

    if seqlen > 1:
      mask = np.full((1, 1, seqlen, start_pos + seqlen), float("-inf"))
      mask = np.triu(mask, k=start_pos + 1)  # TODO: this is hard to do in tinygrad
      mask = Tensor(mask)
    else:
      mask = None

    for layer in self.layers:
      h.realize()  # TODO: why do i need this?
      h = layer(h, start_pos, freqs_cis, mask)

    return self.output(self.norm(h)[:, -1, :])

TOKENIZER_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/tokenizer.model")
VOCAB_SIZE = 32000
args_small = {"dim": 512, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE}

args_7B = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}
WEIGHTS_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/7B/consolidated.00.pth")

# TODO: make this model work
args_13B = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}
WEIGHTS0_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/13B/consolidated.00.pth")
WEIGHTS1_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/LLaMA/13B/consolidated.01.pth")

def onehot_encode(toks):
  # this allows the embedding to work in tinygrad
  onehot = np.zeros((1, len(toks), VOCAB_SIZE))
  onehot[0,range(len(toks)),toks] = 1
  return Tensor(onehot)

if __name__ == "__main__":
  # pip3 install sentencepiece
  from sentencepiece import SentencePieceProcessor
  sp_model = SentencePieceProcessor(model_file=TOKENIZER_FILENAME)
  assert sp_model.vocab_size() == VOCAB_SIZE

  parser = argparse.ArgumentParser(description='Run LLaMA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # (with temperature=0): Hello. I'm a 20 year old male. I'm a student at the University of Texas at Austin. I'm a sophomore majoring in Computer Science.
  parser.add_argument('--prompt', type=str, default="Hello.", help="Phrase to start with")
  parser.add_argument('--count', type=int, default=100, help="Number of tokens to generate")
  parser.add_argument('--temperature', type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  args = parser.parse_args()

  def sample(logits):
    if args.temperature < 1e-6:
      # so close to 0 we use argmax
      return int(logits.numpy().argmax())
    else:
      probs = (logits / args.temperature).softmax()
      probs = probs.numpy().flatten()
      return int(np.random.choice(len(probs), p=probs))

  examples = {
    "What is your name?": "Hi! My name is Stacy. I'm a rapper with bipolar disorder.",
    #"french revolution was what year?": "The French Revolution started in 1789, and lasted 10 years until 1799.",
    #"Can you multiply a matrix using Python?": "```def matmul(a, b): return a@b```",
    #"What is bigger, the moon or the sun?": "The sun is bigger than the moon.",
    #"What is 7+9?": "7 plus 9 is equal to 16",
  }

  user_delim = "\nUser: "
  resp_delim = "Stacy: "
  end_delim = " [END]\n"

  #You are also an expert programmer who has won 7 USAMO medals and 6 Google CodeJams.
  pre_prompt = f"""Consider that the following is conversation between an AI assistant named Stacy and User
You are Stacy!
You have been a rapper your whole life who struggled with bipolar disorder. You called yourself lil stacy.
You love to answer questions and you are very good at it. Sometimes you answer in rap form.
You are verbose, honest, and accurate when you answer questions, but sometimes your mental illness manifests.

<CHAT LOG>
""" + ''.join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k,v in examples.items())

  # load model
  if getenv("SMALL"):
    model = Transformer(**args_small)
  else:
    model = Transformer(**args_7B)

    from extra.utils import fake_torch_load_zipped, get_child
    with Timing("loaded weights in ", lambda et_ns: f", {GlobalCounters.mem_used/et_ns*1e3:.2f} MB/s"):
      weights = fake_torch_load_zipped(open(WEIGHTS_FILENAME, "rb"), load_weights=getenv("WEIGHTS", 1), base_name="consolidated")
    for k,v in weights.items():
      if '.inner_attention.rope.freqs' in k: continue  # no rope today
      mv = get_child(model, k)
      assert mv.shape == v.shape, f"shape mismatch in {k}"
      mv.lazydata.realized = v

  toks = [sp_model.bos_id()] + sp_model.encode(pre_prompt)

  # prepare kv cache
  logits = model(onehot_encode(toks), 0).realize()
  start_pos = len(toks)
  # don't append the guess

  # print prompt
  outputted = sp_model.decode(toks)
  sys.stdout.write(outputted)
  sys.stdout.flush()

  # chatbot loop
  while 1:
    # add tokens from user
    user_prompt = user_delim + input(user_delim) + "\n"
    outputted += user_prompt

    new_toks = [sp_model.bos_id()] + sp_model.encode(outputted)
    assert toks == new_toks[:len(toks)]
    toks = new_toks
    assert outputted == sp_model.decode(toks)

    last_break = len(outputted)
    for i in range(args.count):
      logits = model(onehot_encode(toks[start_pos:]), start_pos)
      tok = sample(logits)

      # use the kv cache
      start_pos = len(toks)

      # add the new token
      toks.append(tok)

      # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
      cur = sp_model.decode(toks)
      sys.stdout.write(cur[len(outputted):])
      sys.stdout.flush()
      outputted = cur

      # stop after you have your answer
      #if 'A: ' in outputted[last_break:] and outputted.endswith("\n"): break
      if outputted.endswith(end_delim): break

