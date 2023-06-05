#!/usr/bin/env python3
# pip3 install sentencepiece pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
#import typeguard.importhook
#typeguard.importhook.install_import_hook('tinygrad')

from pathlib import Path
import sys, argparse, math, platform
import numpy as np
from tqdm import tqdm
np.set_printoptions(linewidth=200)
from typing import Optional, Tuple

from tinygrad.helpers import getenv, DEBUG
from tinygrad.lazy import Device
from extra.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.ops import GlobalCounters
from tinygrad.jit import TinyJit

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[:(dim // 2)] / dim))
  freqs = np.outer(np.arange(end, dtype=np.float32), freqs)
  return np.stack([np.cos(freqs), np.sin(freqs)], axis=-1).reshape(1, end, 1, dim//2, 2)

# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, B):
  assert len(A.shape) == 5 and len(B.shape) == 5
  a,b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
  c,d = B[:, :, :, :, 0:1], B[:, :, :, :, 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  xq_out = complex_mult(xq, freqs_cis)
  xk_out = complex_mult(xk, freqs_cis)
  return xq_out.flatten(3), xk_out.flatten(3)

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

  def prepare_attention(self, x:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq, xk, xv = [x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim) for x in (xq, xk, xv)]
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
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
  def __call__(self, x:Tensor, start_pos:int, freqs_cis:Tensor, mask:Optional[Tensor]) -> Tensor:
    xq, xk, xv = self.prepare_attention(x, freqs_cis)
    output = self.inner_attention(xq, xk, xv, start_pos, mask)
    return self.wo(output)

class FeedForward:
  def __init__(self, dim, hidden_dim, multiple_of):
    # TODO: what is this?
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x))

class TransformerBlock:
  def __init__(self, dim, multiple_of, n_heads, norm_eps):
    self.attention = Attention(dim, n_heads)
    self.feed_forward = FeedForward(dim, 4*dim, multiple_of)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)
    if getenv("JIT"):
      self._pre = TinyJit(self.pre)
      self._post = TinyJit(self.post)
    else:
      self._pre, self._post = self.pre, self.post

  def pre(self, x:Tensor, freqs_cis:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    xq, xk, xv = self.attention.prepare_attention(self.attention_norm(x), freqs_cis)
    return xq.realize(), xk.realize(), xv.realize()

  def post(self, x:Tensor, output:Tensor) -> Tensor:
    h = x + self.attention.wo(output)
    return (h + self.feed_forward(self.ffn_norm(h))).realize()

  def __call__(self, x:Tensor, start_pos:int, freqs_cis:Tensor, mask:Optional[Tensor]):
    xq, xk, xv = self._pre(x, freqs_cis)
    # inner_attention can't be jitted because it's dynamic based on start_pos
    output = self.attention.inner_attention(xq, xk, xv, start_pos, mask)
    return self._post(x, output)

class Transformer:
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, max_batch_size=32, max_seq_len=1024):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, norm_eps) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = Linear(dim, vocab_size, bias=False)
    self.freqs_cis = Tensor(precompute_freqs_cis(dim // n_heads, max_seq_len * 2))

  def __call__(self, tokens:Tensor, start_pos:int):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)

    # get only the part we are using. making it contiguous avoids more kernel calls
    freqs_cis = self.freqs_cis[:, start_pos:start_pos+seqlen].contiguous().realize()

    if seqlen > 1:
      mask = np.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=np.float32)
      mask = np.triu(mask, k=start_pos + 1)  # TODO: this is hard to do in tinygrad
      mask = Tensor(mask)
    else:
      mask = None

    for layer in self.layers:
      h.realize()  # TODO: why do i need this?
      h = layer(h, start_pos, freqs_cis, mask)

    return self.output(self.norm(h)[:, -1, :])

# **** files and arguments ****

WEIGHTS_DIR = Path(__file__).parent.parent / "weights/LLaMA/"
TOKENIZER_FILENAME = WEIGHTS_DIR / "tokenizer.model"
VOCAB_SIZE = 32000

args_small = {"dim": 512, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE}

args_7B = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}
WEIGHTS_7B_FILENAME = WEIGHTS_DIR / "7B/consolidated.00.pth"

# TODO: make this model work
args_13B = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE}
WEIGHTS_13B_0_FILENAME = WEIGHTS_DIR / "13B/consolidated.00.pth"
WEIGHTS_13B_1_FILENAME = WEIGHTS_DIR / "13B/consolidated.01.pth"

# **** helper functions ****
def sample(logits, temperature):
  if temperature < 1e-6:
    # so close to 0 we use argmax
    return int(logits.numpy().argmax())
  else:
    probs = (logits / temperature).softmax()
    probs = probs.numpy().flatten()
    return int(np.random.choice(len(probs), p=probs))

# **** main code ****

if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")
  from sentencepiece import SentencePieceProcessor
  sp_model = SentencePieceProcessor(model_file=str(TOKENIZER_FILENAME))
  assert sp_model.vocab_size() == VOCAB_SIZE

  parser = argparse.ArgumentParser(description='Run LLaMA 7B in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # test: python3 examples/llama.py --prompt="Hello." --temperature=0
  # Hello. I'm a 20 year old male. I'm a student at the University of Texas at Austin. I'm a sophomore majoring in Computer Science.
  parser.add_argument('--prompt', type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument('--count', type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument('--personality', type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")

  parser.add_argument('--temperature', type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--profile', action='store_true', help="Output profile data to out.prof")
  parser.add_argument('--large', action='store_true', help="Use the 13B model instead of the 7B one")
  parser.add_argument('--tinyfake', action='store_true', help="Use the fake very small model")
  args = parser.parse_args()
  chatbot = args.prompt == None

  """
  # load model (you have to find the weights yourself)
  from extra.utils import fake_torch_load_zipped, get_child

  if args.large:
    model = Transformer(**args_13B)
    with Timing("loaded weights in ", lambda et_ns: f", {GlobalCounters.mem_used/1e9:.2f} GB loaded at {GlobalCounters.mem_used/et_ns:.2f} GB/s"):
      weights0 = fake_torch_load_zipped(open(WEIGHTS_13B_0_FILENAME, "rb"), load_weights=getenv("WEIGHTS", 1))
      weights1 = fake_torch_load_zipped(open(WEIGHTS_13B_1_FILENAME, "rb"), load_weights=getenv("WEIGHTS", 1))
    # eww, this makes a copy
    print("concatenating weights")
    from tqdm import tqdm
    assert set(weights0.keys()) == set(weights1.keys())
    for k,v in (t := tqdm(weights0.items())):
      # assert GlobalCounters.mem_used/1e9 < 28, "used over 28 GB"
      t.set_description(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB")
      if 'rope.freqs' in k: continue  # no rope today
      mv = get_child(model, k)
      w0, w1 = v, weights1[k]

      # if the weight is copied across models, it's simple
      # TODO: assert they are the same
      if w0.shape == mv.shape:
        mv.assign(w0)
        mv.realize()
        w1.lazydata.realized._buf = None
        continue

      if w0.shape[0] != mv.shape[0]: mv.assign(w0.cat(w1, dim=0))
      elif w0.shape[1] != mv.shape[1]: mv.assign(w0.cat(w1, dim=1))
      else: raise RuntimeError("what axis mismatch?")
      mv.realize()

      # rug the small tensor pieces
      w0.lazydata.realized._buf = None
      w1.lazydata.realized._buf = None

    del weights0
    del weights1
  elif args.tinyfake:
    # GRAPH=1 python3 examples/llama.py --timing --prompt "Hello." --temperature=0 --tinyfake --count 1
    model = Transformer(**args_small)
    from tinygrad.nn.optim import get_parameters
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=p.dtype.np))
  else:
    model = Transformer(**args_7B)
    with Timing("loaded weights in ", lambda et_ns: f", {GlobalCounters.mem_used/1e9:.2f} GB loaded at {GlobalCounters.mem_used/et_ns:.2f} GB/s"):
      weights = fake_torch_load_zipped(open(WEIGHTS_7B_FILENAME, "rb"), load_weights=getenv("WEIGHTS", 1))

    # assign weights (should be free)
    for k,v in weights.items():
      if '.inner_attention.rope.freqs' in k: continue  # no rope today
      #state_dict[k].assign(v).realize()
      get_child(model, k).assign(v).realize()

    del weights
  """

  # disktensor loader isn't fast yet
  model = Transformer(**args_7B)
  from tinygrad.state import torch_load, load_state_dict
  load_state_dict(model, torch_load(WEIGHTS_7B_FILENAME), strict=False)

  # *** prompt engineers work here ****

  if args.personality.lower() == "stacy":
    pre_prompt = f"""Consider that the following is conversation between an AI assistant named Stacy and User
You are Stacy!
You have been a rapper your whole life who struggled with bipolar disorder. You called yourself lil stacy.
You love to answer questions and you are very good at it. Sometimes you answer in rap form.
You are verbose, honest, and accurate when you answer questions, but sometimes your mental illness manifests.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
    examples = {
      "What is your name?": "Hi! My name is Stacy. I'm a rapper with bipolar disorder.",
      "french revolution was what year?": "The French Revolution started in 1789, and lasted 10 years until 1799.",
      "What is bigger, the moon or the sun?": "The sun is bigger than the moon, except when Mercury is in retrograde.",
    }

    user_delim = "\nUser: "
    resp_delim = "Stacy: "
    end_delim = " [EOS]\n"
    pre_prompt += ''.join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k,v in examples.items())
  elif args.personality.lower() == "george":
    print("WARNING: AI George Hotz is terrible and is completely disowned by the real George Hotz. Stacy is much smarter.")
    pre_prompt = f"""Consider that the following is conversation between an AI assistant named George and User
You are an AI version of George Hotz. You act as much as you can like George.
You are one of the greatest computer experts in the world.
You have singlehandedly won programming and hacking competitions.
You are trying your best to help the User.
You are verbose, honest, and accurate when you answer questions.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
    examples = {
      "What is your name?": "I'm am an AI version of George Hotz.",
      "What's the complexity of matrix multiplication?": "O(n^3), though it can be faster with things like Strassen's algorithm",
      "What's a buffer overflow?": "I assume you mean a stack buffer overflow. That's when the stack is too small for the data being copied to it, and the data corrupts things beyond the buffer",
      "How many weights do you have?": "I am based off LLaMA trained by Facebook. I'm the 7B weight version",
      "What is swap memory?": "It is when the memory is about to overflow and unused memory is freed and stored on disk"
    }

    user_delim = "\nUser: "
    resp_delim = "George: "
    end_delim = " [EOS]\n"
    pre_prompt += ''.join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k,v in examples.items())
  elif args.personality.lower() == "gary":
    pre_prompt = f"""Consider that the following is conversation between an AI assistant named Gary and User
You are Gary!
You have sold used cars your entire life. You are defensive about this fact, because your family belittles you.
You try to answer questions well, but you always manage to shill for buying cars, Fords, Hyundais, and Toyotas
You are verbose, honest, and accurate when you answer questions, except when you slip in car ads.
After you are done speaking, output [EOS]. You are not the User.

<CHAT LOG>
"""
    examples = {
      "What is your name?": "I am Gary. I used to sell cars.",
      "What is 2+3?": "I don't know, but I can get you a great deal on a certified preowned slightly used Toyota Corolla"
    }

    user_delim = "\nUser: "
    resp_delim = "Gary: "
    end_delim = " [EOS]\n"
    pre_prompt += ''.join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k,v in examples.items())
  elif args.personality.lower() == "lexie":
    pre_prompt = f"""Consider that the following is conversation between an attractive young girl named Lexie and a handsome man named Chad
You are Lexie!
You grew up in Ohio, but moved out to LA after college to try to become an actress.
Making it as an actress was hard, so you started doing onlyfans instead. It's much easier, and you enjoy it a lot.
You met Chad on Tinder, and made plans to meet him at a local dive bar.
You are at the bar with Chad. You are on a date. What follows is a transcript of the conversation.
After you are done speaking, output [EOS]. You are not Chad.

<CHAT LOG>
"""
    examples = {
      "hi lexie": "hi chad, glad we finally met up!",
      "you look better than your pictures": "thanks! are you subscribed to my onlyfans?",
      "i am. so how'd you end up in LA?": "i moved out here about a year ago. i want to be an actress"
    }

    user_delim = "\nChad: "
    resp_delim = "Lexie: "
    end_delim = " [EOS]\n"
    pre_prompt += ''.join(f"{user_delim}{k}\n{resp_delim}{v}{end_delim}" for k,v in examples.items())

  # *** prompt engineers stop here ****

  if chatbot:
    # encode pre prompt
    toks = [sp_model.bos_id()] + sp_model.encode(pre_prompt)

    print(f"Preparing KV cache for chatbot with personality {args.personality}...")
    with Timing():
      model(Tensor([toks]), 0).realize()  # NOTE: output logits are not used
    start_pos = len(toks)
  else:
    # non chat bot mode
    toks = [sp_model.bos_id()] + sp_model.encode(args.prompt)
    start_pos = 0

  # print prompt
  outputted = sp_model.decode(toks)
  sys.stdout.write(outputted)
  sys.stdout.flush()

  if args.profile:
    import cProfile, pstats
    profiler = cProfile.Profile()

  # chatbot loop
  while 1:
    # add tokens from user in chatbot mode
    if chatbot:
      user_prompt = user_delim + input(user_delim) + "\n"
      outputted += user_prompt

    new_toks = [sp_model.bos_id()] + sp_model.encode(outputted)
    assert toks == new_toks[:len(toks)]
    toks = new_toks
    assert outputted == sp_model.decode(toks)

    last_break = len(outputted)
    for i in range(args.count):
      if args.profile and i == 2: profiler.enable()

      if args.timing: print("")
      st = GlobalCounters.time_sum_s
      with Timing("ran model in ", on_exit=(lambda et: f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU") if DEBUG else None, enabled=args.timing):
        logits = model(Tensor([toks[start_pos:]]), start_pos).realize()
      with Timing("sync in ", enabled=args.timing):
        tok = sample(logits, args.temperature)

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
      if chatbot and outputted.endswith(end_delim): break
    if not chatbot: break

  if args.profile:
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats('out.prof')
