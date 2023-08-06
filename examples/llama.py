#!/usr/bin/env python3
# pip3 install sentencepiece pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
#import typeguard.importhook
#typeguard.importhook.install_import_hook('tinygrad')

from pathlib import Path
import functools, sys, argparse, math, platform
import numpy as np
from tqdm import tqdm
np.set_printoptions(linewidth=200)
from typing import Optional, Tuple

from tinygrad.helpers import Timing, getenv, DEBUG, dtypes
from tinygrad.lazy import Device
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
def complex_mult(A, c, d):
  a,b = A[:, :, :, :, 0:1], A[:, :, :, :, 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis) -> Tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] and freqs_cis.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == 5 and len(xk.shape) == 5 and len(freqs_cis.shape) == 5
  c, d = freqs_cis[:, :xq.shape[1], :, :, 0:1], freqs_cis[:, :xq.shape[1], :, :, 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    # TODO: convert to float?
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight

class Attention:
  def __init__(self, dim, n_heads, linear=Linear):
    self.wq, self.wk, self.wv, self.wo = [linear(dim, dim, bias=False) for _ in range(4)]
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
  def __init__(self, dim, hidden_dim, multiple_of, linear=Linear, ffn_dim_multiplier=None):
    # TODO: what is this?
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = linear(dim, hidden_dim, bias=False)
    self.w2 = linear(hidden_dim, dim, bias=False)
    self.w3 = linear(dim, hidden_dim, bias=False)

  def __call__(self, x:Tensor) -> Tensor:
    return self.w2(self.w1(x).silu() * self.w3(x))

class TransformerBlock:
  def __init__(self, dim, multiple_of, n_heads, norm_eps, linear=Linear, ffn_dim_multiplier=None):
    self.attention = Attention(dim, n_heads, linear)
    self.feed_forward = FeedForward(dim, 4*dim, multiple_of, linear, ffn_dim_multiplier)
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
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_batch_size=32, max_seq_len=1024, ffn_dim_multiplier=None):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, norm_eps, linear, ffn_dim_multiplier) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = linear(dim, vocab_size, bias=False)
    self.freqs_cis = Tensor(precompute_freqs_cis(dim // n_heads, max_seq_len * 2))

  def __call__(self, tokens:Tensor, start_pos:int):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)

    # get only the part we are using. making it contiguous avoids more kernel calls
    freqs_cis = self.freqs_cis[:, start_pos:start_pos+seqlen].contiguous().realize()
    mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=dtypes.float32).triu(start_pos+1).realize() if seqlen > 1 else None
    h = h.sequential([functools.partial(layer, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask) for layer in self.layers])
    return self.output(self.norm(h))

# **** files and arguments ****

VOCAB_SIZE = 32000
MODEL_PARAMS = {
  1: {
    "7B": {
      "args": {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE},
      "files": 2,
    },
    "30B": {
      "args": {"dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE},
      "files": 4,
    },
    "65B": {
      "args": {"dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE},
      "files": 8,
    },
  },
  2: {
    "7B": {
      "args": {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE},
      "files": 2,
    },
#     # 70B is disabled because we do not yet implement n_kv_heads argument
#     "70B": {
#       "args": {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": VOCAB_SIZE},
#       "files": 8,
#     },
  },
}

# **** helper functions ****
def sample(logits, temperature):
  if temperature < 1e-6:
    # so close to 0 we use argmax
    return int(logits.numpy().argmax())
  else:
    probs = (logits / temperature).softmax()
    probs = probs.numpy().flatten()
    return int(np.random.choice(len(probs), p=probs))

def concat_weights(models):
  def convert(name) -> Tensor:
    disk_tensors = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=Device.DEFAULT)
    axis = 1 if name.startswith('tok_embeddings.') or name.endswith('.attention.wo.weight') or name.endswith('.feed_forward.w2.weight') else 0
    lazy_tensors = [data.to(device=Device.DEFAULT) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

class AbsmaxQuantizedLinear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(dtype=dtypes.half).T/self.scale)

  @staticmethod
  def quantize(tensors):
    new_tensors = {}
    for name,v in tensors.items():
      if 'feed_forward' in name or ('attention.w') in name or name == 'output.weight':
        scale = 127.0 / v.abs().max(axis=1)
        int8_weight = (v.T*scale).T.cast(dtype=dtypes.int8)
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
      else:
        new_tensors[name] = v
    return new_tensors

class LLaMa:
  @staticmethod
  def build(model_path, tokenizer_path, model_gen=1, model_size="7B", quantize=False):
    from sentencepiece import SentencePieceProcessor
    sp_model = SentencePieceProcessor(model_file=str(tokenizer_path))
    assert sp_model.vocab_size() == VOCAB_SIZE

    from tinygrad.state import torch_load, load_state_dict
    params = MODEL_PARAMS[model_gen][model_size]
    model = Transformer(**params["args"], linear=AbsmaxQuantizedLinear) if quantize else Transformer(**params["args"])
    weights = concat_weights([torch_load(filename) for filename in [f"{model_path}/{model_size}/consolidated.{i:02d}.pth" for i in range(params["files"])]])
    if quantize:
      weights = AbsmaxQuantizedLinear.quantize(weights)
    load_state_dict(model, weights, strict=False)

    return LLaMa(model, sp_model)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def greedy_until(self, prompt:str, until, max_length, temperature):
    toks = [self.tokenizer.bos_id()] + self.tokenizer.encode(prompt)
    start_pos = 0
    for i in range(max_length):
      logits = self.model(Tensor([toks[start_pos:]]), start_pos).realize()[:, -1, :]
      tok = sample(logits, temperature)
      start_pos = len(toks)
      toks.append(tok)

      if tok == self.tokenizer.eos_id(): break
      output = self.tokenizer.decode(toks)
      for s in until:
        if output.endswith(s): return output[0:-len(s)]
    return output

# **** main code ****

if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description='Run LLaMA in tinygrad', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # test: python3 examples/llama.py --prompt="Hello." --temperature=0
  # Hello. I'm a 20 year old male. I'm a student at the University of Texas at Austin. I'm a sophomore majoring in Computer Science.
  # test: python3 examples/llama.py --gen 2 --prompt="Hello." --temperature=0
  # Hello. I'm a 20 year old girl who is looking for a good lay in Palm Coast. I don't care whether it's at your place or not, as long as it's clean.
  parser.add_argument('--prompt', type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument('--count', type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument('--personality', type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")

  parser.add_argument('--temperature', type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument('--timing', action='store_true', help="Print timing per token")
  parser.add_argument('--profile', action='store_true', help="Output profile data to out.prof")
  parser.add_argument('--size', type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B] for Gen 2")
  parser.add_argument('--gen', type=int, default="1", help="Generation of the model to use [1, 2]")
  parser.add_argument('--quantize', action='store_true', help="Quantize the weights to int8 in memory")

  args = parser.parse_args()
  chatbot = args.prompt == None

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


  LLAMA_SUFFIX = {1: "", 2: "-2"}[args.gen]
  WEIGHTS_DIR = Path(__file__).parent.parent / f"weights/LLaMA{LLAMA_SUFFIX}/"
  TOKENIZER_FILENAME = WEIGHTS_DIR / "tokenizer.model"
  print(f"using LLaMA{LLAMA_SUFFIX}-{args.size} model")
  llama = LLaMa.build(WEIGHTS_DIR, TOKENIZER_FILENAME, model_gen=args.gen, model_size=args.size, quantize=args.quantize)

  if chatbot:
    # encode pre prompt
    toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(pre_prompt)

    print(f"Preparing KV cache for chatbot with personality {args.personality}...")
    with Timing():
      llama.model(Tensor([toks]), 0).realize()  # NOTE: output logits are not used
    start_pos = len(toks)
  else:
    # non chat bot mode
    toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(args.prompt)
    start_pos = 0

  # print prompt
  outputted = llama.tokenizer.decode(toks)
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

    new_toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(outputted)
    assert toks == new_toks[:len(toks)]
    toks = new_toks
    assert outputted == llama.tokenizer.decode(toks)

    last_break = len(outputted)
    for i in range(args.count):
      if args.profile and i == 2: profiler.enable()

      if args.timing: print("")
      st = GlobalCounters.time_sum_s
      with Timing("ran model in ", on_exit=(lambda et: f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU") if DEBUG else None, enabled=args.timing):
        logits = llama.model(Tensor([toks[start_pos:]]), start_pos).realize()[:, -1, :]
      with Timing("sync in ", enabled=args.timing):
        tok = sample(logits, args.temperature)

      # use the kv cache
      start_pos = len(toks)

      # add the new token
      toks.append(tok)

      # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
      cur = llama.tokenizer.decode(toks)
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
