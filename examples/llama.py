#!/usr/bin/env python3
# pip3 install sentencepiece pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
#import typeguard.importhook
#typeguard.importhook.install_import_hook('tinygrad')

from pathlib import Path
import functools, sys, argparse, json, os
import numpy as np
np.set_printoptions(linewidth=200)
from typing import Optional, Tuple, Dict

from tinygrad.helpers import Timing, getenv, DEBUG, dtypes, CI
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.state import safe_load, torch_load, load_state_dict
from tinygrad.helpers import GlobalCounters
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE
from tinygrad.shape.symbolic import Variable, sym_infer

JIT = getenv("JIT", 0 if CI else int(Device.DEFAULT in JIT_SUPPORTED_DEVICE))

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1)*freqs.unsqueeze(dim=0)
  return Tensor.stack([Tensor.cos(freqs), Tensor.sin(freqs)], dim=-1).reshape(1, end, 1, dim//2, 2)

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

def repeat_kv(x:Tensor, n_rep:int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  return x.reshape(bs, seqlen, n_kv_heads, 1, head_dim).expand(bs, seqlen, n_kv_heads, n_rep, head_dim).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    # TODO: convert to float?
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight

class Attention:
  def __init__(self, dim, n_heads, n_kv_heads, linear=Linear):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads

    self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

  def __call__(self, x:Tensor, cache_k:Optional[Tensor], cache_v:Optional[Tensor], start_pos:int, freqs_cis:Tensor, mask:Optional[Tensor], jit_ctx:Optional[Dict[Variable,int]]=None) -> Tuple[Tensor, Tensor, Tensor]:
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # kv caching!
    if start_pos == 0:
      keys, values = xk, xv
    else:
      assert cache_k is not None and cache_v is not None, "no cache"
      assert start_pos == (cache_k.shape[1].val if isinstance(cache_k.shape[1], Variable) else cache_k.shape[1]) == (cache_v.shape[1].val if isinstance(cache_v.shape[1], Variable) else cache_v.shape[1]), f"cache has wrong shape, {start_pos=}, {cache_k.shape[1]=}, {cache_v.shape[1]=}"
      assert seqlen == xk.shape[1] and seqlen == xv.shape[1], "seqlen is wrong shape?!?"
      keys, values = cache_k.cat(xk, dim=1), cache_v.cat(xv, dim=1)

    cache_k, cache_v = keys, values
    keys, values = repeat_kv(keys, self.n_rep).realize(), repeat_kv(values, self.n_rep).realize()
    attn = Tensor.scaled_dot_product_attention(xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), mask).transpose(1, 2).reshape(bsz, seqlen, -1)
    return self.wo(attn).realize(), cache_k.realize(), cache_v.realize()

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
  def __init__(self, dim, multiple_of, n_heads, n_kv_heads, norm_eps, linear=Linear, ffn_dim_multiplier=None):
    self.attention = Attention(dim, n_heads, n_kv_heads, linear)
    self.feed_forward = FeedForward(dim, 4*dim, multiple_of, linear, ffn_dim_multiplier)
    self.attention_norm = RMSNorm(dim, norm_eps)
    self.ffn_norm = RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, cache_k:Optional[Tensor], cache_v:Optional[Tensor], start_pos:int, freqs_cis:Tensor, mask:Optional[Tensor], jit_ctx:Optional[Dict[Variable,int]]=None):
    bsz, seqlen, _ = x.shape
    if JIT and mask is None:
      assert cache_k is not None and cache_v is not None, "no cache"
      pos = Variable("pos", 1, 1024).bind(start_pos)
      cache_k = cache_k.reshape(cache_k.shape[0], pos, cache_k.shape[2], cache_k.shape[3])
      cache_v = cache_v.reshape(cache_v.shape[0], pos, cache_v.shape[2], cache_v.shape[3])

    output, cache_k, cache_v = self.attention(self.attention_norm(x), cache_k, cache_v, start_pos, freqs_cis, mask, jit_ctx=jit_ctx)
    h = x + output
    return (h + self.feed_forward(self.ffn_norm(h))).realize(), cache_k.realize(), cache_v.realize()

class Transformer:
  def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, linear=Linear, max_batch_size=32, max_seq_len=1024, ffn_dim_multiplier=None, n_kv_heads=None, rope_theta=10000):
    self.layers = [TransformerBlock(dim, multiple_of, n_heads, n_kv_heads, norm_eps, linear, ffn_dim_multiplier) for _ in range(n_layers)]
    self.kv_caches = [(None, None) for _ in range(n_layers)]
    self.norm = RMSNorm(dim, norm_eps)
    self.tok_embeddings = Embedding(vocab_size, dim)
    self.output = linear(dim, vocab_size, bias=False)
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2, rope_theta)
    self.norm_output = lambda x: self.output(self.norm(x))

    self.tok_embeddings_jitted = TinyJit(lambda x: self.tok_embeddings(x).realize())
    self.postprocess_jitted = TinyJit(self.postprocess)
    self.layers_jitted = [TinyJit(layer.__call__) for layer in self.layers]

  def postprocess(self, x, temperature:Optional[float]):
    logits = self.output(self.norm(x))
    if temperature is not None: return (logits[:, -1, :] / (temperature+1e-10)).softmax().flatten().realize()
    return logits.realize()

  def __call__(self, tokens:Tensor, start_pos:int, temperature:Optional[float]=None):
    _bsz, seqlen = tokens.shape
    if seqlen == 1 and start_pos > 0 and JIT:
      pos = Variable("pos", 1, 1024).bind(start_pos)
      # get only the part of freqs_cis that we are using.
      freqs_cis = self.freqs_cis.shrink(((0, self.freqs_cis.shape[0]), (pos, pos+seqlen),(0, self.freqs_cis.shape[2]),(0, self.freqs_cis.shape[3]),(0, self.freqs_cis.shape[4])))
      h = self.tok_embeddings_jitted(tokens)
      for i, (layer, (cache_k, cache_v)) in enumerate(zip(self.layers_jitted, self.kv_caches)):
        h, cache_k, cache_v = layer(h, cache_k, cache_v, start_pos=start_pos, freqs_cis=freqs_cis, mask=None, jit_ctx={pos.unbind()[0]: start_pos})
        self.kv_caches[i] = (cache_k, cache_v)
      return self.postprocess_jitted(h, temperature)
    else:
      freqs_cis = self.freqs_cis.shrink(((0, self.freqs_cis.shape[0]), (start_pos, start_pos+seqlen),(0, self.freqs_cis.shape[2]),(0, self.freqs_cis.shape[3]),(0, self.freqs_cis.shape[4])))
      mask = Tensor.full((1, 1, seqlen, start_pos + seqlen), float("-inf"), dtype=dtypes.float32).triu(start_pos+1).realize()
      h = self.tok_embeddings(tokens)
      for i, (layer, (cache_k, cache_v)) in enumerate(zip(self.layers, self.kv_caches)):
        # need this reshape back to int shape in conversational mode because jitted and unjitted calls share the same cache
        if cache_k is not None and start_pos > 0:
          cache_k = cache_k.reshape(cache_k.shape[0], start_pos, cache_k.shape[2], cache_k.shape[3])
          cache_v = cache_v.reshape(cache_v.shape[0], start_pos, cache_v.shape[2], cache_v.shape[3])
        h, cache_k, cache_v = layer(h, cache_k, cache_v, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        self.kv_caches[i] = (cache_k, cache_v)
      return self.postprocess(h, temperature)

# **** files and arguments ****
MODEL_PARAMS = {
  "1": {
    "7B": {
      "args": {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": 32000},
      "files": 2,
    },
    "30B": {
      "args": {"dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": 32000},
      "files": 4,
    },
    "65B": {
      "args": {"dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000},
      "files": 8,
    },
  },
  "2": {
    "7B": {
      "args": {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000},
      "files": 2,
    },
    "70B": {
      "args": {"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000},
      "files": 8,
    },
  },
  "code": {
    "7B": {
      "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32016},
      "files": 1,
    },
    "7B-Python": {
      "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32000},
      "files": 1,
    },
    "7B-Instruct": {
      "args": {"dim": 4096, "n_layers": 32, "n_heads": 32, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32016},
      "files": 1,
    },
    "13B": {
      "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32016},
      "files": 2,
    },
    "13B-Python": {
      "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32000},
      "files": 2,
    },
    "13B-Instruct": {
      "args": {"dim": 5120, "n_layers": 40, "n_heads": 40, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32016},
      "files": 2,
    },
    "34B": {
      "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32000},
      "files": 4,
    },
    "34B-Python": {
      "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32000},
      "files": 4,
    },
    "34B-Instruct": {
      "args": {"dim": 8192, "n_layers": 48, "n_heads": 64, "n_kv_heads": 8, "multiple_of": 256, "ffn_dim_multiplier": 1.0, "norm_eps": 1e-5, "rope_theta": 1000000, "vocab_size": 32000},
      "files": 4,
    },
  }
}

# **** helper functions ****
def concat_weights(models):
  def convert(name) -> Tensor:
    disk_tensors = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=Device.DEFAULT)
    axis = 1 if name.startswith("tok_embeddings.") or name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=Device.DEFAULT) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)

def convert_from_huggingface(weights, model):
  keymap = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attention_norm.weight" for l in range(len(model.layers))},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attention.w{x}.weight" for x in ["q", "k", "v", "o"] for l in range(len(model.layers))},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(len(model.layers))},
    **{f"model.layers.{l}.mlp.{x}_proj.weight": f"layers.{l}.feed_forward.w{y}.weight" for x, y in {"gate": "1", "down": "2", "up": "3"}.items() for l in range(len(model.layers))},
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
  }
  return {keymap[k]: v for k,v in weights.items() if ".rotary_emb." not in k}

class AbsmaxQuantizedLinear:
  def __init__(self, in_features, out_features, bias=False):
    assert bias == False
    self.weight = Tensor.ones(out_features, in_features, dtype=dtypes.int8)
    self.scale = Tensor.ones(out_features, dtype=dtypes.half)

  def __call__(self, x):
    return x.dot(self.weight.cast(dtype=dtypes.half).T*self.scale)

  @staticmethod
  def quantize(tensors):
    new_tensors = {}
    for name,v in tensors.items():
      if "feed_forward" in name or ("attention.w") in name or name == "output.weight":
        scale = v.abs().max(axis=1) / 127.0
        int8_weight = (v.T/scale).T.cast(dtype=dtypes.int8)
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
      else:
        new_tensors[name] = v
    return new_tensors

class LLaMa:
  @staticmethod
  def build(model_path, tokenizer_path, model_gen="1", model_size="7B", quantize=False):
    from sentencepiece import SentencePieceProcessor
    sp_model = SentencePieceProcessor(model_file=str(tokenizer_path))
    assert sp_model.vocab_size() == MODEL_PARAMS[model_gen][model_size]["args"]["vocab_size"], f"{sp_model.vocab_size()=} not equal to {MODEL_PARAMS[model_gen][model_size]['args']['vocab_size']}"

    params = MODEL_PARAMS[model_gen][model_size]
    model = Transformer(**params["args"], linear=AbsmaxQuantizedLinear) if quantize else Transformer(**params["args"])

    if model_path.is_dir():
      weights = concat_weights([load(filename) for filename in [f"{model_path}/consolidated.{i:02d}.pth" for i in range(params["files"])]])
    else:
      weights = load(str(model_path))
    if "model.embed_tokens.weight" in weights:
      weights = convert_from_huggingface(weights, model)

    if quantize:
      weights = AbsmaxQuantizedLinear.quantize(weights)
      for _,v in weights.items(): v.realize()
    load_state_dict(model, weights, strict=False)

    return LLaMa(model, sp_model)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def greedy_until(self, prompt:str, until, max_length, temperature):
    toks = [self.tokenizer.bos_id()] + self.tokenizer.encode(prompt)
    start_pos = 0
    for i in range(max_length):
      probs = llama.model(Tensor([toks[start_pos:]]), start_pos, args.temperature).realize()
      probs_np = probs.numpy()
      tok = int(np.random.choice(len(probs_np), p=probs_np))
      start_pos = len(toks)
      toks.append(tok)

      if tok == self.tokenizer.eos_id(): break
      output = self.tokenizer.decode(toks)
      for s in until:
        if output.endswith(s): return output[0:-len(s)]
    return output

# **** main code ****
"""
test:
python3 examples/llama.py  --temperature=0 --count=50 --prompt="Hello."
output:
Hello. I'm a 20 year old male. I'm a student at the University of Texas at Austin. I'm a sophomore majoring in Computer Science.

test:
python3 examples/llama.py --gen='2' --temperature=0 --count=50 --prompt="Hello."
output:
Hello. I'm a 20 year old girl who is looking for a good lay in Palm Coast. I don't care whether it's at your place or not, as long as it's clean.

test:
python3 examples/llama.py --gen="code" --temperature=0.2 --count=50 --prompt="\
import argparse

def main(string: str):
    print(string)
    print(string[::-1])

if __name__ == "__main__":"
output:
    parser = argparse.ArgumentParser()
    parser.add_argument('string', type=str, help='string to be reversed')
    args = parser.parse_args()
    main(args.string)

test:
python3 examples/llama.py --gen="code" --size="7B-Python" --temperature=0.2 --count=70 --prompt="def add_elements(arr,k):"
output:
    for i in range(len(arr)):
        arr[i] += k
    return arr


arr = [1, 2, 3, 4, 5]
k = 2
print(add_elements(arr, k))

test:
python3 examples/llama.py --gen="code" --size="7B-Instruct" --temperature=0.2 --count=120 --prompt="write a function in c++ that adds three float numbers"
output:
\begin{code}
#include<iostream>
using namespace std;

float add(float a, float b, float c)
{
    return a+b+c;
}

int main()
{
    float a, b, c;
    cout<<"Enter three numbers: ";
    cin>>a>>b>>c;
    cout<<"The sum is: "<<add(a,b,c);
    return 0;
}
\end{code}
"""
if __name__ == "__main__":
  Tensor.no_grad = True
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description="Run LLaMA in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--prompt", type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument("--count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--personality", type=str, default="Stacy", help="Personality, can be Stacy, George, Gary, or Lexie")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Output profile data to out.prof")
  parser.add_argument("--size", type=str, default="7B", help="Size of model to use [7B, 13B, 30B, 65B] for Gen 1, [7B, 13B, 70B] for Gen 2, [7B, 13B, 34B] for Code LLaMA")
  parser.add_argument("--gen", default="1", help="Generation of the model to use ['1', '2', 'code']")
  parser.add_argument("--quantize", action="store_true", help="Quantize the weights to int8 in memory")
  parser.add_argument("--model", type=Path, default=None, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")

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

  LLAMA_SUFFIX = {"1": "", "2": "-2", "code": "-code"}[args.gen]
  MODEL_PATH = args.model or Path(__file__).parents[1] / f"weights/LLaMA{LLAMA_SUFFIX}/{args.size}"
  TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent) / "tokenizer.model"
  print(f"using LLaMA{LLAMA_SUFFIX}-{args.size} model")
  llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_gen=args.gen, model_size=args.size, quantize=args.quantize)

  if chatbot:
    # encode pre prompt
    toks = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(pre_prompt)

    print(f"Preparing KV cache for chatbot with personality {args.personality}...")
    with Timing():
      llama.model(Tensor([toks]), 0, args.temperature).realize()  # NOTE: output logits are not used
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
      GlobalCounters.reset()
      if args.profile and i == 2: profiler.enable()

      if args.timing: print("")
      st = GlobalCounters.time_sum_s
      with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/sec"):
        with Timing("ran model in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU" if DEBUG>=2 else "")+
                    f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                    (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):
          probs = llama.model(Tensor([toks[start_pos:]]), start_pos, args.temperature).realize()
        probs_np = probs.numpy()
        tok = int(np.random.choice(len(probs_np), p=probs_np))

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
    stats.dump_stats("out.prof")
