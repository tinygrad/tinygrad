# install
# pip install -e .["testing"]
# pip install accelerate kernels # needed for mxfp4 quantization

# implementations
# openai - https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
# hf - https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py
# unsloth - https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training#introducing-unsloth-flex-attention-support
# unsloth - https://docs.unsloth.ai/models/gpt-oss-how-to-run-and-fine-tune

# blogs
# https://huggingface.co/blog/faster-transformers
# https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the
# https://cameronrwolfe.substack.com/p/gpt-oss

from __future__ import annotations
import argparse, functools, math, os
from pathlib import Path
from tinygrad import Tensor, TinyJit, UOp, nn, dtypes, Device
from tinygrad.helpers import fetch, getenv
from tinygrad.nn.state import load_state_dict, ggml_data_to_tensor, get_parameters, get_state_dict
from examples.llama3 import load
from extra.models.llama import fix_bf16
from transformers import AutoTokenizer

from icecream import install, ic
install()

def fix(x): return x.cast('float').numpy()

MODELS:dict[str, dict[str, int|str|dict[str, int|float, dict[str, int|float]]]] = {
  "20B": {
    "params": {"dim": 2880, "hidden_dim": 2880, "head_dim": 64, "n_heads": 64, "n_kv_heads": 8, "num_blocks": 24, "n_experts": 32, "n_active_experts": 4,
               "norm_eps": 1e-5, "vocab_size": 201088, "sliding_window": 128, "max_context": 4096,
               "rope_params": {"base": 150000, "scale": 32.0, "ntk_alpha": 1.0, "ntk_beta": 32.0, "initial_context_length": 4096},
               },
    "total_num_weights": 3,
    "model": "openai/gpt-oss-20b",
    "tokenizer": "openai/gpt-oss-20b",
  }
}

# ****** model architecture *****

def apply_rope(x:Tensor, start_pos:int|UOp, base:int=150_000, scale:float=32.0, ntk_alpha:float=1, ntk_beta:float=32, initial_context_length:int=4096) -> Tensor:
  B, H, T, Hd = x.shape
  assert (Hd & 1) == 0, "RoPE requires an even head dimension"
  half = Hd // 2
  freqs = (base ** (-(Tensor.arange(half, dtype="float32") / half)))

  def rotate(x_pairs, freqs):
    angles = ((Tensor.arange(T, dtype="float32") + start_pos)[:, None] * freqs[None, :]).reshape(1, 1, T, half)
    # contiguous here allows RoPE to be pruned in the JIT
    cos, sin = angles.cos().cast(x_pairs.dtype).contiguous(), angles.sin().cast(x_pairs.dtype).contiguous() # todo: cast to float32 ??
    return Tensor.stack(x_pairs[..., 0] * cos - x_pairs[..., 1] * sin, x_pairs[..., 0] * sin + x_pairs[..., 1] * cos, dim=-1)

  # rope https://arxiv.org/pdf/2104.09864
  if scale <= 1:
    return rotate(x.reshape(B, H, T, 2, half).transpose(-1, -2), freqs).transpose(-1, -2).reshape(B, H, T, Hd)

  # yarn https://arxiv.org/pdf/2309.00071
  attn_scale = 0.1 * math.log(scale) + 1.0
  def _ratio(ntk): return half * math.log(initial_context_length / (ntk * 2 * math.pi)) / math.log(base)
  low, high = _ratio(ntk_beta), _ratio(ntk_alpha)
  interpolation, extrapolation = freqs / scale, freqs
  ramp = (Tensor.arange(half, dtype=dtypes.float32) - low) / (high - low)
  mask = 1 - ramp.clamp(0, 1)
  freqs = interpolation * (1 - mask) + extrapolation * mask
  return rotate(x.reshape(B, H, T, 2, half).transpose(-1, -2) * attn_scale, freqs).transpose(-1, -2).reshape(B, H, T, Hd)


# arxiv.org/pdf/2002.05202v1
def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu = x[..., ::2].clamp(max_=limit)
    x_linear = x[..., 1::2].clamp(-limit, limit)
    return x_glu * (alpha * x_glu).sigmoid() * (x_linear + 1)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, head_dim:int, n_heads:int, n_kv_heads:int, n_experts:int, n_active_experts:int, norm_eps:float, max_context:int=0, sliding_window:int=0,):
    self.n_heads            = n_heads
    self.n_kv_heads         = n_kv_heads
    self.head_dim           = head_dim if head_dim else dim // n_heads
    self.sliding_window     = sliding_window
    self.max_context        = max_context

    # --- attention projections (linear with bias) ------------------------
    self.attn_q             = nn.Linear(dim, n_heads * head_dim,    bias=True)
    self.attn_k             = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
    self.attn_v             = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
    self.attn_o             = nn.Linear(n_heads * head_dim, dim,    bias=True)
    self.attn_sink         = Tensor.empty(n_heads)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm          = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm           = nn.RMSNorm(dim, norm_eps)

    # --- MoE feed-forward ------------------------------------------------
    self.n_active_experts   = n_active_experts
    self.ffn_gate           = nn.Linear(dim, n_experts, bias=True)
    self.ffn_up_proj        = Tensor.zeros(n_experts, hidden_dim * 2, dim, dtype='bfloat16') # todo: Tensor.empty? remove dtype?
    self.ffn_down_proj      = Tensor.zeros(n_experts, dim, hidden_dim, dtype='bfloat16')
    self.ffn_up_proj_bias   = Tensor.zeros(n_experts, hidden_dim * 2, dtype='bfloat16') # todo: Tensor.empty? remove dtype?
    self.ffn_down_proj_bias = Tensor.zeros(n_experts, dim, dtype='bfloat16')


  def _feed_forward(self, x: Tensor) -> Tensor:
    assert x.shape[0] == 1 and x.shape[1] == 1, "expected BS=1 and seqlen=1 but got BS={x.shape[0]} and seqlen={x.shape[1]}"

    # Select top-k experts
    g = self.ffn_gate(x).softmax(-1) # (B,T,D) -> (B,T,E)
    g = g.squeeze() # (B,T,E) -> (E,)
    probs, sel = g.topk(self.n_active_experts) # (E,) -> (E,) (E,)

    # reshape
    w1 = self.ffn_up_proj[sel].unsqueeze(0) # (1,E,D2,D)
    w2 = self.ffn_down_proj[sel].unsqueeze(0) # (1,E,D,D)
    b1 = self.ffn_up_proj_bias[sel].unsqueeze(0) # (1,E,D2)
    b2 = self.ffn_down_proj_bias[sel].unsqueeze(0) # (1,E,D)

    # MLP forward
    t = x.squeeze(1)  # (B,T,D) -> (B,1,T,D)
    t = swiglu(Tensor.einsum("bk,beck->bec", t, w1) + b1)  # (B,1,T,D) (1,E,D2,D) -> ?
    t = Tensor.einsum("bek,beck->bec", t, w2) + b2  # (1, 4, 2880)

    # Weighted sum over experts
    return (t * probs.reshape(1, -1, 1)).sum(1, keepdim=True)  # (1, 1, 2880)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm) # (B,T,D)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)     # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)     # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)     # (B,KvH,T,Hd)
    s = self.attn_sink.reshape(1, -1, 1, 1).expand(B, self.head_dim, T, 1)  # (B,H,T,1)

    q = apply_rope(q, start_pos)
    k = apply_rope(k, start_pos)

    # TODO: remove these kv cache realizes
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim, dtype=k.dtype, device=k.device).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()  # type: ignore
    k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if T > 1 else None
    if self.sliding_window:
     sliding_mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).tril(-self.sliding_window)
     mask = sliding_mask if mask is None else mask+sliding_mask

    attn = q.scaled_dot_product_attention(k, v, sink=s, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    attn = self.attn_o(attn)
    1/0
    return x + attn

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

class Transformer:
  def __init__(self, dim, hidden_dim, head_dim, num_blocks, n_heads, n_kv_heads, n_experts, n_active_experts, norm_eps, rope_params, vocab_size, sliding_window, max_context):
    self.blk          = [TransformerBlock(dim, hidden_dim, head_dim, n_heads, n_kv_heads, n_experts, n_active_experts, norm_eps, max_context) for _ in range(num_blocks)]
    self.token_embd   = nn.Embedding(vocab_size, dim)
    self.output_norm  = nn.RMSNorm(dim, norm_eps)
    self.output       = nn.Linear(dim, vocab_size, bias=False)
    self.max_context  = max_context
    # JIT is used if T=1 and start_pos is a UOp. TODO: make this not needed by including T in the JIT and making start_pos always a UOp
    self.forward_jit  = TinyJit(self.forward)

    # add sliding attention to all even layers
    for i in range(0, num_blocks, 2): self.blk[i].sliding_window = sliding_window

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens, start_pos)

  @staticmethod
  def from_pretrained(model_path:Path, params:dict[str, int|float], fakeweights:bool) -> Transformer:
    # load model
    model = Transformer(**params)

    # load weights
    if not fakeweights:
      weights = load(str(model_path / "model.safetensors.index.json"))
      weights = convert_from_huggingface(weights, params["num_blocks"], params["n_heads"], params["n_kv_heads"], permute_layers=False)
      weights = fix_mxfp4(weights, params["num_blocks"])
      # weights = fix_bf16(weights) # todo: do we need ??
      load_state_dict(model, weights, strict=False, consume=True)
    return model

# ***** model loading *****

def download_weights(model:str, total_num_weights:int) -> Path:
  model_path = fetch(f"https://huggingface.co/{model}/resolve/main/model.safetensors.index.json", "model.safetensors.index.json", subdir=(subdir:=model.split('/')[-1]))
  for i in range(total_num_weights):
    filename = f"model-{i:05d}-of-{total_num_weights-1:05d}.safetensors"
    fetch(f"https://huggingface.co/{model}/resolve/main/{filename}?download=true", filename, subdir=subdir)
  return Path(os.path.dirname(model_path))

def convert_from_huggingface(weights:dict[str, Tensor], num_blocks: int, n_heads: int, n_kv_heads: int, permute_layers: bool = True):
  # huggingface stores Q and K permuted! it is mostly correct without this, but without it makes RoPE different, so it will diverge after 10+ toks.
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

  # map hf to tinygrad
  keymap = {
    "model.embed_tokens.weight": "token_embd.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"blk.{l}.attn_norm.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"blk.{l}.attn_{x}.weight" for x in ["q", "k", "v", "o"] for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.{x}_proj.bias": f"blk.{l}.attn_{x}.bias" for x in ["q", "k", "v", "o"] for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.sinks": f"blk.{l}.attn_sink" for l in range(num_blocks)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"blk.{l}.ffn_norm.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.router.weight": f"blk.{l}.ffn_gate.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.router.bias": f"blk.{l}.ffn_gate.bias" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d1}_proj_bias": f"blk.{l}.ffn_{d2}_proj_bias" for d1, d2 in {"gate_up": "up", "down":"down"}.items() for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d1}_proj_blocks": f"blk.{l}.ffn_{d2}_proj_blocks" for d1, d2 in {"gate_up": "up", "down":"down"}.items() for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d1}_proj_scales": f"blk.{l}.ffn_{d2}_proj_scales" for d1, d2 in {"gate_up": "up", "down":"down"}.items() for l in range(num_blocks)},
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
  }

  sd = {}
  for k, v in weights.items():
    if ".rotary_emb." in k: continue
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if "q_proj" in k and permute_layers: v = permute(v, n_heads)
      elif "k_proj" in k and permute_layers: v = permute(v, n_kv_heads)
    sd[keymap[k]] = v
  return sd

def fix_mxfp4(weights, num_blocks) -> Tensor:
  def dequantize_mxfp4(blocks: Tensor, scales: Tensor) -> Tensor:
    """Dequantize MXFP4 to float32. blocks: (*batch, num_blocks, 16), scales: (*batch, num_blocks) -> (*batch, num_blocks*32)"""
    assert blocks.shape[:-1] == scales.shape and blocks.shape[-1] == 16
    mxfp4_data = Tensor.cat(scales.unsqueeze(-1), blocks, dim=-1).flatten()  # interleave and flatten to 1D
    return ggml_data_to_tensor(mxfp4_data, scales.numel() * 32, 39).reshape(*scales.shape[:2], -1)

  # dequantize only the ffn_up_proj and ffn_down_proj
  for l in range(num_blocks):
    for d in ['up', 'down']:
      blocks = f'blk.{l}.ffn_{d}_proj_blocks'
      scales = f'blk.{l}.ffn_{d}_proj_scales'
      proj = dequantize_mxfp4(weights.pop(blocks), weights.pop(scales))
      weights[f'layers.{l}.ffn_{d}_proj'] = proj
  return weights

def main(args):

  if args.seed is not None: Tensor.manual_seed(args.seed)
  args.prompt *= 8
  ic(args.prompt)

  model_info = MODELS[args.size]

  model_path = Path(args.weights) if args.weights or args.fakeweights else download_weights(model_info["model"], model_info["total_num_weights"])
  tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"], cache_dir=model_path)

  if getenv("TORCH"):
    print("Using torch, not tinygrad.")
    import torch
    from transformers import GptOssForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using {device=}")
    fetch(f"https://huggingface.co/{model_info['model']}/resolve/main/config.json", "config.json", subdir=model_info["model"].split('/')[-1])
    model = GptOssForCausalLM.from_pretrained(model_path, local_files_only=True, cache_dir=model_path, device_map="auto")
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    generate_ids = model.generate(input_ids, max_new_tokens=args.count) # tensor([[12194,    11,   357,   939,   261]], device='cuda:0')
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
    return

  # build model
  model = Transformer.from_pretrained(model_path, model_info["params"], args.fakeweights)

  outputted = args.prompt
  start_pos, toks = 0, tokenizer(outputted)["input_ids"]
  print(outputted, end="", flush=True)

  tok_tensor = None
  for i in range(args.count):
    # forward pass
    next_tok = Tensor([toks[start_pos:]]) if tok_tensor is None or (len(toks)-start_pos) > 1 else tok_tensor.reshape(1, 1)
    tok_tensor = model(next_tok, start_pos) # todo: add temperature
    tok = tok_tensor.item()

    # use the kv cache
    start_pos = len(toks)

    # add the new token
    toks.append(tok)

    # display
    cur = tokenizer.decode(toks, skip_special_tokens=True)
    print(cur[len(outputted):], flush=True)
    outputted = cur


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run gpt-oss in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--size", choices=list(MODELS.keys()), default=list(MODELS.keys())[0], help="Model size")
  parser.add_argument("--count", type=int, default=1, help="Max number of tokens to generate")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--prompt", type=str, default="Hi", help="Phrase to start with")
  parser.add_argument("--weights", type=str, default=None, help="Path to the downloaded weights")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument('--fakeweights',  action='store_true', help="Load fake weights")
  args = parser.parse_args()

  main(args)
