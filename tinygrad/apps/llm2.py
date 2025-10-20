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

from icecream import install
install()

# MODELS:dict[str, dict[str, int|str|dict[str, int|float, dict[str, int|float]]]] = {
#   "20B": {
#     "params": {"dim": 2880, "hidden_dim": 2880, "head_dim": 64, "n_heads": 64, "n_kv_heads": 8, "num_blocks": 24, "n_experts": 32, "n_active_experts": 4,
#                "norm_eps": 1e-5, "vocab_size": 201088, "sliding_window": 128, "max_context": 4096,
#                "rope_params": {"theta": 150000, "scale": 32.0, "ntk_alpha": 1.0, "ntk_beta": 32.0, "initial_context_length": 4096},
#                },
#     "total_num_weights": 3,
#     "model": "unsloth/gpt-oss-20b-GGUF",
#     "tokenizer": "unsloth/gpt-oss-20b-GGUF",
#   }
# }

MODELS = {
  "20B-Q6_K": "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q6_K.gguf",
  "20B-Q8_0": "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q8_0.gguf",
}

# ****** model architecture *****

def precompute_freqs_cis(dim:int, end:int, theta:float = 10000.0, scale:float=1.0, ntk_alpha:float=1, ntk_beta:float=32, initial_context_length:int=4096) -> Tensor:
  half = dim // 2
  freqs = (theta ** (-(Tensor.arange(half, dtype=dtypes.float32) / half)))[None, :]

  def _angles(freqs, mscale=1): return (Tensor.arange(end, dtype=dtypes.float32)[:, None] * Tensor.stack(freqs.cos()*mscale, freqs.sin()*mscale, dim=-1)).reshape(1, end, 1, half, 2)
  def _ratio(ntk): return half * math.log(initial_context_length / (ntk * 2 * math.pi)) / math.log(theta)

  # rope https://arxiv.org/pdf/2104.09864
  if scale <= 1: return _angles(freqs)

  # yarn https://arxiv.org/pdf/2309.00071
  low, high = _ratio(ntk_alpha), _ratio(ntk_beta)
  interpolation, extrapolation = freqs, freqs / scale
  ramp = (Tensor.arange(half, dtype=dtypes.float32) - low) / (high - low)
  mask = 1 - ramp.clamp(0, 1)
  freqs = interpolation * (1 - mask) + extrapolation * mask
  return _angles(freqs, 0.1 * math.log(scale) + 1.0)

# matches meta, non hugging face weights
# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rope3(x:Tensor, start_pos:int|UOp, base:float = 10000.0) -> Tensor:
  B, H, T, Hd = x.shape
  assert (Hd & 1) == 0, "RoPE requires an even head dimension"
  half = Hd // 2
  freqs = (base ** (-(Tensor.arange(half, dtype="float32") / half)))[None, :]

  def _angles(freqs, mscale=1.0):
    angles =((Tensor.arange(T, dtype=dtypes.float32) + start_pos)[:, None] * freqs).reshape(1, 1, T, half)
    return angles.cos().cast(x.dtype).contiguous(), angles.sin().cast(x.dtype).contiguous()
  angles = _angles(freqs)
  complex_mult(angles.cos())

  # contiguous here allows RoPE to be pruned in the JIT
  cos, sin = angles.cos().cast(x.dtype).contiguous(), angles.sin().cast(x.dtype).contiguous()
  x_pairs = x.reshape(B, H, T, half, 2)
  return Tensor.stack(x_pairs[..., 0] * cos - x_pairs[..., 1] * sin,
                      x_pairs[..., 0] * sin + x_pairs[..., 1] * cos, dim=-1).reshape(B, H, T, Hd)

def apply_rope2(x:Tensor, start_pos:int|UOp, base:float=10000.0) -> Tensor:
  B, H, T, Hd = x.shape
  assert (Hd & 1) == 0, "RoPE requires an even head dimension"
  half = Hd // 2

  def _angles(freqs, mscale=1):
    angles = ((Tensor.arange(T, dtype=dtypes.float32) + start_pos)[:, None] * freqs * mscale).reshape(1, 1, T, half)
    # contiguous here allows RoPE to be pruned in the JIT
    cos, sin = angles.cos().cast(x.dtype).contiguous(), angles.sin().cast(x.dtype).contiguous()

    x_pairs = x.reshape(B, H, T, half, 2)
    return Tensor.stack(x_pairs[..., 0] * cos - x_pairs[..., 1] * sin,
                        x_pairs[..., 0] * sin + x_pairs[..., 1] * cos, dim=-1).reshape(B, H, T, Hd)
  def _ratio(ntk): return half * math.log(initial_context_length / (ntk * 2 * math.pi)) / math.log(theta)

  # rope https://arxiv.org/pdf/2104.09864
  freqs = (theta ** (-(Tensor.arange(half, dtype=dtypes.float32) / half)))[None, :]
  if scale <= 1: return _angles(freqs)

  # yarn https://arxiv.org/pdf/2309.00071
  low, high = _ratio(ntk_alpha), _ratio(ntk_beta)
  interpolation, extrapolation = freqs, freqs / scale
  ramp = (Tensor.arange(half, dtype=dtypes.float32) - low) / (high - low)
  mask = 1 - ramp.clamp(0, 1)
  freqs = interpolation * (1 - mask) + extrapolation * mask
  return _angles(freqs, 0.1 * math.log(scale) + 1.0)


def apply_rope(x:Tensor, start_pos:int|UOp, base:float=10000.0) -> Tensor:
  B, H, T, Hd = x.shape
  assert (Hd & 1) == 0, "RoPE requires an even head dimension"
  half = Hd // 2
  angles = (Tensor.arange(T, dtype="float32") + start_pos)[:, None] * (base ** (-(Tensor.arange(half, dtype="float32") / half)))[None, :]
  # contiguous here allows RoPE to be pruned in the JIT
  cos, sin = angles.cos().reshape(1, 1, T, half).cast(x.dtype).contiguous(), angles.sin().reshape(1, 1, T, half).cast(x.dtype).contiguous()
  x_pairs = x.reshape(B, H, T, half, 2)
  return Tensor.stack(x_pairs[..., 0] * cos - x_pairs[..., 1] * sin,
                      x_pairs[..., 0] * sin + x_pairs[..., 1] * cos, dim=-1).reshape(B, H, T, Hd)

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
    self.attn_sinks         = Tensor.empty(n_heads)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm          = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm           = nn.RMSNorm(dim, norm_eps)

    # --- MoE feed-forward ------------------------------------------------
    self.n_active_experts   = n_active_experts
    self.ffn_gate           = nn.Linear(dim, n_experts, bias=True)
    self.ffn_gate_proj      = Tensor.zeros(n_experts, hidden_dim, dim, dtype='bfloat16')
    self.ffn_up_proj        = Tensor.zeros(n_experts, hidden_dim, dim, dtype='bfloat16') # todo: Tensor.empty? remove dtype?
    self.ffn_down_proj      = Tensor.zeros(n_experts, dim, hidden_dim, dtype='bfloat16')
    self.ffn_gate_proj_bias = Tensor.zeros(n_experts, hidden_dim, dtype='bfloat16')
    self.ffn_up_proj_bias   = Tensor.zeros(n_experts, hidden_dim, dtype='bfloat16') # todo: Tensor.empty? remove dtype?
    self.ffn_down_proj_bias = Tensor.zeros(n_experts, dim, dtype='bfloat16')


  def _feed_forward(self, x:Tensor) -> Tensor:
    assert x.shape[0] == 1, "only BS=1"
    assert x.shape[1] == 1, "only length=1"
    g = self.ffn_gate(x).softmax(-1)

    g = g.squeeze() # (BS, length, num_experts) -> (num_experts,)
    probs, sel = g.topk(self.n_active_experts)

    # run MoE
    # todo: add bias
    x_up_gate = x.dot(self.ffn_gate_proj[sel].permute(0,2,1)).silu() * x.dot(self.ffn_up_proj[sel].permute(0,2,1))
    x_down = x_up_gate.dot(self.ffn_down_proj[sel].permute(0,2,1))
    return (x_down * probs.reshape(self.n_active_experts, 1, 1)).sum(axis=0)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)

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

    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    attn = self.attn_o(attn)
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

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None) -> tuple[Transformer, dict]:
    kv, state_dict = nn.state.gguf_load(gguf)
    state_dict = convert_from_gguf(state_dict, kv[f"{kv['general.architecture']}.block_count"])

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    params = dict(
      dim=kv[f'{arch}.embedding_length'], hidden_dim=kv[f'{arch}.feed_forward_length'], head_dim=kv[f'{arch}.attention.key_length'],
      num_blocks=kv[f'{arch}.block_count'], n_heads=kv[f'{arch}.attention.head_count'], n_kv_heads=kv[f'{arch}.attention.head_count_kv'],
      n_experts=kv[f'{arch}.expert_count'], n_active_experts=kv[f'{arch}.expert_used_count'], norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']), sliding_window=kv[f'{arch}.attention.sliding_window'], max_context=max_context,
      rope_params=dict(
        ntk_alpha=1.0, ntk_beta=kv[f'{arch}.rope.scaling.factor'], scale=kv[f'{arch}.rope.scaling.factor'],
        theta=kv[f'{arch}.rope.freq_base'], initial_context_length=kv[f'{arch}.rope.scaling.original_context_length'],
        ),
    )
    model = Transformer(**params)
    load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    for s in nn.state.get_parameters(model): s.replace(s.contiguous())
    return model, kv

# ***** model loading *****

def download_weights(model:str, total_num_weights:int) -> Path:
  model_path = fetch(f"https://huggingface.co/{model}/resolve/main/model.safetensors.index.json", "model.safetensors.index.json", subdir=(subdir:=model.split('/')[-1]))
  for i in range(total_num_weights):
    filename = f"model-{i:05d}-of-{total_num_weights-1:05d}.safetensors"
    fetch(f"https://huggingface.co/{model}/resolve/main/{filename}?download=true", filename, subdir=subdir)
  return Path(os.path.dirname(model_path))

def convert_from_gguf(weights:dict[str, Tensor], num_blocks:int):
  # map gguf to tinygrad
  keymap = {
    **{f"blk.{l}.attn_sinks.weight": f"blk.{l}.attn_sinks" for l in range(num_blocks)},
    **{f"blk.{l}.attn_output.{p}": f"blk.{l}.attn_o.{p}" for p in ["weight", "bias"] for l in range(num_blocks)},
    **{f"blk.{l}.post_attention_norm.weight": f"blk.{l}.ffn_norm.weight" for l in range(num_blocks)},
    **{f"blk.{l}.ffn_gate_inp.{p}": f"blk.{l}.ffn_gate.{p}" for p in ["weight", "bias"] for l in range(num_blocks)},
    **{f"blk.{l}.ffn_gate_exps.weight": f"blk.{l}.ffn_gate_proj" for l in range(num_blocks)},
    **{f"blk.{l}.ffn_up_exps.weight": f"blk.{l}.ffn_up_proj" for l in range(num_blocks)},
    **{f"blk.{l}.ffn_down_exps.weight": f"blk.{l}.ffn_down_proj" for l in range(num_blocks)},
    **{f"blk.{l}.ffn_gate_exps.bias": f"blk.{l}.ffn_gate_proj_bias" for l in range(num_blocks)},
    **{f"blk.{l}.ffn_up_exps.bias": f"blk.{l}.ffn_up_proj_bias" for l in range(num_blocks)},
    **{f"blk.{l}.ffn_down_exps.bias": f"blk.{l}.ffn_down_proj_bias" for l in range(num_blocks)},
  }

  sd = {}
  for k, v in weights.items():
    if ".rotary_emb." in k: continue
    v = v.to(Device.DEFAULT)
    sd[keymap.get(k, k)] = v
  return sd


def convert_from_huggingface(weights:dict[str, Tensor], num_blocks: int, n_heads: int, n_kv_heads: int, permute_layers: bool = True):
  # huggingface stores Q and K permuted! it is mostly correct without this, but without it makes RoPE different, so it will diverge after 10+ toks.
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

  # map hf to tinygrad
  keymap = {
    "model.embed_tokens.weight": "token_embd.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"layers.{l}.attn_norm.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"layers.{l}.attn_{x}.weight" for x in ["q", "k", "v", "o"] for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.{x}_proj.bias": f"layers.{l}.attn_{x}.bias" for x in ["q", "k", "v", "o"] for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.sinks": f"layers.{l}.attn_sinks" for l in range(num_blocks)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"layers.{l}.ffn_norm.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.router.weight": f"layers.{l}.router.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.router.bias": f"layers.{l}.router.bias" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d1}_proj_bias": f"layers.{l}.ffn_{d2}_proj_bias" for d1, d2 in {"gate_up": "up", "down":"down"}.items() for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d1}_proj_blocks": f"layers.{l}.ffn_{d2}_proj_blocks" for d1, d2 in {"gate_up": "up", "down":"down"}.items() for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d1}_proj_scales": f"layers.{l}.ffn_{d2}_proj_scales" for d1, d2 in {"gate_up": "up", "down":"down"}.items() for l in range(num_blocks)},
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
    # ic(blocks, scales, mxfp4_data)
    return ggml_data_to_tensor(mxfp4_data, scales.numel() * 32, 39).reshape(*scales.shape[:2], -1)

  # dequantize only the ffn_up_proj and ffn_down_proj
  for l in range(num_blocks):
    for d in ['up', 'down']:
      blocks = f'layers.{l}.ffn_{d}_proj_blocks'
      scales = f'layers.{l}.ffn_{d}_proj_scales'
      proj = dequantize_mxfp4(weights.pop(blocks), weights.pop(scales))
      # ic(proj)
      weights[f'layers.{l}.ffn_{d}_proj'] = proj
    return weights

def main(args):

  if args.seed is not None: Tensor.manual_seed(args.seed)

  if getenv("TORCH"):
    print("Using torch, not tinygrad.")
    import torch
    from transformers import GptOssForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using {device=}")

    model_info = MODELS[args.size]
    model_path = Path(args.weights) if args.weights or args.fakeweights else download_weights(model_info["model"], model_info["total_num_weights"])
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"], cache_dir=model_path)
    fetch(f"https://huggingface.co/{model_info['model']}/resolve/main/config.json", "config.json", subdir=model_info["model"].split('/')[-1])
    model = GptOssForCausalLM.from_pretrained(model_path, load_in_8bit=True, local_files_only=True, cache_dir=model_path, device_map=device)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    generate_ids = model.generate(input_ids, max_new_tokens=args.count) # tensor([[12194,    11,   357,   939,   261]], device='cuda:0')
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
    return


  # build model
  # model = Transformer.from_pretrained(model_path, model_info["params"], args.fakeweights)
  model, kv = Transformer.from_gguf(Tensor.from_url(MODELS[args.size]), args.count)
  tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

  ic(model.token_embd.weight.numpy())

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
