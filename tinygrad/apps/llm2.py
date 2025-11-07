# install
# uv pip install -e .["testing"]
# uv pip install accelerate kernels # needed for mxfp4 quantization

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
from tinygrad.helpers import fetch, getenv, DEBUG
from tinygrad.nn.state import load_state_dict, ggml_data_to_tensor, get_parameters, get_state_dict
from examples.llama3 import load
from extra.models.llama import fix_bf16, sample
from transformers import AutoTokenizer

from icecream import install, ic
install()

def fix(x): return x.shape, x.dtype, x.numpy()

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
def swiglu(x:Tensor, alpha: float = 1.702, limit: float = 7.0):
  x_glu, x_up = x[..., ::2].clamp(None, limit), x[..., 1::2].clamp(-limit, limit)
  return x_glu * (alpha * x_glu).sigmoid() * (x_up + 1)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, head_dim:int, n_heads:int, n_kv_heads:int, n_experts:int, n_active_experts:int, norm_eps:float, max_context:int=0, sliding_window:int=0):
    self.n_heads                = n_heads
    self.n_kv_heads             = n_kv_heads
    self.head_dim               = head_dim if head_dim else dim // n_heads
    self.sliding_window         = sliding_window
    self.max_context            = max_context

    # --- attention projections (linear with bias) ------------------------
    self.attn_q                 = nn.Linear(dim, n_heads * head_dim,    bias=True)  # (D,H*Hd)
    self.attn_k                 = nn.Linear(dim, n_kv_heads * head_dim, bias=True)  # (D,Hkv*Hd)
    self.attn_v                 = nn.Linear(dim, n_kv_heads * head_dim, bias=True)  # (D,Hkv*Hd)
    self.attn_o                 = nn.Linear(n_heads * head_dim, dim,    bias=True)  # (H*Dh,D)
    self.attn_sink              = Tensor.empty(n_heads)                             # (H,)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm              = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm               = nn.RMSNorm(dim, norm_eps)

    # --- MoE feed-forward ------------------------------------------------
    self.n_experts              = n_experts
    self.n_active_experts       = n_active_experts                                  # k
    self.ffn_gate               = nn.Linear(dim, n_experts, bias=True)              # (D,E)
    self.ffn_gate_up_proj       = Tensor.empty(n_experts, dim, hidden_dim * 2)      # (E,D,D2*2)
    self.ffn_gate_up_proj_bias  = Tensor.empty(n_experts, hidden_dim * 2)           # (E,D2*2)
    self.ffn_down_proj          = Tensor.empty(n_experts, hidden_dim, dim)          # (E,D2,D)
    self.ffn_down_proj_bias     = Tensor.empty(n_experts, dim)                      # (E,D)

  def _feed_forward(self, x: Tensor) -> Tensor:
    x_norm = self.ffn_norm(x)
    (B, T, D), E = x.shape, self.n_experts

    # Select top-k experts
    x_norm = x_norm.reshape(B*T, D)                                       # (B,T,D)  -> (B*T,D)
    logits, sel = self.ffn_gate(x_norm).topk(self.n_active_experts, -1)   # (B*T,D)  -> (B,T,k), (B,T,k)
    probs = Tensor.zeros(B*T, E).scatter(1, sel, logits.softmax(-1))      # (B*T,k)  -> (B*T,E)

    # run MoE
    x_norm = x_norm.repeat(E, 1).reshape(E, B*T, D)                       # (B*T,D) -> (E,B*T,D)
    probs = probs.transpose(0, 1).reshape(E, B, -1).unsqueeze(-1)         # (B*T,E) -> (E,B,T,1)

    x_up_gate = swiglu(x_norm @ self.ffn_gate_up_proj + self.ffn_gate_up_proj_bias.unsqueeze(1))  # (E,B*T,D) (E,D,D2*2) (E,1,D2*2) -> (E,B*T,D2)
    x_down = x_up_gate @ self.ffn_down_proj + self.ffn_down_proj_bias.unsqueeze(1)                # (E,B*T,D2) (E,D2,D) (E,1,D)     -> (E,B*T,D)
    x_out = (x_down.reshape(E, B, T, D) * probs).sum(0)                                           # (E,B,T,D) (E,B,T,1)             -> (B,T,D)
    return x + x_out

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                                              # (B,T,D) -> (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm) # (B,T,D) -> (B,T,D)

    B, T, _ = x.shape
    q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)     # (B,T,D) -> (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)     # (B,T,D) -> (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)     # (B,T,D) -> (B,KvH,T,Hd)
    s = self.attn_sink.reshape(1, -1, 1, 1).expand(B, self.head_dim, T, 1)  # (B)     -> (B,H,T,1)

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

    attn = q.scaled_dot_product_attention(k, v, sink=s, attn_mask=mask, enable_gqa=True)  #             -> (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                         # (B,H,T,Hd)  -> (B,T,D)
    attn = self.attn_o(attn)                                                              # (B,T,D)     -> (B,T,D)
    return x + attn

  def __call__(self, x: Tensor, start_pos: int|UOp):
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

class Transformer:
  def __init__(self, dim, hidden_dim, head_dim, num_blocks, n_heads, n_kv_heads, n_experts, n_active_experts, norm_eps, rope_params, vocab_size, sliding_window, max_context):
    self.blk          = [TransformerBlock(dim, hidden_dim, head_dim, n_heads, n_kv_heads, n_experts, n_active_experts, norm_eps, max_context, sliding_window*(i%2==0)) for i in range(num_blocks)]
    self.token_embd   = nn.Embedding(vocab_size, dim)
    self.output_norm  = nn.RMSNorm(dim, norm_eps)
    self.output       = nn.Linear(dim, vocab_size, bias=False)
    self.max_context  = max_context
    # JIT is used if T=1 and start_pos is a UOp. TODO: make this not needed by including T in the JIT and making start_pos always a UOp
    self.forward_jit  = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:float, top_k:int, top_p:float, alpha_f:float, alpha_p:float) -> Tensor:
    x = self.token_embd(tokens)                                                             # (B,T)   -> (B,T,D)
    for block in self.blk: x = block(x, start_pos)                                          # (B,T,D) -> (B,T,D)
    logits = self.output(self.output_norm(x))                                               # (B,T,D) -> (B,T)
    return sample(logits[:, -1, :].flatten(), temperature, top_k, top_p, alpha_f, alpha_p)  # (B,T)   -> (B)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0, temperature:float=0.0, top_k:int=0, top_p:float=0.8, alpha_f:float=0.0, alpha_p:float=0.0) -> Tensor:
    forward = self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward
    return forward(tokens, start_pos, temperature, top_k, top_p, alpha_f, alpha_p)

  @staticmethod
  def from_pretrained(model_path:Path, params:dict[str, int|float|dict], fakeweights:bool=False) -> Transformer:
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

def get_keymap(num_blocks):
  return {
    "model.embed_tokens.weight": "token_embd.weight",
    **{f"model.layers.{l}.input_layernorm.weight": f"blk.{l}.attn_norm.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.{x}_proj.weight": f"blk.{l}.attn_{x}.weight" for x in ["q", "k", "v", "o"] for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.{x}_proj.bias": f"blk.{l}.attn_{x}.bias" for x in ["q", "k", "v", "o"] for l in range(num_blocks)},
    **{f"model.layers.{l}.self_attn.sinks": f"blk.{l}.attn_sink" for l in range(num_blocks)},
    **{f"model.layers.{l}.post_attention_layernorm.weight": f"blk.{l}.ffn_norm.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.router.weight": f"blk.{l}.ffn_gate.weight" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.router.bias": f"blk.{l}.ffn_gate.bias" for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d}_proj_bias": f"blk.{l}.ffn_{d}_proj_bias" for d in ["gate_up", "down"] for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d}_proj_blocks": f"blk.{l}.ffn_{d}_proj_blocks" for d in ["gate_up", "down"] for l in range(num_blocks)},
    **{f"model.layers.{l}.mlp.experts.{d}_proj_scales": f"blk.{l}.ffn_{d}_proj_scales" for d in ["gate_up", "down"] for l in range(num_blocks)},
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
  }

def convert_from_huggingface(weights:dict[str, Tensor], num_blocks: int, n_heads: int, n_kv_heads: int, permute_layers: bool = True):

  keymap = get_keymap(num_blocks) # map hf to tinygrad keys
  sd = {}
  for k, v in weights.items():
    if ".rotary_emb." in k: continue
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if "q_proj" in k and permute_layers: v = permute(v, n_heads)
      elif "k_proj" in k and permute_layers: v = permute(v, n_kv_heads)
    # todo: remove
    if k not in keymap:
      if DEBUG >= 1: print(f"WARNING: {k} not in {keymap.keys()}")
      continue
    sd[keymap[k]] = v
  return sd

def fix_mxfp4(weights, num_blocks) -> Tensor:
  def dequantize_mxfp4(blocks: Tensor, scales: Tensor) -> Tensor:
    """Dequantize MXFP4 to float32. blocks: (*batch, num_blocks, 16), scales: (*batch, num_blocks) -> (*batch, num_blocks*32)"""
    assert blocks.shape[:-1] == scales.shape and blocks.shape[-1] == 16
    MXFP4_ID = 39
    mxfp4_data = scales.unsqueeze(-1).cat(blocks, dim=-1).flatten()  # interleave and flatten to 1D
    return ggml_data_to_tensor(mxfp4_data, scales.numel() * 32, MXFP4_ID).reshape(*scales.shape[:2], -1)

  # dequantize only the ffn_gate_up_proj and ffn_down_proj
  for l in range(num_blocks):
    for d in ['gate_up', 'down']:
      blocks = f'blk.{l}.ffn_{d}_proj_blocks'
      scales = f'blk.{l}.ffn_{d}_proj_scales'
      proj = dequantize_mxfp4(weights.pop(blocks), weights.pop(scales))
      ic(proj, fix(proj))
      weights[f'layers.{l}.ffn_{d}_proj'] = proj
  return weights

def main(args):

  if args.seed is not None: Tensor.manual_seed(args.seed)

  model_info = MODELS[args.size]

  model_path = Path(args.weights) if args.weights or args.fakeweights else download_weights(model_info["model"], model_info["total_num_weights"])
  tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"], cache_dir=model_path)
  expected = [12194,    11,  1495,   553,   481,    30,   357,   939,  8975,    13]

  if getenv("TORCH"):
    print("Using torch, not tinygrad.")
    import torch
    from transformers import GptOssForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using {device}")
    fetch(f"https://huggingface.co/{model_info['model']}/resolve/main/config.json", "config.json", subdir=model_info["model"].split('/')[-1])
    model = GptOssForCausalLM.from_pretrained(model_path, local_files_only=True, cache_dir=model_path, device_map=device)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    generate_ids = model.generate(input_ids, max_new_tokens=args.count) # tensor([[12194,    11,   1495,   553,   481,    30]], device='cuda:0')
    out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(out)
    torch.testing.assert_close(generate_ids[0], torch.tensor(expected, device=device))
    return

  # build model
  model = Transformer.from_pretrained(model_path, model_info["params"], args.fakeweights)

  outputted = args.prompt
  start_pos, toks = 0, tokenizer(outputted)["input_ids"]
  print(outputted, end="", flush=True)

  tok_tensor = None
  for _ in range(args.count):
    # forward pass
    next_tok = Tensor([toks[start_pos:]], dtype=dtypes.int64) if tok_tensor is None or (len(toks)-start_pos) > 1 else tok_tensor.reshape(1, 1)
    tok_tensor = model(next_tok, start_pos, temperature=args.temperature)
    tok = tok_tensor.item()

    # use the kv cache
    start_pos = len(toks)

    # add the new token
    toks.append(tok)
    print(toks)

    # display
    cur = tokenizer.decode(toks, skip_special_tokens=True)
    print(cur[len(outputted):], flush=True)
    outputted = cur

  if args.temperature == 0:
    assert toks == expected, f"generated {toks} but expected {expected}"


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run gpt-oss in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--size", choices=list(MODELS.keys()), default=list(MODELS.keys())[0], help="Model size")
  parser.add_argument("--weights", type=str, default=None, help="Path to the downloaded weights")
  parser.add_argument('--fakeweights',  action='store_true', help="Load fake weights")
  parser.add_argument("--seed", type=int, help="Random seed")
  parser.add_argument("--prompt", type=str, default="Hi, how are you?", help="Phrase to start with")
  parser.add_argument("--count", type=int, default=4, help="Max number of tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.0, help="Temperature in the softmax")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  args = parser.parse_args()

  main(args)
