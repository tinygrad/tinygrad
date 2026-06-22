"""FLUX.2-klein-4B DiT (transformer) forward pass in tinygrad.

Mirrors mflux's mflux/models/flux2/model/flux2_transformer/* exactly (arch, weight key names,
RoPE, modulation/AdaLN, joint attention). Attention defaults to Tensor.scaled_dot_product_attention;
examples/flux2.py --flash opts into the ThunderKittens flash kernel on METAL (extra/thunder/metal/fa.py)
or CUDA (extra/thunder/cuda/fa.py).

Dims (from transformer/config.json):
  hidden (inner_dim) = num_attention_heads(24) * attention_head_dim(128) = 3072
  num_layers (double-stream)        = 5
  num_single_layers (single-stream) = 20
  mlp_ratio = 3.0  -> double-stream FF inner = 9216, single MLP hidden = 9216
  in_channels = 128, joint_attention_dim = 7680, timestep_guidance_channels = 256
  axes_dims_rope = (32,32,32,32) -> head_dim/2 = 64 rope freqs total, rope_theta = 2000
  guidance_embeds = False
"""
import math
from tinygrad import Tensor, Device
from tinygrad.nn import Linear, RMSNorm
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
from extra.models.flux2 import HF_BASE

# ThunderKittens flash kernels: the Metal one (extra/thunder/metal/fa.py) on METAL, the CUDA one
# (extra/thunder/cuda/fa.py) on CUDA, loaded when the backend supports it. Default attention is
# Tensor.scaled_dot_product_attention (which BEAM autotunes well); opt into the flash kernel by
# setting USE_FLASH=True (examples/flux2.py --flash).
flash_attention = None
if Device.DEFAULT in ("METAL", "CUDA"):
  try:
    if Device.DEFAULT == "METAL": from extra.thunder.metal.fa import flash_attention
    elif Device.DEFAULT == "CUDA": from extra.thunder.cuda.fa import flash_attention_cuda as flash_attention
  except Exception: flash_attention = None  # kernel/toolchain unavailable
USE_FLASH = False                           # default SDPA; opt in with --flash (set USE_FLASH=True)

def _ln(x: Tensor) -> Tensor:
  # affine-free LayerNorm. mflux computes the stats in fp32 and casts back to the
  # input dtype (bf16); that rounding cadence keeps the 25-block residual stream
  # bit-close to the reference, so upcast around the built-in layernorm.
  return x.float().layernorm(eps=1e-6).cast(x.dtype)

def apply_rope_interleaved(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
  """x: (B, H, S, D). cos/sin: (S, D/2). Interleaved (paired) rotation matching
  mflux AttentionUtils.apply_rope_bshd: reshape D into (D/2, 2) pairs (real,imag),
  rotate each pair by (cos,sin)."""
  B, H, S, D = x.shape
  xf = x.float().reshape(B, H, S, D // 2, 2)
  real, imag = xf[:, :, :, :, 0], xf[:, :, :, :, 1]
  cos_b, sin_b = cos.float().reshape(1, 1, S, D // 2), sin.float().reshape(1, 1, S, D // 2)
  out = Tensor.stack(real * cos_b - imag * sin_b, imag * cos_b + real * sin_b, dim=-1)
  return out.reshape(B, H, S, D).cast(x.dtype)

# ---------------------------------------------------------------------------
# positional embedding (RoPE)
# ---------------------------------------------------------------------------
class PosEmbed:
  def __init__(self, theta: int = 2000, axes_dim=(32, 32, 32, 32)):
    self.theta, self.axes_dim = theta, axes_dim

  def __call__(self, ids: Tensor):
    # ids: (seq, 4) int.  returns (cos,sin) each (seq, sum(axes_dim)//2)
    pos = ids.float()
    cos_out, sin_out = [], []
    for i, dim in enumerate(self.axes_dim):
      omega = 1.0 / (self.theta ** (Tensor.arange(0, dim, 2).float() / dim))  # (dim/2,)
      out = pos[:, i:i + 1] * omega.reshape(1, -1)                            # (seq, dim/2)
      cos_out.append(out.cos()); sin_out.append(out.sin())
    return Tensor.cat(*cos_out, dim=-1), Tensor.cat(*sin_out, dim=-1)

# ---------------------------------------------------------------------------
# timestep + guidance embedding
# ---------------------------------------------------------------------------
def timestep_embedding(t: Tensor, dim: int) -> Tensor:
  # same freq schedule as extra/models/unet.py:10-15 (flip_sin_to_cos=True -> [cos, sin] order);
  # kept in fp32 here (unet force-casts to mixed_precision f16) for the bit-close residual stream.
  half = dim // 2
  freqs = (-math.log(10000.0) * Tensor.arange(0, half).float() / half).exp()
  args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
  return Tensor.cat(args.cos(), args.sin(), dim=-1)

class TimestepEmbedder:
  def __init__(self, in_channels: int, embedding_dim: int):
    self.linear_1 = Linear(in_channels, embedding_dim, bias=False)
    self.linear_2 = Linear(embedding_dim, embedding_dim, bias=False)

  def __call__(self, t: Tensor) -> Tensor:
    return self.linear_2(self.linear_1(t).silu())

class TimestepGuidanceEmbeddings:
  def __init__(self, in_channels: int, embedding_dim: int):
    self.in_channels = in_channels
    self.timestep_embedder = TimestepEmbedder(in_channels, embedding_dim)

  def __call__(self, timestep: Tensor) -> Tensor:
    return self.timestep_embedder(timestep_embedding(timestep, self.in_channels))

# ---------------------------------------------------------------------------
# modulation (AdaLN parameter producer)
# ---------------------------------------------------------------------------
class Modulation:
  def __init__(self, dim: int, mod_param_sets: int = 2):
    self.mod_param_sets = mod_param_sets
    self.linear = Linear(dim, dim * 3 * mod_param_sets, bias=False)

  def __call__(self, temb: Tensor):
    mod = self.linear(temb.silu())
    if mod.ndim == 2: mod = mod.reshape(mod.shape[0], 1, mod.shape[1])
    parts = mod.chunk(3 * self.mod_param_sets, dim=-1)
    return tuple(parts[3 * i: 3 * (i + 1)] for i in range(self.mod_param_sets))

# ---------------------------------------------------------------------------
# feed-forward (SwiGLU)
# ---------------------------------------------------------------------------
def swiglu(x: Tensor) -> Tensor:
  # fused-projection SwiGLU: up+gate come from a single Linear (mflux layout), so chunk here.
  # cf. extra/models/llama.py FeedForward, which keeps gate/up as separate Linears.
  x1, x2 = x.chunk(2, dim=-1)
  return x1.silu() * x2

class FeedForward:
  def __init__(self, dim: int, mult: float = 3.0):
    inner = int(dim * mult)
    self.linear_in = Linear(dim, inner * 2, bias=False)
    self.linear_out = Linear(inner, dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.linear_out(swiglu(self.linear_in(x)))

# ---------------------------------------------------------------------------
# attention building blocks
# ---------------------------------------------------------------------------
def _to_bhsd(x: Tensor, heads: int, dim_head: int) -> Tensor:
  B, S, _ = x.shape
  return x.reshape(B, S, heads, dim_head).permute(0, 2, 1, 3)  # (B,H,S,D)

def _joint_flash(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
  # q,k,v: (B,H,S,D). returns (B, S, H*D).
  B, H, S, D = q.shape
  if flash_attention is None or not USE_FLASH:
    out = q.scaled_dot_product_attention(k, v)              # (B,H,S,D)
  elif Device.DEFAULT == "CUDA":
    out = flash_attention(q, k, v)                          # CUDA kernel takes (B,H,S,D) natively
  else:
    qf, kf, vf = (t.reshape(B * H, S, D) for t in (q, k, v))
    out = flash_attention(qf, kf, vf).reshape(B, H, S, D)   # METAL: folded (B*H,S,D)
  return out.cast(q.dtype).permute(0, 2, 1, 3).reshape(B, S, H * D)

class DoubleAttention:
  def __init__(self, dim: int, heads: int, dim_head: int):
    self.heads, self.dim_head = heads, dim_head
    inner = heads * dim_head
    self.to_q = Linear(dim, inner, bias=False)
    self.to_k = Linear(dim, inner, bias=False)
    self.to_v = Linear(dim, inner, bias=False)
    self.norm_q = RMSNorm(dim_head, eps=1e-5)
    self.norm_k = RMSNorm(dim_head, eps=1e-5)
    self.to_out = [Linear(inner, dim, bias=False)]        # to_out.0 in weights
    self.add_q_proj = Linear(dim, inner, bias=False)
    self.add_k_proj = Linear(dim, inner, bias=False)
    self.add_v_proj = Linear(dim, inner, bias=False)
    self.norm_added_q = RMSNorm(dim_head, eps=1e-5)
    self.norm_added_k = RMSNorm(dim_head, eps=1e-5)
    self.to_add_out = Linear(inner, dim, bias=False)

  def __call__(self, hidden: Tensor, enc: Tensor, cos: Tensor, sin: Tensor):
    H, D = self.heads, self.dim_head
    q = self.norm_q(_to_bhsd(self.to_q(hidden), H, D))
    k = self.norm_k(_to_bhsd(self.to_k(hidden), H, D))
    v = _to_bhsd(self.to_v(hidden), H, D)
    eq = self.norm_added_q(_to_bhsd(self.add_q_proj(enc), H, D))
    ek = self.norm_added_k(_to_bhsd(self.add_k_proj(enc), H, D))
    ev = _to_bhsd(self.add_v_proj(enc), H, D)
    # concat encoder (text) first, then image -- mflux order [enc, img] on seq axis
    q = apply_rope_interleaved(eq.cat(q, dim=2), cos, sin)
    k = apply_rope_interleaved(ek.cat(k, dim=2), cos, sin)
    v = ev.cat(v, dim=2)
    attn = _joint_flash(q, k, v)                          # (B, S_txt+S_img, inner)
    txt_len = enc.shape[1]
    return self.to_out[0](attn[:, txt_len:]), self.to_add_out(attn[:, :txt_len])

class ParallelSelfAttention:
  def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 3.0):
    self.heads, self.dim_head = heads, dim_head
    self.inner = heads * dim_head
    self.mlp_hidden = int(dim * mlp_ratio)
    self.to_qkv_mlp_proj = Linear(dim, self.inner * 3 + self.mlp_hidden * 2, bias=False)
    self.norm_q = RMSNorm(dim_head, eps=1e-5)
    self.norm_k = RMSNorm(dim_head, eps=1e-5)
    self.to_out = Linear(self.inner + self.mlp_hidden, dim, bias=False)

  def __call__(self, x: Tensor, cos: Tensor, sin: Tensor):
    H, D = self.heads, self.dim_head
    qkv, mlp_hidden = self.to_qkv_mlp_proj(x).split([self.inner * 3, self.mlp_hidden * 2], dim=-1)
    q, k, v = qkv.chunk(3, dim=-1)
    q = apply_rope_interleaved(self.norm_q(_to_bhsd(q, H, D)), cos, sin)
    k = apply_rope_interleaved(self.norm_k(_to_bhsd(k, H, D)), cos, sin)
    v = _to_bhsd(v, H, D)
    attn = _joint_flash(q, k, v)                          # (B,S,inner)
    return self.to_out(attn.cat(swiglu(mlp_hidden), dim=-1))

# ---------------------------------------------------------------------------
# transformer blocks
# ---------------------------------------------------------------------------
class DoubleBlock:
  def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 3.0):
    self.attn = DoubleAttention(dim, heads, dim_head)
    self.ff = FeedForward(dim, mlp_ratio)
    self.ff_context = FeedForward(dim, mlp_ratio)

  def __call__(self, hidden, enc, mod_img, mod_txt, cos, sin):
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = mod_img
    (cshift_msa, cscale_msa, cgate_msa), (cshift_mlp, cscale_mlp, cgate_mlp) = mod_txt
    attn_out, enc_attn_out = self.attn((1 + scale_msa) * _ln(hidden) + shift_msa,
                                       (1 + cscale_msa) * _ln(enc) + cshift_msa, cos, sin)
    hidden = hidden + gate_msa * attn_out
    enc = enc + cgate_msa * enc_attn_out
    hidden = hidden + gate_mlp * self.ff((1 + scale_mlp) * _ln(hidden) + shift_mlp)
    enc = enc + cgate_mlp * self.ff_context((1 + cscale_mlp) * _ln(enc) + cshift_mlp)
    return enc, hidden

class SingleBlock:
  def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 3.0):
    self.attn = ParallelSelfAttention(dim, heads, dim_head, mlp_ratio)

  def __call__(self, hidden, mod, cos, sin):
    shift, scale, gate = mod
    return hidden + gate * self.attn((1 + scale) * _ln(hidden) + shift, cos, sin)

class AdaLayerNormContinuous:
  def __init__(self, embedding_dim: int, cond_dim: int):
    self.embedding_dim = embedding_dim
    self.linear = Linear(cond_dim, embedding_dim * 2, bias=False)

  def __call__(self, x: Tensor, temb: Tensor) -> Tensor:
    scale, shift = self.linear(temb.silu()).reshape(temb.shape[0], 1, -1).chunk(2, dim=-1)
    return _ln(x) * (1 + scale) + shift

# ---------------------------------------------------------------------------
# full transformer
# ---------------------------------------------------------------------------
class Flux2Transformer:
  def __init__(self, in_channels=128, num_layers=5, num_single_layers=20,
               attention_head_dim=128, num_attention_heads=24, joint_attention_dim=7680,
               timestep_guidance_channels=256, mlp_ratio=3.0,
               axes_dims_rope=(32, 32, 32, 32), rope_theta=2000, patch_size=1):
    self.heads, self.dim_head = num_attention_heads, attention_head_dim
    self.inner_dim = num_attention_heads * attention_head_dim
    self.out_channels, self.patch_size = in_channels, patch_size
    self.pos_embed = PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
    self.time_guidance_embed = TimestepGuidanceEmbeddings(timestep_guidance_channels, self.inner_dim)
    self.double_stream_modulation_img = Modulation(self.inner_dim, 2)
    self.double_stream_modulation_txt = Modulation(self.inner_dim, 2)
    self.single_stream_modulation = Modulation(self.inner_dim, 1)
    self.x_embedder = Linear(in_channels, self.inner_dim, bias=False)
    self.context_embedder = Linear(joint_attention_dim, self.inner_dim, bias=False)
    self.transformer_blocks = [DoubleBlock(self.inner_dim, self.heads, self.dim_head, mlp_ratio) for _ in range(num_layers)]
    self.single_transformer_blocks = [SingleBlock(self.inner_dim, self.heads, self.dim_head, mlp_ratio) for _ in range(num_single_layers)]
    self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
    self.proj_out = Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

  def __call__(self, hidden_states: Tensor, encoder_hidden_states: Tensor,
               timestep: Tensor, img_ids: Tensor, txt_ids: Tensor) -> Tensor:
    temb = self.time_guidance_embed(timestep.float()).cast(hidden_states.dtype)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    img_cos, img_sin = self.pos_embed(img_ids)
    txt_cos, txt_sin = self.pos_embed(txt_ids)
    cos, sin = txt_cos.cat(img_cos, dim=0), txt_sin.cat(img_sin, dim=0)

    mod_img = self.double_stream_modulation_img(temb)
    mod_txt = self.double_stream_modulation_txt(temb)
    for blk in self.transformer_blocks:
      encoder_hidden_states, hidden_states = blk(hidden_states, encoder_hidden_states, mod_img, mod_txt, cos, sin)

    hidden_states = encoder_hidden_states.cat(hidden_states, dim=1)
    mod_single = self.single_stream_modulation(temb)[0]
    for blk in self.single_transformer_blocks:
      hidden_states = blk(hidden_states, mod_single, cos, sin)

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:]
    return self.proj_out(self.norm_out(hidden_states, temb))


DIT_WEIGHTS_URL = HF_BASE + "transformer/diffusion_pytorch_model.safetensors"

def load_dit() -> Flux2Transformer:
  model = Flux2Transformer()
  load_state_dict(model, safe_load(fetch(DIT_WEIGHTS_URL, "diffusion_pytorch_model.safetensors",
                                         subdir="flux2-klein-4b/transformer")), strict=True)
  return model
