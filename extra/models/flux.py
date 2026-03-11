"""
Flux.1 model implementation for tinygrad.
MLPerf Training v5.1 text-to-image benchmark.

Reference: Black Forest Labs Flux.1 (https://github.com/black-forest-labs/flux)
Architecture: Diffusion Transformer (DiT) with double-stream and single-stream blocks.
  - Double-stream blocks: separate image + text streams with bidirectional attention
  - Single-stream blocks: fused image+text processing
  - Rectified flow training (not DDPM)
  - Pre-computed VAE latents + T5/CLIP embeddings (frozen encoders)

Model size: ~11.9B parameters (Flux.1-dev / Flux.1-schnell share same arch)
MLPerf uses: 1,099,776 CC12M samples, validation loss target (not FID)
"""

import math
from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import getenv
from typing import Optional

# ── patching hooks (same pattern as unet.py) ──────────────────────────────────
Linear    = nn.Linear
LayerNorm = nn.LayerNorm

# ── helpers ───────────────────────────────────────────────────────────────────

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
  """AdaLN modulation: x * (1 + scale) + shift"""
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# ── positional embeddings ─────────────────────────────────────────────────────

def rope_freqs(dim: int, theta: float = 10000.0) -> Tensor:
  """Standard 1D RoPE frequencies."""
  assert dim % 2 == 0
  inv_freq = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtypes.float32) / dim))
  return inv_freq

def apply_rope_1d(x: Tensor, freqs: Tensor) -> Tensor:
  """Apply 1D RoPE to a tensor of shape (B, H, S, D)."""
  # x: (B, H, S, D)  freqs: (S, D//2)
  d = x.shape[-1]
  x1, x2 = x[..., :d//2], x[..., d//2:]
  cos = freqs.cos().unsqueeze(0).unsqueeze(0)   # (1, 1, S, D//2)
  sin = freqs.sin().unsqueeze(0).unsqueeze(0)
  return Tensor.cat(x1 * cos - x2 * sin, x1 * sin + x2 * cos, dim=-1)

def apply_rope_2d(x: Tensor, h_freqs: Tensor, w_freqs: Tensor) -> Tensor:
  """Apply 2D RoPE to image tokens — interleave H and W frequencies."""
  # x: (B, H_heads, S_img, D_head)
  # h_freqs, w_freqs: (S_img, D//4)  — split D_head into 4 chunks
  d = x.shape[-1]
  q = d // 4
  x0, x1, x2, x3 = x[..., :q], x[..., q:2*q], x[..., 2*q:3*q], x[..., 3*q:]
  ch = h_freqs.cos().unsqueeze(0).unsqueeze(0)
  sh = h_freqs.sin().unsqueeze(0).unsqueeze(0)
  cw = w_freqs.cos().unsqueeze(0).unsqueeze(0)
  sw = w_freqs.sin().unsqueeze(0).unsqueeze(0)
  return Tensor.cat(
    x0 * ch - x1 * sh,
    x0 * sh + x1 * ch,
    x2 * cw - x3 * sw,
    x2 * sw + x3 * cw,
    dim=-1
  )

def build_2d_rope_freqs(h: int, w: int, d_head: int, theta: float = 10000.0):
  """Build H and W rope frequency tensors for a grid of h*w image patches."""
  assert d_head % 4 == 0, "d_head must be divisible by 4 for 2D RoPE"
  half_dim = d_head // 4
  freqs = rope_freqs(half_dim * 2, theta=theta)[:half_dim]  # (D//4,)
  h_pos = Tensor.arange(h, dtype=dtypes.float32)            # (H,)
  w_pos = Tensor.arange(w, dtype=dtypes.float32)            # (W,)
  # outer product → grid frequencies
  h_freqs = h_pos.unsqueeze(1) * freqs.unsqueeze(0)         # (H, D//4)
  w_freqs = w_pos.unsqueeze(1) * freqs.unsqueeze(0)         # (W, D//4)
  # tile to (H*W, D//4)
  h_freqs = h_freqs.unsqueeze(1).expand(h, w, half_dim).reshape(h * w, half_dim)
  w_freqs = w_freqs.unsqueeze(0).expand(h, w, half_dim).reshape(h * w, half_dim)
  return h_freqs, w_freqs

# ── attention ─────────────────────────────────────────────────────────────────

class SelfAttention:
  """Multi-head self-attention with optional RoPE."""
  def __init__(self, dim: int, n_heads: int, qkv_bias: bool = True):
    self.n_heads = n_heads
    self.head_dim = dim // n_heads
    self.qkv = Linear(dim, 3 * dim, bias=qkv_bias)
    self.proj = Linear(dim, dim)
    self.norm_q = nn.RMSNorm(self.head_dim)
    self.norm_k = nn.RMSNorm(self.head_dim)

  def __call__(self, x: Tensor, rope_h: Optional[Tensor] = None, rope_w: Optional[Tensor] = None) -> Tensor:
    B, S, _ = x.shape
    qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]   # each: (B, H, S, D)
    q = self.norm_q(q)
    k = self.norm_k(k)
    if rope_h is not None:
      q = apply_rope_2d(q, rope_h, rope_w)
      k = apply_rope_2d(k, rope_h, rope_w)
    attn = q.scaled_dot_product_attention(k, v)
    return self.proj(attn.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim))

# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP:
  def __init__(self, in_dim: int, hidden_dim: int):
    self.fc1 = Linear(in_dim, hidden_dim)
    self.fc2 = Linear(hidden_dim, in_dim)

  def __call__(self, x: Tensor) -> Tensor:
    return self.fc2(self.fc1(x).gelu())

# ── time/conditioning embeddings ──────────────────────────────────────────────

class TimestepEmbedding:
  """Sinusoidal timestep → MLP projection."""
  def __init__(self, dim: int, freq_embed_dim: int = 256):
    self.dim = freq_embed_dim
    self.mlp = [Linear(freq_embed_dim, dim), Tensor.silu, Linear(dim, dim)]

  def _sinusoidal(self, t: Tensor) -> Tensor:
    half = self.dim // 2
    freqs = (-math.log(10000.0) * Tensor.arange(half, dtype=dtypes.float32) / half).exp()
    args = t.unsqueeze(1).cast(dtypes.float32) * freqs.unsqueeze(0)
    return Tensor.cat(args.cos(), args.sin(), dim=-1)

  def __call__(self, t: Tensor) -> Tensor:
    return self._sinusoidal(t).sequential(self.mlp)

class VectorEmbedding:
  """Project pooled CLIP vector to model dim."""
  def __init__(self, in_dim: int, out_dim: int):
    self.mlp = [Linear(in_dim, out_dim), Tensor.silu, Linear(out_dim, out_dim)]

  def __call__(self, x: Tensor) -> Tensor:
    return x.sequential(self.mlp)

# ── Double-stream block ───────────────────────────────────────────────────────

class DoubleStreamBlock:
  """
  Processes image tokens (x) and text tokens (c) in parallel streams,
  with bidirectional cross-attention by concatenating for a single attention call.
  Ref: Flux paper section 3.1
  """
  def __init__(self, hidden_size: int, n_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
    mlp_hidden = int(hidden_size * mlp_ratio)
    self.num_heads = n_heads
    self.head_dim = hidden_size // n_heads

    # image stream
    self.img_norm1    = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn     = SelfAttention(hidden_size, n_heads, qkv_bias)
    self.img_norm2    = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_mlp      = MLP(hidden_size, mlp_hidden)
    self.img_mod      = Linear(hidden_size, 6 * hidden_size)  # AdaLN modulation (shift/scale/gate x3)

    # text stream
    self.txt_norm1    = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_attn     = SelfAttention(hidden_size, n_heads, qkv_bias)
    self.txt_norm2    = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_mlp      = MLP(hidden_size, mlp_hidden)
    self.txt_mod      = Linear(hidden_size, 6 * hidden_size)

  def __call__(self, img: Tensor, txt: Tensor, vec: Tensor,
               rope_h: Tensor, rope_w: Tensor) -> tuple:
    # conditioning vectors → AdaLN parameters
    img_mods = self.img_mod(vec.silu()).chunk(6, dim=-1)
    txt_mods = self.txt_mod(vec.silu()).chunk(6, dim=-1)
    img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = img_mods
    txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = txt_mods

    # ── attention (joint img+txt) ──────────────────────────────────────────
    img_norm = modulate(self.img_norm1(img), img_shift1, img_scale1)
    txt_norm = modulate(self.txt_norm1(txt), txt_shift1, txt_scale1)

    B, S_img, D = img_norm.shape
    S_txt = txt_norm.shape[1]

    # compute QKV separately so we can apply 2D RoPE only to image Q/K
    def qkv(attn_mod: SelfAttention, x: Tensor):
      qkv_ = attn_mod.qkv(x).reshape(B, x.shape[1], 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
      q_, k_, v_ = qkv_[0], qkv_[1], qkv_[2]
      q_ = attn_mod.norm_q(q_)
      k_ = attn_mod.norm_k(k_)
      return q_, k_, v_

    img_q, img_k, img_v = qkv(self.img_attn, img_norm)
    txt_q, txt_k, txt_v = qkv(self.txt_attn, txt_norm)

    # Apply 2D RoPE to image tokens only
    img_q = apply_rope_2d(img_q, rope_h, rope_w)
    img_k = apply_rope_2d(img_k, rope_h, rope_w)

    # Joint attention: concatenate along sequence
    q = Tensor.cat(img_q, txt_q, dim=2)
    k = Tensor.cat(img_k, txt_k, dim=2)
    v = Tensor.cat(img_v, txt_v, dim=2)

    attn = q.scaled_dot_product_attention(k, v)   # (B, H, S_img+S_txt, D)
    attn = attn.transpose(1, 2).reshape(B, S_img + S_txt, self.num_heads * self.head_dim)

    img_attn_out = self.img_attn.proj(attn[:, :S_img])
    txt_attn_out = self.txt_attn.proj(attn[:, S_img:])

    # ── residual + MLP ────────────────────────────────────────────────────
    img = img + img_gate1.unsqueeze(1) * img_attn_out
    txt = txt + txt_gate1.unsqueeze(1) * txt_attn_out

    img = img + img_gate2.unsqueeze(1) * self.img_mlp(modulate(self.img_norm2(img), img_shift2, img_scale2))
    txt = txt + txt_gate2.unsqueeze(1) * self.txt_mlp(modulate(self.txt_norm2(txt), txt_shift2, txt_scale2))

    return img, txt

# ── Single-stream block ───────────────────────────────────────────────────────

class SingleStreamBlock:
  """
  Single-stream block: img+txt concatenated, processed jointly.
  Ref: Flux paper section 3.1
  """
  def __init__(self, hidden_size: int, n_heads: int, mlp_ratio: float = 4.0, qk_scale: Optional[float] = None):
    self.hidden_size = hidden_size
    self.n_heads = n_heads
    self.head_dim = hidden_size // n_heads
    mlp_hidden = int(hidden_size * mlp_ratio)
    self.mlp_hidden = mlp_hidden

    self.linear1 = Linear(hidden_size, hidden_size * 3 + mlp_hidden)  # qkv + mlp pre-activation
    self.linear2 = Linear(hidden_size + mlp_hidden, hidden_size)
    self.norm     = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.pre_norm = nn.RMSNorm(self.head_dim)
    self.mod      = Linear(hidden_size, hidden_size, bias=True)        # AdaLN (shift only in single stream)
    self.norm_q   = nn.RMSNorm(self.head_dim)
    self.norm_k   = nn.RMSNorm(self.head_dim)

  def __call__(self, x: Tensor, vec: Tensor,
               rope_h: Tensor, rope_w: Tensor, txt_len: int) -> Tensor:
    B, S, D = x.shape
    mod = self.mod(vec.silu())
    x_norm = (1 + mod.unsqueeze(1)) * self.norm(x)

    # project to qkv + mlp_hidden
    proj = self.linear1(x_norm)
    qkv, mlp_in = proj[..., :D * 3], proj[..., D * 3:]

    q, k, v = qkv.reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q = self.norm_q(q)
    k = self.norm_k(k)

    # Apply 2D RoPE to image portion only
    img_len = S - txt_len
    img_q = apply_rope_2d(q[:, :, :img_len], rope_h, rope_w)
    txt_q = q[:, :, img_len:]
    img_k = apply_rope_2d(k[:, :, :img_len], rope_h, rope_w)
    txt_k = k[:, :, img_len:]
    q_full = Tensor.cat(img_q, txt_q, dim=2)
    k_full = Tensor.cat(img_k, txt_k, dim=2)

    attn = q_full.scaled_dot_product_attention(k_full, v)
    attn = attn.transpose(1, 2).reshape(B, S, D)

    # MLP gate (GELU activation on the extra hidden chunk)
    mlp_out = mlp_in.gelu()
    # project combined [attn, mlp] back to hidden_size
    out = self.linear2(Tensor.cat(attn, mlp_out, dim=-1))
    return x + out

# ── Final layer ───────────────────────────────────────────────────────────────

class LastLayer:
  def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
    self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation = Linear(hidden_size, 2 * hidden_size, bias=True)

  def __call__(self, x: Tensor, vec: Tensor) -> Tensor:
    shift, scale = self.adaLN_modulation(vec.silu()).chunk(2, dim=-1)
    x = modulate(self.norm_final(x), shift, scale)
    return self.linear(x)

# ── Main Flux model ───────────────────────────────────────────────────────────

class FluxModel:
  """
  Flux.1 Diffusion Transformer.

  Args (MLPerf Flux.1 config):
    in_channels:      16   (VAE latent channels)
    hidden_size:      3072
    n_heads:          24
    double_layers:    19
    single_layers:    38
    txt_in_dim:       4096  (T5-XXL embedding dim)
    vec_in_dim:       768   (CLIP embedding dim)
    patch_size:       2
    mlp_ratio:        4.0
  """
  def __init__(
    self,
    in_channels:    int = 16,
    hidden_size:    int = 3072,
    n_heads:        int = 24,
    double_layers:  int = 19,
    single_layers:  int = 38,
    txt_in_dim:     int = 4096,
    vec_in_dim:     int = 768,
    patch_size:     int = 2,
    mlp_ratio:      float = 4.0,
    theta:          float = 10000.0,
  ):
    self.patch_size  = patch_size
    self.in_channels = in_channels
    self.hidden_size = hidden_size

    # patchify projection
    self.img_in = Linear(in_channels * patch_size * patch_size, hidden_size, bias=True)
    # text token projection
    self.txt_in = Linear(txt_in_dim, hidden_size, bias=True)

    # conditioning: time + pooled CLIP
    self.time_in = TimestepEmbedding(hidden_size)
    self.vec_in  = VectorEmbedding(vec_in_dim, hidden_size)
    self.guidance_in = TimestepEmbedding(hidden_size)  # guidance distillation (used in dev, not schnell)

    self.double_blocks = [DoubleStreamBlock(hidden_size, n_heads, mlp_ratio) for _ in range(double_layers)]
    self.single_blocks = [SingleStreamBlock(hidden_size, n_heads, mlp_ratio) for _ in range(single_layers)]
    self.final_layer   = LastLayer(hidden_size, patch_size, in_channels)

    self.theta = theta

  def patchify(self, x: Tensor) -> tuple:
    """Convert (B, C, H, W) latent → (B, S, D) patch tokens + return (ph, pw) for RoPE."""
    B, C, H, W = x.shape
    p = self.patch_size
    assert H % p == 0 and W % p == 0, f"spatial dims {H}x{W} not divisible by patch_size {p}"
    ph, pw = H // p, W // p
    # (B, C, ph, p, pw, p) → (B, ph, pw, C, p, p) → (B, ph*pw, C*p*p)
    x = x.reshape(B, C, ph, p, pw, p).permute(0, 2, 4, 1, 3, 5).reshape(B, ph * pw, C * p * p)
    return self.img_in(x), ph, pw

  def unpatchify(self, x: Tensor, ph: int, pw: int) -> Tensor:
    """Convert (B, S, D_out) → (B, C, H, W)."""
    B = x.shape[0]
    p = self.patch_size
    C = self.in_channels
    x = x.reshape(B, ph, pw, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, ph * p, pw * p)
    return x

  def __call__(
    self,
    img_latents: Tensor,   # (B, C, H, W) — pre-encoded VAE latents
    txt_tokens:  Tensor,   # (B, S_txt, D_t5) — pre-encoded T5 embeddings
    vec:         Tensor,   # (B, D_clip) — pre-encoded pooled CLIP embeddings
    timesteps:   Tensor,   # (B,) — continuous in [0, 1] for rectified flow
    guidance:    Optional[Tensor] = None,  # (B,) — guidance scale distillation
  ) -> Tensor:

    # ── patchify ──────────────────────────────────────────────────────────
    img, ph, pw = self.patchify(img_latents)   # (B, S_img, hidden_size)
    txt = self.txt_in(txt_tokens)              # (B, S_txt, hidden_size)

    # ── conditioning vector ───────────────────────────────────────────────
    cond_vec = self.time_in(timesteps) + self.vec_in(vec)
    if guidance is not None:
      cond_vec = cond_vec + self.guidance_in(guidance)

    # ── build 2D RoPE for this spatial resolution ─────────────────────────
    rope_h, rope_w = build_2d_rope_freqs(ph, pw, self.hidden_size // self.hidden_size, theta=self.theta)
    # NOTE: RoPE uses head_dim = hidden_size // n_heads, recompute properly:
    head_dim = self.hidden_size // len(self.double_blocks[0].img_attn.qkv.weight) if False else (self.hidden_size // 24)
    rope_h, rope_w = build_2d_rope_freqs(ph, pw, head_dim, theta=self.theta)
    rope_h = rope_h.to(img.device)
    rope_w = rope_w.to(img.device)

    # ── double-stream blocks ──────────────────────────────────────────────
    for block in self.double_blocks:
      img, txt = block(img, txt, cond_vec, rope_h, rope_w)

    # ── single-stream blocks ──────────────────────────────────────────────
    x = Tensor.cat(img, txt, dim=1)
    txt_len = txt.shape[1]
    for block in self.single_blocks:
      x = block(x, cond_vec, rope_h, rope_w, txt_len)
    img = x[:, :img.shape[1]]

    # ── final layer → unpatchify ──────────────────────────────────────────
    img = self.final_layer(img, cond_vec)
    return self.unpatchify(img, ph, pw)   # (B, C, H, W)


# ── MLPerf config ─────────────────────────────────────────────────────────────

def flux_mlperf() -> FluxModel:
  """Instantiate Flux.1 with MLPerf Training v5.1 config."""
  return FluxModel(
    in_channels   = 16,
    hidden_size   = 3072,
    n_heads       = 24,
    double_layers = 19,
    single_layers = 38,
    txt_in_dim    = 4096,
    vec_in_dim    = 768,
    patch_size    = 2,
    mlp_ratio     = 4.0,
  )


# ── Rectified flow loss ───────────────────────────────────────────────────────

def rectified_flow_loss(model: FluxModel, latents: Tensor, txt: Tensor, vec: Tensor,
                        guidance: Optional[Tensor] = None) -> Tensor:
  """
  Rectified flow (flow-matching) training loss.
  x0 = noise, x1 = data (latent), t ~ Uniform(0, 1)
  x_t = (1 - t) * x0 + t * x1
  target velocity v = x1 - x0
  loss = MSE(model(x_t, t) - v)

  Ref: Lipman et al. "Flow Matching for Generative Modeling"
       Esser et al. "Scaling Rectified Flow Transformers" (Flux paper)
  """
  B = latents.shape[0]
  # sample continuous timestep t in (0, 1)
  # Flux uses logit-normal timestep sampling for training efficiency
  t_raw = Tensor.randn(B, device=latents.device)  # standard normal
  t = t_raw.sigmoid()                              # logit-normal → (0,1)
  t_broadcast = t.reshape(B, 1, 1, 1)

  noise = Tensor.randn(*latents.shape, device=latents.device)
  # linear interpolation between noise and data
  x_t = (1 - t_broadcast) * noise + t_broadcast * latents

  # target vector field (velocity from noise → data)
  v_target = latents - noise   # (x1 - x0)

  v_pred = model(x_t, txt, vec, t, guidance)
  return ((v_pred - v_target) ** 2).mean()
