import math
from typing import List, Optional

from tinygrad import Tensor
from tinygrad.nn import Linear, LayerNorm

__all__ = ["FluxSchnellTiny"]

TINY_PATCH = 2
TINY_DIM = 256
TINY_DEPTH = 6
TINY_HEADS = 8
TINY_CROSS_INTERVAL = 2

class PatchEmbed:
  def __init__(self, in_channels:int, hidden_dim:int, patch_size:int):
    self.patch_size = patch_size
    self.hidden_dim = hidden_dim
    self.proj = Linear(in_channels * patch_size * patch_size, hidden_dim)

  def __call__(self, x:Tensor) -> Tensor:
    b, c, h, w = x.shape
    assert h % self.patch_size == 0 and w % self.patch_size == 0, "spatial dims must be divisible by patch size"
    ph, pw = h // self.patch_size, w // self.patch_size
    x = x.reshape(b, c, ph, self.patch_size, pw, self.patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, ph * pw, c * self.patch_size * self.patch_size)
    return self.proj(x)

class PatchUnembed:
  def __init__(self, out_channels:int, hidden_dim:int, patch_size:int):
    self.patch_size = patch_size
    self.reconstruct = Linear(hidden_dim, out_channels * patch_size * patch_size)

  def __call__(self, tokens:Tensor, height:int, width:int) -> Tensor:
    b, n, _ = tokens.shape
    ph, pw = height // self.patch_size, width // self.patch_size
    assert n == ph * pw, f"token count mismatch, expected {ph*pw} got {n}"
    x = self.reconstruct(tokens).reshape(b, ph, pw, -1, self.patch_size, self.patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, -1, height, width)
    return x

def timestep_embedding(t:Tensor, dim:int) -> Tensor:
  if t.ndim == 1:
    t = t.reshape(-1, 1)
  half = dim // 2
  device = t.device
  freqs = Tensor.arange(half, device=device).float()
  freqs = (Tensor(1.0, device=device) / (Tensor(10000.0, device=device) ** (freqs / max(1, half)))).reshape(1, half)
  angles = t * freqs
  emb = angles.cos().cat(angles.sin(), dim=-1)
  if dim % 2 == 1:
    emb = emb.cat(Tensor.zeros(emb.shape[0], 1, device=device), dim=-1)
  return emb

class TimeEmbedding:
  def __init__(self, dim:int):
    self.dim = dim
    self.lin1 = Linear(dim, dim * 4)
    self.lin2 = Linear(dim * 4, dim)

  def __call__(self, t:Tensor) -> Tensor:
    emb = timestep_embedding(t, self.dim)
    emb = self.lin1(emb).gelu()
    return self.lin2(emb)

class MultiHeadAttention:
  def __init__(self, hidden_dim:int, heads:int, context_dim:Optional[int]=None):
    self.heads = heads
    self.dim_head = hidden_dim // heads
    self.scale = 1.0 / math.sqrt(self.dim_head)
    ctx_dim = context_dim or hidden_dim
    self.to_q = Linear(hidden_dim, hidden_dim)
    self.to_k = Linear(ctx_dim, hidden_dim)
    self.to_v = Linear(ctx_dim, hidden_dim)
    self.out = Linear(hidden_dim, hidden_dim)

  def __call__(self, x:Tensor, context:Optional[Tensor]=None) -> Tensor:
    b, n, _ = x.shape
    context = x if context is None else context
    q = self.to_q(x).reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
    k = self.to_k(context).reshape(b, -1, self.heads, self.dim_head).permute(0, 2, 3, 1)
    v = self.to_v(context).reshape(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
    attn = q @ k * self.scale
    weights = attn.softmax(axis=-1)
    out = (weights @ v).permute(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)
    return self.out(out)

class FeedForward:
  def __init__(self, hidden_dim:int, mlp_ratio:int=4):
    self.net = [
      Linear(hidden_dim, hidden_dim * mlp_ratio),
      Linear(hidden_dim * mlp_ratio, hidden_dim),
    ]

  def __call__(self, x:Tensor) -> Tensor:
    x = self.net[0](x).gelu()
    return self.net[1](x)

class TransformerBlock:
  def __init__(self, hidden_dim:int, heads:int, use_cross_attn:bool, mlp_ratio:int=4):
    self.norm1 = LayerNorm(hidden_dim)
    self.self_attn = MultiHeadAttention(hidden_dim, heads)
    self.use_cross_attn = use_cross_attn
    if use_cross_attn:
      self.cross_norm = LayerNorm(hidden_dim)
      self.cross_attn = MultiHeadAttention(hidden_dim, heads)
    self.norm2 = LayerNorm(hidden_dim)
    self.ff = FeedForward(hidden_dim, mlp_ratio)

  def __call__(self, x:Tensor, text_context:Optional[Tensor]=None) -> Tensor:
    x = x + self.self_attn(self.norm1(x))
    if self.use_cross_attn and text_context is not None:
      x = x + self.cross_attn(self.cross_norm(x), context=text_context)
    x = x + self.ff(self.norm2(x))
    return x

class FluxSchnellTiny:
  def __init__(self, latent_channels:int, text_embedding_dim:int):
    self.patch = PatchEmbed(latent_channels, TINY_DIM, TINY_PATCH)
    self.time_embed = TimeEmbedding(TINY_DIM)
    self.text_proj = Linear(text_embedding_dim, TINY_DIM)
    self.blocks:List[TransformerBlock] = []
    for i in range(TINY_DEPTH):
      use_cross = (i % TINY_CROSS_INTERVAL) == 0
      self.blocks.append(TransformerBlock(TINY_DIM, TINY_HEADS, use_cross))
    self.norm_out = LayerNorm(TINY_DIM)
    self.unpatch = PatchUnembed(latent_channels, TINY_DIM, TINY_PATCH)

  def forward(self, latent_tokens:Tensor, text_context:Tensor, timesteps:Tensor) -> Tensor:
    b, c, h, w = latent_tokens.shape
    x = self.patch(latent_tokens)
    t = self.time_embed(timesteps).unsqueeze(1)
    text_tokens = self.text_proj(text_context)
    x = x + t
    for block in self.blocks:
      x = block(x, text_tokens)
    x = self.norm_out(x)
    return self.unpatch(x, h, w)

  def __call__(self, latent_tokens:Tensor, text_context:Tensor, timesteps:Tensor) -> Tensor:
    return self.forward(latent_tokens, text_context, timesteps)
