"""The model and math portion of the MLPerf FLUX rectified-flow transformer."""

import math
from dataclasses import dataclass

from tinygrad import Tensor, dtypes, nn


def _xavier(linear:nn.Linear) -> None:
  bound = math.sqrt(6.0 / (linear.weight.shape[0] + linear.weight.shape[1]))
  linear.weight = Tensor.uniform(*linear.weight.shape, low=-bound, high=bound)
  if linear.bias is not None: linear.bias = Tensor.zeros(*linear.bias.shape)


def _normal(linear:nn.Linear, std:float=0.02) -> None:
  linear.weight = Tensor.normal(*linear.weight.shape, mean=0.0, std=std)
  if linear.bias is not None: linear.bias = Tensor.zeros(*linear.bias.shape)


def attention(q:Tensor, k:Tensor, v:Tensor, pe:Tensor) -> Tensor:
  q, k = apply_rope(q, k, pe)
  return q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def rope(pos:Tensor, dim:int, theta:int) -> Tensor:
  if dim % 2: raise ValueError(f"RoPE dimension must be even, got {dim}")
  scale = Tensor.arange(0, dim, 2, dtype=pos.dtype).to(pos.device) / dim
  omega = theta ** (-scale)
  out = Tensor.einsum("...n,d->...nd", pos, omega)
  out = Tensor.stack(out.cos(), -out.sin(), out.sin(), out.cos(), dim=-1)
  return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
  xq_float = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_float = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_float[..., 0] + freqs_cis[..., 1] * xq_float[..., 1]
  xk_out = freqs_cis[..., 0] * xk_float[..., 0] + freqs_cis[..., 1] * xk_float[..., 1]
  return xq_out.reshape(*xq.shape).cast(xq.dtype), xk_out.reshape(*xk.shape).cast(xk.dtype)


class EmbedND:
  def __init__(self, dim:int, theta:int, axes_dim:tuple[int, ...]|list[int]):
    if any(axis % 2 for axis in axes_dim): raise ValueError(f"every axes_dim must be even, got {axes_dim}")
    if sum(axes_dim) != dim: raise ValueError(f"Got {axes_dim} but expected positional dim {dim}")
    self.dim, self.theta, self.axes_dim = dim, theta, tuple(axes_dim)

  def __call__(self, ids:Tensor) -> Tensor:
    if ids.ndim != 3 or ids.shape[-1] != len(self.axes_dim):
      raise ValueError(f"IDs must have {len(self.axes_dim)} axes, got shape {ids.shape}")
    return Tensor.cat(*[rope(ids[..., i], axis_dim, self.theta) for i, axis_dim in enumerate(self.axes_dim)], dim=-3).unsqueeze(1)


def timestep_embedding(t:Tensor, dim:int, max_period:int=10_000, time_factor:float=1000.0) -> Tensor:
  if t.ndim != 1: raise ValueError(f"timesteps must be one-dimensional, got shape {t.shape}")
  t = time_factor * t
  half = dim // 2
  freqs = (-math.log(max_period) * Tensor.arange(half, dtype=dtypes.float32) / half).exp().to(t.device)
  args = t[:, None].float() * freqs[None]
  embedding = Tensor.cat(args.cos(), args.sin(), dim=-1)
  if dim % 2: embedding = Tensor.cat(embedding, Tensor.zeros_like(embedding[:, :1]), dim=-1)
  return embedding.cast(t.dtype) if t.is_floating_point() else embedding


class MLPEmbedder:
  def __init__(self, in_dim:int, hidden_dim:int):
    self.in_layer, self.out_layer = nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)

  def init_weights(self) -> None:
    _normal(self.in_layer)
    _normal(self.out_layer)

  def __call__(self, x:Tensor) -> Tensor: return self.out_layer(self.in_layer(x).silu())


class QKNorm:
  def __init__(self, dim:int):
    self.query_norm, self.key_norm = nn.RMSNorm(dim), nn.RMSNorm(dim)

  def init_weights(self) -> None:
    self.query_norm.weight = Tensor.ones(*self.query_norm.weight.shape)
    self.key_norm.weight = Tensor.ones(*self.key_norm.weight.shape)

  def __call__(self, q:Tensor, k:Tensor, v:Tensor) -> tuple[Tensor, Tensor]:
    # Pinned TorchTitan on torch 2.9.1 uses the float32 accumulator epsilon for BF16 RMSNorm.
    eps = 2.0 ** -dtypes.finfo(dtypes.float32)[1]
    q = q * (q.square().mean(-1, keepdim=True) + eps).rsqrt() * self.query_norm.weight
    k = k * (k.square().mean(-1, keepdim=True) + eps).rsqrt() * self.key_norm.weight
    return q.cast(v.dtype), k.cast(v.dtype)


class SelfAttention:
  def __init__(self, dim:int, num_heads:int=8, qkv_bias:bool=False):
    self.num_heads = num_heads
    self.qkv, self.norm, self.proj = nn.Linear(dim, dim * 3, bias=qkv_bias), QKNorm(dim // num_heads), nn.Linear(dim, dim)

  def init_weights(self) -> None:
    _xavier(self.qkv)
    _xavier(self.proj)
    self.norm.init_weights()

  def __call__(self, x:Tensor, pe:Tensor) -> Tensor:
    q, k, v = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)
    return self.proj(attention(q, k, v, pe))


@dataclass
class ModulationOut:
  shift:Tensor
  scale:Tensor
  gate:Tensor


class Modulation:
  def __init__(self, dim:int, double:bool):
    self.is_double, self.multiplier = double, 6 if double else 3
    self.lin = nn.Linear(dim, self.multiplier * dim)

  def init_weights(self) -> None:
    self.lin.weight = Tensor.zeros(*self.lin.weight.shape)
    self.lin.bias = Tensor.zeros(*self.lin.bias.shape)

  def __call__(self, vec:Tensor) -> tuple[ModulationOut, ModulationOut|None]:
    out = self.lin(vec.silu())[:, None, :].chunk(self.multiplier, dim=-1)
    return ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None


class DoubleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float, qkv_bias:bool=False):
    mlp_hidden = int(hidden_size * mlp_ratio)
    self.num_heads, self.hidden_size = num_heads, hidden_size
    self.img_mod, self.txt_mod = Modulation(hidden_size, True), Modulation(hidden_size, True)
    self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
    self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
    self.img_mlp = (nn.Linear(hidden_size, mlp_hidden), nn.Linear(mlp_hidden, hidden_size))
    self.txt_mlp = (nn.Linear(hidden_size, mlp_hidden), nn.Linear(mlp_hidden, hidden_size))

  def init_weights(self) -> None:
    for linear in (*self.img_mlp, *self.txt_mlp): _xavier(linear)
    for module in (self.img_mod, self.txt_mod, self.img_attn, self.txt_attn): module.init_weights()

  def _qkv(self, x:Tensor, attn:SelfAttention) -> tuple[Tensor, Tensor, Tensor]:
    q, k, v = attn.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = attn.norm(q, k, v)
    return q, k, v

  @staticmethod
  def _mlp(x:Tensor, layers:tuple[nn.Linear, nn.Linear]) -> Tensor: return layers[1](layers[0](x).gelu(approximate="tanh"))

  def __call__(self, img:Tensor, txt:Tensor, vec:Tensor, pe:Tensor) -> tuple[Tensor, Tensor]:
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    assert img_mod2 is not None and txt_mod2 is not None
    img_q, img_k, img_v = self._qkv((1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift, self.img_attn)
    txt_q, txt_k, txt_v = self._qkv((1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift, self.txt_attn)
    attn = attention(Tensor.cat(txt_q, img_q, dim=2), Tensor.cat(txt_k, img_k, dim=2), Tensor.cat(txt_v, img_v, dim=2), pe)
    txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self._mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift, self.img_mlp)
    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * self._mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift, self.txt_mlp)
    return img, txt


class SingleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float=4.0):
    self.hidden_size, self.num_heads = hidden_size, num_heads
    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
    self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
    self.norm = QKNorm(hidden_size // num_heads)
    self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.modulation = Modulation(hidden_size, False)

  def init_weights(self) -> None:
    _xavier(self.linear1)
    _xavier(self.linear2)
    self.norm.init_weights()
    self.modulation.init_weights()

  def __call__(self, x:Tensor, vec:Tensor, pe:Tensor) -> Tensor:
    mod, _ = self.modulation(vec)
    qkv, mlp = self.linear1((1 + mod.scale) * self.pre_norm(x) + mod.shift).split([3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
    q, k, v = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)
    output = self.linear2(Tensor.cat(attention(q, k, v, pe), mlp.gelu(approximate="tanh"), dim=2))
    return x + mod.gate * output


class LastLayer:
  def __init__(self, hidden_size:int, patch_size:int, out_channels:int):
    self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
    self.adaLN_modulation = nn.Linear(hidden_size, 2 * hidden_size)

  def init_weights(self) -> None:
    self.adaLN_modulation.weight = Tensor.zeros(*self.adaLN_modulation.weight.shape)
    self.adaLN_modulation.bias = Tensor.zeros(*self.adaLN_modulation.bias.shape)
    self.linear.weight = Tensor.zeros(*self.linear.weight.shape)
    self.linear.bias = Tensor.zeros(*self.linear.bias.shape)

  def __call__(self, x:Tensor, vec:Tensor) -> Tensor:
    shift, scale = self.adaLN_modulation(vec.silu()).chunk(2, dim=1)
    return self.linear((1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :])


class Flux:
  def __init__(self, guidance_embed:bool=False, in_channels:int=64, out_channels:int|None=None, vec_in_dim:int=768,
               context_in_dim:int=4096, hidden_size:int=3072, mlp_ratio:float=4.0, num_heads:int=24, depth:int=19,
               depth_single_blocks:int=38, axes_dim:tuple[int, ...]|list[int]=(16, 56, 56), theta:int=10_000,
               qkv_bias:bool=True):
    if hidden_size % num_heads: raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")
    if any(axis % 2 for axis in axes_dim): raise ValueError(f"every axes_dim must be even, got {axes_dim}")
    pe_dim = hidden_size // num_heads
    if sum(axes_dim) != pe_dim: raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
    self.guidance_embed, self.in_channels = guidance_embed, in_channels
    self.out_channels, self.hidden_size, self.num_heads = out_channels if out_channels is not None else in_channels, hidden_size, num_heads
    self.pe_embedder = EmbedND(pe_dim, theta, axes_dim)
    self.img_in, self.txt_in = nn.Linear(in_channels, hidden_size), nn.Linear(context_in_dim, hidden_size)
    self.time_in, self.vector_in = MLPEmbedder(256, hidden_size), MLPEmbedder(vec_in_dim, hidden_size)
    self.guidance_in = MLPEmbedder(256, hidden_size) if guidance_embed else None
    self.double_blocks = [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias) for _ in range(depth)]
    self.single_blocks = [SingleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth_single_blocks)]
    self.final_layer = LastLayer(hidden_size, 1, self.out_channels)
    self.init_weights()

  def init_weights(self) -> None:
    _xavier(self.img_in)
    _xavier(self.txt_in)
    self.time_in.init_weights()
    self.vector_in.init_weights()
    if self.guidance_in is not None: self.guidance_in.init_weights()
    for block in (*self.single_blocks, *self.double_blocks): block.init_weights()
    self.final_layer.init_weights()

  def __call__(self, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, timesteps:Tensor, y:Tensor,
               guidance:Tensor|None=None) -> Tensor:
    if img.ndim != 3 or txt.ndim != 3: raise ValueError("Input img and txt tensors must have 3 dimensions")
    batch = img.shape[0]
    if txt.shape[0] != batch: raise ValueError("img and txt batch sizes must match")
    for name, ids, tokens in (("img", img_ids, img.shape[1]), ("txt", txt_ids, txt.shape[1])):
      if ids.ndim != 3 or ids.shape[0] != batch or ids.shape[1] != tokens:
        raise ValueError(f"{name} IDs token and batch dimensions must match {name}, got {ids.shape}")
      if ids.shape[2] != len(self.pe_embedder.axes_dim): raise ValueError(f"{name} IDs must have {len(self.pe_embedder.axes_dim)} axes")
    if timesteps.shape != (batch,) or y.shape[0] != batch: raise ValueError("timesteps and vector conditioning must match batch size")
    if self.guidance_embed and guidance is None: raise ValueError("guidance strength is required for a guidance-distilled model")
    if guidance is not None and guidance.shape != (batch,): raise ValueError("guidance must contain one value per batch element")
    model_dtype = self.img_in.weight.dtype
    img, txt, y, timesteps = img.cast(model_dtype), txt.cast(model_dtype), y.cast(model_dtype), timesteps.cast(model_dtype)
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256)) + self.vector_in(y)
    if self.guidance_in is not None:
      assert guidance is not None
      vec = vec + self.guidance_in(timestep_embedding(guidance.cast(model_dtype), 256))
    txt = self.txt_in(txt)
    pe = self.pe_embedder(Tensor.cat(txt_ids.cast(model_dtype), img_ids.cast(model_dtype), dim=1))
    for block in self.double_blocks: img, txt = block(img, txt, vec, pe)
    img = Tensor.cat(txt, img, dim=1)
    for block in self.single_blocks: img = block(img, vec, pe)
    return self.final_layer(img[:, txt.shape[1]:], vec)

  forward = __call__


def rectified_flow_inputs(data:Tensor, noise:Tensor, t:Tensor) -> tuple[Tensor, Tensor]:
  if data.shape != noise.shape: raise ValueError("data and noise shapes must match")
  if t.shape != (data.shape[0],): raise ValueError("t must contain one value per batch element")
  t_view = t.reshape(t.shape[0], *([1] * (data.ndim - 1)))
  return (1 - t_view) * data + t_view * noise, noise - data


def mse_loss(pred:Tensor, target:Tensor) -> Tensor:
  return ((pred.float() - target.detach().float()) ** 2).mean()
