from dataclasses import dataclass
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import Tensor, dtypes
from typing import Optional
import tinygrad.nn as nn

def rope(pos:Tensor, dim:int, theta:int) -> Tensor:
  assert dim % 2 == 0
  scale = Tensor.arange(0, dim, step=2, dtype=dtypes.float64, device=pos.device) / dim
  omega = 1.0 / (theta ** scale)
  out = Tensor.einsum("...n,d->...nd", pos, omega)
  out = Tensor.stack(out.cos(), -out.sin(), out.sin(), out.cos(), dim=-1)
  out = out.rearrange("b n d (i j) -> b n d i j", i=2, j=2)
  return out.float()

def apply_rope(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk[..., 1]
  return xq_out.reshape(*xq.shape).cast(xq.dtype), xk_out.reshape(*xk.shape).cast(xk.dtype)

def attention(q:Tensor, k:Tensor, v:Tensor, pe:Tensor):
  q, k = apply_rope(q, k, pe)
  x = Tensor.scaled_dot_product_attention(q, k, v)
  return x.rearrange("B H L D -> B L (H D)")


class EmbedND:
  def __init__(self, dim:int, theta:int, axes_dim:list[int]):
    self.dim, self.theta, self.axes_dim = dim, theta, axes_dim

  def __call__(self, ids:Tensor):
    emb = Tensor.cat(*[rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])])
    return emb.unsqueeze(1)


class MLPEmbedder:
  def __init__(self, in_dim:int, hidden_dim:int):
    self.in_layer, self.out_layer = nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)

  def __call__(self, x:Tensor):
    return self.out_layer(self.in_layer(x).silu())


@dataclass
class ModulationOut:
  shift:Tensor
  scale:Tensor
  gate:Tensor


class Modulation:
  def __init__(self, dim:int, double:bool):
    self.is_double, self.multiplier = double, 6 if double else 3
    self.lin = nn.Linear(dim, self.multiplier * dim)

  def __call__(self, vec:Tensor):
    out = self.lin(vec.silu())[:, None, :].chunk(self.mulitpler, dim=-1)
    return (ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None)


class QKNorm:
  def __init__(self, dim:int):
    self.query_norm, self.key_norm = nn.RMSNorm(dim), nn.RMSNorm(dim)

  def __call__(self, q:Tensor, k:Tensor, v:Tensor):
    q, k = self.query_norm(q), self.key_norm(k)
    return (q.cast(v.dtype), k.cast(v.dtype))


class SelfAttention:
  def __init__(self, dim:int, num_heads:int=8, qkv_bias:bool = False):
    self.num_heads = num_heads
    self.qkv, self.norm, self.proj = nn.Linear(dim, dim * 3, bias=qkv_bias), QKNorm(dim // num_heads), nn.Linear(dim, dim)

  def __call__(self, x:Tensor, pe:Tensor):
    qkv = self.qkv(x)
    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k, v)
    x = attention(q, k, v, pe)
    return self.proj(x)


class DoubleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float, qkv_bias:bool = False):
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.num_heads = num_heads
    self.img_mod = Modulation(hidden_size, True)
    self.img_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.img_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.img_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.img_mlp = [nn.Linear(hidden_size, mlp_hidden_dim), Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size)]

    self.txt_mod = Modulation(hidden_size, True)
    self.txt_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.txt_attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    self.txt_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.txt_mlp = [nn.Linear(hidden_size, mlp_hidden_dim), Tensor.gelu, nn.Linear(mlp_hidden_dim, hidden_size)]

  def __call__(self, img:Tensor, txt:Tensor, vec:Tensor, pe:Tensor):
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    q, k, v = txt_q.cat(img_q, dim=2), txt_k.cat(img_k, dim=2), txt_v.cat(img_v, dim=2)

    attn = attention(q, k, v, pe)
    txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * ((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift).sequential(self.img_mlp)

    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt = txt + txt_mod2.gate * ((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift).sequential(self.txt_mlp)

    return img, txt


class SingleStreamBlock:
  def __init__(self, hidden_size:int, num_heads:int, mlp_ratio:float = 4.0, qk_scale:Optional[float] = None):
    head_dim = hidden_size // num_heads
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
    self.linear1, self.linear2 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim), nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
    self.norm = QKNorm(head_dim)
    self.pre_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.modulation = Modulation(hidden_size, False)

  def __call__(self, x:Tensor, vec:Tensor, pe:Tensor):
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = self.linear1(x_mod).split([3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.rearrange("B L (K H D) -> K B H L D", K=3, H=self.num_heads)
    q, k = self.norm(q, k, v)

    attn = attention(q, k , v, pe)
    output = self.linear2(attn.cat(mlp.gelu(), dim=2))
    return x + mod.gate * output


class LastLayer:
  def __init__(self, hidden_size:int, patch_size:int, out_channels:int):
    self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation = [Tensor.silu, nn.Linear(hidden_size, 2 * hidden_size)]

  def __call__(self, x:Tensor, vec:Tensor):
    shift, scale = vec.sequential(self.adaLN_modulation).chunk(2, dim=1)
    x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
    return self.linear(x)


class Flux:
  def __init__(
    self, in_channels:int, vec_in_dim:int, context_in_dim:int, hidden_size:int, mlp_ratio:float, num_heads:int, 
    depth:int, depth_single_blocks:int, axes_dim: list[int], theta:int, qkv_bias:bool, guidance_embed:bool
  ):
    self.pe_embedder = EmbedND(hidden_size // num_heads, theta, axes_dim)
    self.img_in = nn.Linear(in_channels, hidden_size)
    self.time_in = MLPEmbedder(256, hidden_size)
    self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
    self.guidance_in = MLPEmbedder(256, hidden_size) if guidance_embed else lambda x: x
    self.txt_in = nn.Linear(context_in_dim, hidden_size)
    self.double_blocks = [DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias=qkv_bias) for _ in range(depth)]
    self.single_blocks = [SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth_single_blocks)]
    self.final_layer = LastLayer(hidden_size, 1, in_channels)

  def __call__(self, img:Tensor, img_ids:Tensor, txt:Tensor, txt_ids:Tensor, timesteps:Tensor, y:Tensor, guidance:Optional[Tensor] = None):
    if img.ndim != 3 or txt.ndim != 3: raise ValueError("Input img and txt tensors must have 3 dimensions.")

    img = self.img_in(img)


if __name__ == "__main__":
  weights_fn = fetch("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors?download=true")
  state_dict = safe_load(weights_fn)
  for key in list(state_dict.keys()):
    if "scale" in key:
      new_key = key.replace("scale", "weight")
      state_dict[new_key] = state_dict.pop(key)
  model = Flux(
    in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0, num_heads=24,
    depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True, guidance_embed=False
  )
  load_state_dict(model, state_dict)
