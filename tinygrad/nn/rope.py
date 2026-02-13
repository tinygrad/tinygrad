from __future__ import annotations
import math, functools
from dataclasses import dataclass
from tinygrad import Tensor

@dataclass(frozen=True)
class YarnParams:
  freq_base: float = 10000.0
  factor: float = 1.0
  original_context_length: int = 4096
  beta_fast: float = 32.0
  beta_slow: float = 1.0
  attn_factor: float = 1.0
  yarn_log_multiplier: float = 0.0
  ext_factor: float = 1.0

def yarn_corr_dims(n_dims: int, n_ctx_orig: int, freq_base: float, beta_fast: float, beta_slow: float) -> tuple[float, float]:
  if freq_base <= 1.0 or n_ctx_orig <= 0 or n_dims <= 0: return (0.0, float(n_dims - 1))
  log_base = math.log(freq_base)
  two_pi = 2.0 * math.pi
  def corr_dim(beta: float) -> float:
    return 0.0 if beta <= 0 else n_dims * math.log(n_ctx_orig / (beta * two_pi)) / (2.0 * log_base)
  return (max(0.0, math.floor(corr_dim(beta_fast))), min(float(n_dims - 1), math.ceil(corr_dim(beta_slow))))

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scaling_factor: float = 1.0) -> Tensor:
  """Precompute RoPE frequencies. Returns (end, dim//2, 2) with cos/sin in last dim."""
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  if scaling_factor > 1.0: freqs = freqs / scaling_factor
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(end, dim//2, 2).contiguous()

@functools.cache
def precompute_freqs_cis_yarn(dim: int, end: int, p: YarnParams) -> Tensor:
  """Precompute RoPE frequencies with YaRN interpolation."""
  freq_scale = 1.0 if p.factor == 0.0 else 1.0 / p.factor
  corr_start, corr_end = yarn_corr_dims(dim, p.original_context_length, p.freq_base, p.beta_fast, p.beta_slow)
  mscale = p.attn_factor * (1.0 + 0.1 * math.log(1.0 / freq_scale)) if abs(p.ext_factor) > 1e-9 and freq_scale < 1.0 else p.attn_factor
  dim_indices = Tensor.arange(0, dim, 2)[:(dim // 2)]
  inv_freqs = 1.0 / (p.freq_base ** (dim_indices / dim))
  positions = Tensor.arange(end).float()
  theta_extrap = positions.unsqueeze(1) * inv_freqs.unsqueeze(0)
  denom = max(0.001, corr_end - corr_start)
  ramp_mix = (1.0 - ((dim_indices / 2.0 - corr_start) / denom).clip(0.0, 1.0)) * p.ext_factor
  theta_interp = freq_scale * theta_extrap
  theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix if abs(p.ext_factor) > 1e-9 else theta_interp
  return Tensor.stack(theta.cos() * mscale, theta.sin() * mscale, dim=-1).contiguous()

def precompute_rope_freqs_cis(dim: int, end: int, theta: float, yarn_params: YarnParams|None=None, yarn_scaling_factor: float=1.0) -> Tensor:
  """Unified RoPE frequency provider for standard and YaRN-scaled variants."""
  if yarn_params is not None: return precompute_freqs_cis_yarn(dim, end, yarn_params).realize()
  scaling = yarn_scaling_factor if yarn_scaling_factor > 1.0 else 1.0
  return precompute_freqs_cis(dim, end, theta, scaling).realize()

def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
  """Apply RoPE with half-split format: x split into [first_half | second_half] pairs."""
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis[..., 0].reshape(1, 1, x.shape[2], -1), freqs_cis[..., 1].reshape(1, 1, x.shape[2], -1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def apply_rope_interleaved(x: Tensor, freqs_cis: Tensor, use_float32: bool=True) -> Tensor:
  """
  Apply interleaved RoPE: x has alternating [x0, x1, x2, x3, ...] pairs.
  `freqs_cis` shape: (T, dim//2, 2).
  """
  cos = freqs_cis[..., 0].reshape(1, 1, x.shape[2], -1)
  sin = freqs_cis[..., 1].reshape(1, 1, x.shape[2], -1)
  assert x.shape[-1] % 2 == 0
  input_dtype = x.dtype
  if use_float32: x = x.float()
  x1, x2 = x[..., 0::2], x[..., 1::2]
  out1, out2 = x1 * cos - x2 * sin, x2 * cos + x1 * sin
  out = out1.unsqueeze(-1).cat(out2.unsqueeze(-1), dim=-1).flatten(-2)
  return out.cast(input_dtype) if use_float32 else out

def load_yarn_params_from_gguf(kv: dict, arch: str, rope_theta: float) -> tuple[YarnParams|None, float, float]:
  """Extract YaRN/RoPE scaling params from GGUF metadata. Returns (yarn_params, mscale, yarn_scaling_factor)."""
  def rk(s, d=None): return kv.get(f'{arch}.rope.scaling.{s}', d)
  scaling_factor, scaling_type = rk('factor', 1.0), rk('type')
  raw_log_mul = rk('yarn_log_multiplier')
  yarn_log_mul = raw_log_mul / 0.1 if raw_log_mul is not None else 0.0
  attn_factor = rk('attn_factor', 1.0)

  if scaling_type == "yarn" or scaling_factor > 1.0:
    freq_scale = (1.0 / scaling_factor) if scaling_factor > 0 else 1.0

    # compute yarn_attn_factor (llama.cpp cparams.yarn_attn_factor)
    if freq_scale >= 1.0: yarn_attn_factor = attn_factor
    else:
      inv = 1.0 / freq_scale
      def get_ms(s, m): return 1.0 if s <= 1.0 else (0.1 * m * math.log(s) + 1.0)
      if yarn_log_mul != 0.0:
        ms = yarn_log_mul if (arch == "deepseek2" and yarn_log_mul != 1.0) else 1.0
        yarn_attn_factor = get_ms(inv, ms) / get_ms(inv, yarn_log_mul) * attn_factor
        if not (arch == "deepseek2" and abs(ms - yarn_log_mul) < 1e-6):
          yarn_attn_factor /= (1.0 + 0.1 * math.log(inv))
      else:
        yarn_attn_factor = get_ms(inv, 1.0) / (1.0 + 0.1 * math.log(inv)) * attn_factor

    # compute mscale (attention scaling)
    if freq_scale >= 1.0: mscale = attn_factor
    else:
      log_inv = math.log(1.0 / freq_scale)
      mscale = attn_factor * (1.0 + 0.1 * log_inv) * (1.0 + 0.1 * yarn_log_mul * log_inv)

    yarn_params = YarnParams(freq_base=rope_theta, factor=scaling_factor, attn_factor=yarn_attn_factor, yarn_log_multiplier=yarn_log_mul,
      original_context_length=rk('original_context_length', 4096), beta_fast=rk('yarn_beta_fast', 32.0), beta_slow=rk('yarn_beta_slow', 1.0),
      ext_factor=rk('yarn_ext_factor', 1.0 if scaling_type == "yarn" else 0.0))
    return yarn_params, mscale, scaling_factor
  return None, attn_factor, 1.0
