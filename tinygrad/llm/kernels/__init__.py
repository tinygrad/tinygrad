"""Pregenerated templated UIR kernels for `tinygrad.llm` (zero runtime kernel builders, zero `slopkernels` imports).

Every `.uir` file in this directory is generated offline with `AddrSpace.TEMPLATE` parameters and
specialized at dispatch time via `UOp.specialize(...)`.
"""
from __future__ import annotations

import functools
import pathlib
import struct
from typing import Any

from tinygrad import UOp, dtypes
from tinygrad.dtype import least_upper_dtype
from tinygrad.helpers import prod
from tinygrad.uop.ops import Ops
from tinygrad.uop.uir import parse_uir

_DIR = pathlib.Path(__file__).resolve().parent

_QUANT_NAMES: dict[int, str] = {
  8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6", 23: "iq4_xs",
  16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s",
  22: "iq2_s", 39: "mxfp4", 40: "nvfp4",
}
_IQ_GRIDS = {16, 17, 18, 19, 21, 22}


@functools.cache
def load_uir(name:str) -> UOp|None:
  """Lazily load and parse a pregenerated templated `.uir` file by stem name."""
  p = _DIR / f"{name}.uir"
  return parse_uir(p.read_text(encoding="utf-8")) if p.is_file() else None


def _spec(name:str, **values:int|float) -> UOp|None:
  u = load_uir(name)
  return None if u is None else u.specialize(**values)


def _sum_last(u:UOp) -> UOp:
  nd = len(u.shape)
  return UOp(Ops.REDUCE, (u.permute((nd - 1, *range(nd - 1))),), (Ops.ADD, 1))


def _sum_first(u:UOp) -> UOp:
  return UOp(Ops.REDUCE, (u,), (Ops.ADD, 1))


@functools.cache
def _iq4_half_lut_uop(device:str|tuple[str, ...]) -> UOp:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  vals = [kvalues_iq4nl[k] for j in range(16) for i in range(16) for k in (i, j)]
  words = struct.unpack(f"<{len(vals) // 2}I", struct.pack(f"<{len(vals)}e", *vals))
  return UOp(Ops.STACK, tuple(UOp.const(w, dtypes.uint32) for w in words)).cast(dtypes.uint32).copy_to_device(device)


@functools.cache
def _iq_grid_uop(ggml_type:int, device:str|tuple[str, ...], half:bool=False) -> UOp:
  from tinygrad.runtime.autogen import ggml_common as ggml
  if half:
    hgrid, nb = (ggml.iq3xxs_grid, 4) if ggml_type == 18 else (ggml.iq2s_grid, 8)
    vals = [float((v >> (8 * i)) & 0xff) / 64.0 for v in hgrid for i in range(nb)]
    words = struct.unpack(f"<{len(vals) // 2}I", struct.pack(f"<{len(vals)}e", *vals))
    return UOp(Ops.STACK, tuple(UOp.const(w, dtypes.uint32) for w in words)).cast(dtypes.uint32).copy_to_device(device)
  grid, nwords = {16: (ggml.iq2xxs_grid, 2), 17: (ggml.iq2xs_grid, 2), 19: (ggml.iq1s_grid, 2),
                  22: (ggml.iq2s_grid, 2), 18: (ggml.iq3xxs_grid, 1), 21: (ggml.iq3s_grid, 1)}[ggml_type]
  words = tuple((v >> (32 * i)) & 0xffffffff for v in grid for i in range(nwords))
  return UOp(Ops.STACK, tuple(UOp.const(w, dtypes.uint32) for w in words)).cast(dtypes.uint32).copy_to_device(device)


def _q8_quantize(x:UOp, tokens:int, in_features:int, device:str|tuple[str, ...], axis:int|None=None) -> tuple[UOp, UOp, UOp]|None:
  k = _spec(f"amd_q8_quantize_{x.dtype.name}", tokens=tokens, in_features=in_features)
  if k is None: return None
  g = in_features // 32
  q = UOp.alloc((tokens, g, 8), dtypes.uint32, device=device, axis=axis)
  d = UOp.alloc((tokens, g), dtypes.float32, device=device, axis=axis)
  s = UOp.alloc((tokens, g, 2), dtypes.float32, device=device, axis=axis)
  c = k.call(q, d, s, x.reshape((tokens, in_features)) if axis is None else x)
  return q.after(c), d.after(c), s.after(c)


def _dispatch_dense_linear(target:tuple[int, ...], device:str|tuple[str, ...], srcs:tuple[UOp, ...]) -> UOp|None:
  if target[0] not in (9, 11, 12) or len(srcs) not in (2, 3): return None
  w, x = srcs[0], srcs[1]
  if len(w.shape) != 2 or w.axis is not None or x.axis is not None: return None
  out_f, in_f = int(w.shape[0]), int(w.shape[1])
  if in_f % 32 != 0 or int(x.shape[-1]) != in_f: return None
  if w.dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32) or x.dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32): return None
  dot_dt = least_upper_dtype(w.dtype, x.dtype)
  if dot_dt != dtypes.float32: return None
  tokens = int(prod(int(d) for d in x.shape[:-1]))
  if tokens >= 16: return None
  bias: UOp|None = None
  if len(srcs) == 3:
    bias = srcs[2]
    if bias.axis is not None or tuple(int(d) for d in bias.shape) != (out_f,): return None
    if bias.dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32): return None
    if least_upper_dtype(dot_dt, bias.dtype) != dtypes.float32: return None
  kname = f"amd_gemv_{w.dtype.name}_{x.dtype.name}{'' if in_f % 128 == 0 else '_v1'}"
  k = _spec(kname, tokens=tokens, out_features=out_f, in_features=in_f)
  if k is None: return None
  out = UOp.alloc((tokens, out_f), dtypes.float32, device=device)
  res = out.after(k.call(out, w, x.reshape((tokens, in_f))))
  if bias is not None: res = res + bias.cast(dtypes.float32)
  return res.reshape((*tuple(int(d) for d in x.shape[:-1]), out_f))


def _dispatch_quant_linear(target:tuple[int, ...], device:str|tuple[str, ...], srcs:tuple[UOp, ...],
                           ggml_type:int, in_features:int, out_features:int, groups:int=1, w_axis:int|None=None) -> UOp|None:
  if target[0] not in (9, 11, 12) or len(srcs) != 2 or ggml_type not in _QUANT_NAMES: return None
  w, x = srcs[0], srcs[1]
  x_shape, x_shard, x_axis = tuple(int(d) for d in x.shape), tuple(int(d) for d in x.max_shard_shape), x.axis
  if groups > 1 and (w_axis, x_axis) not in ((None, None), (0, len(x_shape) - 2)): return None
  rdna4, cdna = target[0] == 12, target[0] == 9
  ndev = len(device) if isinstance(device, tuple) else 1
  w_len = int(prod(int(d) for d in w.max_shard_shape))
  qname = _QUANT_NAMES[ggml_type]
  rows = int(prod(x_shape[:-2]))
  x_u = x.reshape((rows, groups, -1)).permute((1, 0, 2)).reshape((groups * rows, in_features)) if groups > 1 else x
  tokens = int(prod(x_shape)) // in_features
  out_f, in_f = out_features // groups, x_shard[-1]
  splits = in_features // in_f
  if (w_axis == 1) != (splits > 1): return None
  local_out = out_f // ndev if ndev > 1 and w_axis == 0 and groups == 1 else out_f
  use_wmma = groups == 1 and tokens % 16 == 0 and local_out % 16 == 0
  axis = 0 if groups > 1 and x_axis is not None else None if w_axis is None else 1 - w_axis
  extra: list[UOp] = []
  if use_wmma:
    if ggml_type == 23: extra = [_iq4_half_lut_uop(device)]
    elif ggml_type in (18, 22): extra = [_iq_grid_uop(ggml_type, device, half=True)]
    elif ggml_type in _IQ_GRIDS: extra = [_iq_grid_uop(ggml_type, device)]
    out_shape: tuple[int, ...] = (splits * tokens, out_f)
    sh_out = tuple(s // ndev if axis is not None and i == axis else s for i, s in enumerate(out_shape))
    arch = "cdna" if cdna else "rdna4" if rdna4 else "rdna3"
    k = _spec(f"{arch}_gemm_{qname}_f16", tokens=sh_out[0], out_features=local_out, in_features=in_f, raw_len=w_len)
    if k is None: return None
    x_f16 = x_u.cast(dtypes.float16).contiguous()
    if x_axis is None: x_f16 = x_f16.reshape((sh_out[0], in_f))
    out = UOp.alloc(sh_out, dtypes.float32, device=device, axis=axis)
    res = out.after(k.call(out, w, x_f16, *extra))
  else:
    if ggml_type in _IQ_GRIDS: extra = [_iq_grid_uop(ggml_type, device)]
    x_tok_sh = int(prod(x_shard[:-1]))
    q_ax = 0 if groups > 1 and x_axis is not None else None
    q8 = _q8_quantize(x_u, x_tok_sh, in_f, device, axis=q_ax)
    if q8 is None: return None
    chunks = (in_f + 1023) // 1024
    out_shape = (splits * tokens, out_f, chunks)
    sh_out = tuple(s // ndev if axis is not None and i == axis else s for i, s in enumerate(out_shape))
    k = _spec(f"{'cdna' if cdna else 'rdna'}_gemv_{qname}", tokens=x_tok_sh, out_features=local_out,
              in_features=in_f, chunks=chunks, raw_len=w_len, group_rows=rows if groups > 1 else 0)
    if k is None: return None
    out = UOp.alloc(sh_out, dtypes.float32, device=device, axis=axis)
    res = out.after(k.call(out, w, *q8, *extra))
    res = _sum_last(res) if chunks > 1 else res.reshape((splits * tokens, out_f))
  if splits > 1: res = _sum_first(res.reshape((splits, tokens, out_f)))
  if groups > 1: res = res.reshape((groups, rows, out_f)).permute((1, 0, 2))
  return res.reshape((*x_shape[:-1], out_f))


def _dispatch_moe_linear(target:tuple[int, ...], device:str|tuple[str, ...], srcs:tuple[UOp, ...],
                         ggml_type:int, sel_shape:tuple[int, int, int], in_features:int, local_out:int, global_out:int,
                         w_axis:int|None=None, ndev:int=1) -> UOp|None:
  if target[0] not in (9, 11, 12) or len(srcs) != 3 or ggml_type not in _QUANT_NAMES: return None
  w, sel, x = srcs[0], srcs[1], srcs[2]
  cdna = target[0] == 9
  x_shape, x_shard = tuple(int(d) for d in x.shape), tuple(int(d) for d in x.max_shard_shape)
  B, T, slots = sel_shape
  w_len, local_in, x_slots = int(prod(int(d) for d in w.max_shard_shape)), x_shard[-1], x_shape[-2]
  splits = in_features // local_in
  if (w_axis == 2) != (splits > 1) or x_slots not in (1, slots): return None
  x_rows = B * T * x_slots
  q8 = _q8_quantize(x, x_rows, local_in, device)
  if q8 is None: return None
  chunks = (local_in + 1023) // 1024
  out_shape = (ndev * B * T * slots, global_out, splits * chunks)
  sh_out = tuple(s // (ndev if w_axis == 0 else splits if w_axis == 2 else (global_out // local_out))
                 if w_axis is not None and i == w_axis else s for i, s in enumerate(out_shape))
  extra = [_iq_grid_uop(ggml_type, device)] if ggml_type in _IQ_GRIDS else []
  k = _spec(f"{'cdna' if cdna else 'rdna'}_moe_{_QUANT_NAMES[ggml_type]}",
            rows=B * T * slots, x_rows=x_rows, out_features=local_out, in_features=local_in,
            chunks=chunks, raw_len=w_len, x_share=slots // x_slots, ndev=ndev)
  if k is None: return None
  out = UOp.alloc(sh_out, dtypes.float32, device=device, axis=w_axis)
  res = _sum_last(out.after(k.call(out, w, *q8, sel, *extra)))
  return res.reshape((ndev, B, T, slots, global_out) if ndev > 1 else (B, T, slots, global_out))


def _dispatch_flash_attention(target:tuple[int, ...], device:str|tuple[str, ...], srcs:tuple[UOp, ...],
                              valid_kv_len:int|None=None, valid_kv_offset:int=0, q_start_from_end:bool=False,
                              decode:bool=False) -> UOp|None:
  if target[0] not in (9, 11, 12) or len(srcs) < 2: return None
  rdna4, cdna = target[0] == 12, target[0] == 9
  q, kv = srcs[0], srcs[1]
  q_shape, q_shard, q_axis = tuple(int(d) for d in q.shape), tuple(int(d) for d in q.max_shard_shape), q.axis
  kv_shard = tuple(int(d) for d in kv.max_shard_shape)
  if len(q_shape) != 4 or len(kv_shard) != 5: return None
  B, H, T, D = q_shape
  sh_b, sh_h, sh_m, _ = q_shard
  _, _, sh_hkv, N_phys, _ = kv_shard
  group = sh_h // sh_hkv
  sym_srcs = srcs[2:]
  vkl_uop: UOp
  q_len_uop: UOp|None = None
  if valid_kv_len is None:
    if not sym_srcs: return None
    vkl_uop = sym_srcs[0]
    if len(sym_srcs) > 1: q_len_uop = sym_srcs[1]
  else:
    vkl_uop = UOp.const(valid_kv_len)
    if sym_srcs: q_len_uop = sym_srcs[0]
  if decode:
    if sh_m != 1 or D not in (64, 128, 256) or group not in (1, 2, 4, 8) or N_phys % 64 != 0: return None
    chunks = min(48, (N_phys + 63) // 64)
    kp = _spec(f"amd_flash_decode_partial_g{group}_d{D}", B=sh_b, H_KV=sh_hkv, n_blks=N_phys // 64, C=chunks, valid_kv_offset=valid_kv_offset)
    kc = _spec("amd_flash_decode_combine", B=sh_b, H=sh_h, C=chunks, d_blks=D // 64, valid_kv_offset=valid_kv_offset)
    if kp is None or kc is None: return None
    partial = UOp.alloc((sh_b, sh_h, chunks, D), dtypes.float32, device=device, axis=q_axis)
    stats = UOp.alloc((sh_b, sh_h, chunks, 2), dtypes.float32, device=device, axis=q_axis)
    out = UOp.alloc(q_shard, dtypes.float32, device=device, axis=q_axis)
    cp = kp.call(partial, stats, q, kv, vkl_uop)
    return out.after(kc.call(out, partial.after(cp), stats.after(cp), vkl_uop))
  if cdna or D not in (64, 128, 256) or N_phys % 32 != 0 or T % 32 != 0: return None
  from_end = 1 if (q_len_uop is None or q_start_from_end) else 0
  k = _spec(f"{'rdna4' if rdna4 else 'rdna3'}_flash_attn_d{D}", BH=sh_b * sh_h, m_blks=T // 32, B=sh_b, H_KV=sh_hkv,
            n_blks=N_phys // 32, valid_kv_offset=valid_kv_offset, q_start_from_end=from_end)
  if k is None: return None
  out = UOp.alloc(q_shard, dtypes.float32, device=device, axis=q_axis).reshape((B * H, T, D))
  return out.after(k.call(out, q.reshape((B * H, T, D)), kv, vkl_uop, q_len_uop if q_len_uop is not None else UOp.const(T))).reshape((B, H, T, D))


def _dispatch_hc_coeffs(target:tuple[int, ...], device:str|tuple[str, ...], srcs:tuple[UOp, ...],
                        count:int, iters:int, eps:float, norm_eps:float) -> UOp|None:
  if target[0] not in (9, 11, 12) or len(srcs) != 4 or count != 4: return None
  mix, flat, base, scale = srcs
  tokens, features = int(mix.shape[0]), int(flat.shape[-1])
  if features % 32 != 0: return None
  k = _spec(f"amd_hc_coeffs_{base.dtype.name}_{scale.dtype.name}",
            tokens=tokens, features=features, iters=iters, eps=float(eps), norm_eps=float(norm_eps))
  if k is None: return None
  sq = _sum_last((flat * flat).reshape((tokens, 32, features // 32))).contiguous()
  out = UOp.alloc((tokens, 24), dtypes.float32, device=device)
  return out.after(k.call(out, mix, sq, base, scale))


def _dispatch_gated_delta_prefill(target:tuple[int, ...], device:str|tuple[str, ...], srcs:tuple[UOp, ...],
                                  start_pos:int|None=None) -> UOp|None:
  if target[0] not in (9, 11, 12) or len(srcs) not in (6, 7): return None
  if any(s.axis is not None or s.dtype != dtypes.float32 for s in srcs[:6]): return None
  q, k_t, v, beta, alpha, state = srcs[:6]
  batch, heads, tokens, key_dim = (int(d) for d in q.shape)
  value_dim = int(v.shape[-1])
  if key_dim not in (32, 64, 128, 256) or value_dim % 4 != 0: return None
  col_alpha = len(alpha.shape) == 4 and int(alpha.shape[-1]) == key_dim
  if not col_alpha and len(alpha.shape) != 3 and not (len(alpha.shape) == 4 and int(alpha.shape[-1]) == 1): return None
  kern = _spec(f"amd_gated_delta_k{key_dim}_{'ak' if col_alpha else 'a1'}",
               batch=batch, heads=heads, tokens=tokens, v_tiles=value_dim // 4)
  if kern is None: return None
  kq = _sum_last(q * k_t).contiguous()
  out = UOp.alloc((batch, heads, tokens, value_dim), dtypes.float32, device=device)
  sp_uop = srcs[6] if len(srcs) == 7 else UOp.const(1 if start_pos is None else int(start_pos))
  return out.after(kern.call(out, q, k_t, v, beta, alpha.reshape((batch, heads, tokens, key_dim) if col_alpha else (batch, heads, tokens)),
                             state, kq, sp_uop))


_DISPATCHERS: dict[str, Any] = {
  "dense_linear": _dispatch_dense_linear,
  "quant_linear": _dispatch_quant_linear,
  "moe_linear": _dispatch_moe_linear,
  "flash_attention_op": _dispatch_flash_attention,
  "hc_coeffs_op": _dispatch_hc_coeffs,
  "gated_delta_prefill_op": _dispatch_gated_delta_prefill,
}


def dispatch(name:str, target:tuple[int, ...], device:str|tuple[str, ...], *srcs:UOp, **attrs:Any) -> UOp|None:
  """Dispatch a named LLM op using pregenerated templated `.uir` kernels from `tinygrad/llm/kernels/`."""
  fn = _DISPATCHERS.get(name)
  return None if fn is None else fn(target, device, srcs, **attrs)
