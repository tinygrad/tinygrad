"""Logical multi-kernel op UIR builders for slopkernels.

Every builder takes pure metadata (`target`, `device`, `srcs`, plus pure semantic kwargs)
and returns a pure UOp graph (or `None` when unsupported on the target/shapes) — zero
Tensor, Buffer, Device, or getenv.

`srcs` is a tuple of pure per-input descriptors matching the positional arguments passed
to `parse_uir(uir).call_with_output(*srcs)`:
  - Tensor input:  `(shape, shard_shape, axis, dtype_name)`
  - Symbolic scalar UOp (`Variable`): `(vmin, vmax)`
"""
from __future__ import annotations

import functools
import re
import struct
from typing import cast

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, DType, dtypes, least_upper_dtype
from tinygrad.helpers import prod
from tinygrad.uop.ops import Ops, ParamArg

from slopkernels.src import amd as _amd
from slopkernels.src import amd_gguf as _gguf
from slopkernels.src import extra as _extra

DeviceSpec = str | tuple[str, ...]
TensorDesc = tuple[tuple[int, ...], tuple[int, ...], int | None, str]
SymDesc = tuple[int, int]
SrcDesc = TensorDesc | SymDesc

_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
def _dt(d:str|DType) -> DType: return d if isinstance(d, DType) else _DTYPES[d]

def _p(slot:int, shape:tuple[int, ...], dt:str|DType, device:DeviceSpec|None=None, axis:int|None=None) -> UOp:
  p = UOp.param(slot, _dt(dt), shape, device)
  return p.unshard(axis) if axis is not None and isinstance(device, tuple) else p

def _v(slot:int, vmin:int, vmax:int, name:str) -> UOp:
  return UOp.variable(name, vmin, vmax).replace(arg=ParamArg(slot, dtypes.weakint, (vmin, vmax), name, AddrSpace.ALU, None))

def _sum_last(u:UOp) -> UOp:
  nd = len(u.shape)
  return UOp(Ops.REDUCE, (u.permute((nd - 1, *range(nd - 1))),), (Ops.ADD, 1))

def _sum_first(u:UOp) -> UOp:
  return UOp(Ops.REDUCE, (u,), (Ops.ADD, 1))

def _word_dt(ggml_type:int) -> DType:
  return dtypes.uint8 if ggml_type == _gguf.MXFP4 else dtypes.uint16 if ggml_type in _gguf.HALFWORD_QUANTS else dtypes.uint32

def parse_amd_target(target:str|tuple[int, ...]) -> tuple[bool, bool]|None:
  """Return (rdna4, cdna) if `target` is RDNA3/RDNA4/CDNA3/4, else None."""
  if isinstance(target, tuple): major = target[0] if target else 0
  elif (m := re.search(r"gfx(\d+)", target)):
    digits = m.group(1)
    major = int(digits[:-2]) if len(digits) > 2 else int(digits[0])
  elif (m := re.match(r"\((\d+),", target)): major = int(m.group(1))
  else: return None
  if major not in (9, 11, 12): return None
  return (major == 12, major == 9)

@functools.cache
def _iq4_half_lut_uop(device:DeviceSpec) -> UOp:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  vals = [kvalues_iq4nl[k] for j in range(16) for i in range(16) for k in (i, j)]
  raw = struct.pack(f"<{len(vals)}e", *vals)
  words = struct.unpack(f"<{len(vals) // 2}I", raw)
  return UOp(Ops.STACK, tuple(UOp.const(w, dtypes.uint32) for w in words)).cast(dtypes.uint32).copy_to_device(device)

@functools.cache
def _iq_grid_uop(ggml_type:int, device:DeviceSpec, half:bool=False) -> UOp:
  from tinygrad.runtime.autogen import ggml_common as ggml
  if half:
    hgrid, nb = (ggml.iq3xxs_grid, 4) if ggml_type == _gguf.IQ3_XXS else (ggml.iq2s_grid, 8)
    vals = [float((v >> (8 * i)) & 0xff) / 64.0 for v in hgrid for i in range(nb)]
    raw = struct.pack(f"<{len(vals)}e", *vals)
    words = struct.unpack(f"<{len(vals) // 2}I", raw)
    return UOp(Ops.STACK, tuple(UOp.const(w, dtypes.uint32) for w in words)).cast(dtypes.uint32).copy_to_device(device)
  grid, nwords = {_gguf.IQ2_XXS: (ggml.iq2xxs_grid, 2), _gguf.IQ2_XS: (ggml.iq2xs_grid, 2), _gguf.IQ1_S: (ggml.iq1s_grid, 2),
                  _gguf.IQ2_S: (ggml.iq2s_grid, 2), _gguf.IQ3_XXS: (ggml.iq3xxs_grid, 1), _gguf.IQ3_S: (ggml.iq3s_grid, 1)}[ggml_type]
  words = tuple((v >> (32 * i)) & 0xffffffff for v in grid for i in range(nwords))
  return UOp(Ops.STACK, tuple(UOp.const(w, dtypes.uint32) for w in words)).cast(dtypes.uint32).copy_to_device(device)

def _q8_quantize_uops(x:UOp, tokens:int, in_features:int, in_shard:tuple[int, ...], in_dtype:str,
                      device:DeviceSpec, axis:int|None=None) -> tuple[UOp, UOp, UOp]:
  g = in_features // 32
  q = UOp.alloc((tokens, g, 8), dtypes.uint32, device=device, axis=axis)
  d = UOp.alloc((tokens, g), dtypes.float32, device=device, axis=axis)
  s = UOp.alloc((tokens, g, 2), dtypes.float32, device=device, axis=axis)
  k = _gguf._q8_quantize_kernel(_p(0, (tokens, g, 8), "u32"), _p(1, (tokens, g), "f32"),
                                _p(2, (tokens, g, 2), "f32"), _p(3, in_shard, in_dtype), tokens=tokens, in_features=in_features)
  c = k.call(q, d, s, x)
  return q.after(c), d.after(c), s.after(c)

# ******** Logical Op UIR Builders (return UOp or None if unsupported) ********

def build_dense_linear(*, target:str|tuple[int, ...], device:DeviceSpec="AMD", srcs:tuple[SrcDesc, ...]) -> UOp|None:
  """Dense GEMV (`tokens < 16`, unsharded f16/bf16/f32 weight). srcs: `(w, x)` or `(w, x, bias)`."""
  if parse_amd_target(target) is None or len(srcs) not in (2, 3): return None
  (w_shape, _, w_axis, w_dtype), (x_shape, _, x_axis, x_dtype) = cast(TensorDesc, srcs[0]), cast(TensorDesc, srcs[1])
  if len(w_shape) != 2 or w_axis is not None or x_axis is not None: return None
  out_features, in_features = w_shape
  if in_features % 32 != 0 or x_shape[-1] != in_features: return None
  if w_dtype not in ("f16", "f32", "bf16") or x_dtype not in ("f16", "f32", "bf16"): return None
  # Preserve Linear fallback semantics: both the dot-product dtype and final (post-bias) dtype must be f32.
  dot_dt = least_upper_dtype(_dt(w_dtype), _dt(x_dtype))
  if dot_dt != dtypes.float32: return None
  tokens = int(prod(x_shape[:-1]))
  if tokens >= 16: return None
  p_w, p_x = _p(0, (out_features, in_features), w_dtype, device), _p(1, x_shape, x_dtype, device)
  call_srcs = [p_w, p_x.reshape((tokens, in_features))]
  k_ps = [_p(0, (tokens, out_features), "f32"), _p(1, (out_features, in_features), w_dtype), _p(2, (tokens, in_features), x_dtype)]
  if len(srcs) == 3:
    b_shape, _, b_axis, bias_dtype = cast(TensorDesc, srcs[2])
    if b_axis is not None or b_shape != (out_features,) or bias_dtype not in ("f16", "f32", "bf16"): return None
    if least_upper_dtype(dot_dt, _dt(bias_dtype)) != dtypes.float32: return None
    call_srcs.append(_p(2, (out_features,), bias_dtype, device))
    k_ps.append(_p(3, (out_features,), bias_dtype))
  out = UOp.alloc((tokens, out_features), dtypes.float32, device=device)
  k = _amd._amd_f16_gemv_kernel(*k_ps, in_features=in_features, out_features=out_features, tokens=tokens)
  return out.after(k.call(out, *call_srcs)).reshape((*x_shape[:-1], out_features))

def build_quant_linear(*, target:str|tuple[int, ...], device:DeviceSpec="AMD", srcs:tuple[SrcDesc, ...],
                       ggml_type:int, in_features:int, out_features:int, groups:int=1, w_axis:int|None=None) -> UOp|None:
  """Quantized GGUF GEMV (decode) / WMMA GEMM (prefill). srcs: `(packed_weight, x)`."""
  arch = parse_amd_target(target)
  if arch is None or len(srcs) != 2 or ggml_type not in _gguf.QUANT_SIZES: return None
  (_, w_shard, _, _), (x_shape, x_shard, x_axis, x_dtype) = cast(TensorDesc, srcs[0]), cast(TensorDesc, srcs[1])
  if groups > 1 and (w_axis, x_axis) not in ((None, None), (0, len(x_shape) - 2)): return None
  rdna4, cdna = arch
  ndev = len(device) if isinstance(device, tuple) else 1
  w_len, w_dt = int(prod(w_shard)), _word_dt(ggml_type)
  p_w = _p(0, (w_len,), w_dt, device, axis=0 if w_axis is not None else None)
  p_x = _p(1, x_shard, x_dtype, device, axis=x_axis)
  rows = int(prod(x_shape[:-2]))
  x_u = p_x
  if groups > 1:
    x_u = p_x.reshape((rows, groups, -1)).permute((1, 0, 2)).reshape((groups * rows, in_features))
  tokens = int(prod(x_shape)) // in_features
  out_f, in_f = out_features // groups, x_shard[-1]
  splits = in_features // in_f
  if (w_axis == 1) != (splits > 1): return None
  local_out = out_f // ndev if ndev > 1 and w_axis == 0 and groups == 1 else out_f
  use_wmma = groups == 1 and tokens % 16 == 0 and local_out % 16 == 0
  axis = 0 if groups > 1 and x_axis is not None else None if w_axis is None else 1 - w_axis
  out_shape: tuple[int, ...] = (splits * tokens, out_f)
  extra: list[UOp] = []
  if use_wmma:
    if ggml_type == _gguf.IQ4_XS: extra = [_iq4_half_lut_uop(device)]
    elif ggml_type in (_gguf.IQ3_XXS, _gguf.IQ2_S): extra = [_iq_grid_uop(ggml_type, device, half=True)]
    elif ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ1_S, _gguf.IQ3_S): extra = [_iq_grid_uop(ggml_type, device)]
    x_f16 = x_u.cast(dtypes.float16).contiguous()
    call_srcs = [x_f16, *extra]
  else:
    if ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ3_XXS, _gguf.IQ1_S, _gguf.IQ3_S, _gguf.IQ2_S):
      extra = [_iq_grid_uop(ggml_type, device)]
    x_tok_sh = int(prod(x_shard[:-1]))
    q_ax = 0 if groups > 1 and x_axis is not None else None
    xq, xd, xs = _q8_quantize_uops(x_u, x_tok_sh, in_f, (x_tok_sh, in_f) if groups > 1 else x_shard, x_dtype, device, axis=q_ax)
    call_srcs = [xq, xd, xs, *extra]
    chunks = (in_f + 1023) // 1024
    if splits > 1 or chunks > 1: out_shape += (chunks,)
  sh_out = tuple(s // ndev if axis is not None and i == axis else s for i, s in enumerate(out_shape))
  out = UOp.alloc(sh_out, dtypes.float32, device=device, axis=axis)
  if use_wmma:
    kgroups, lanes = (4, 64) if cdna else (2, 32) if rdna4 else (1, 32)
    k_ps = [_p(0, sh_out, "f32"), _p(1, (w_len,), w_dt), _p(2, x_shard, "f16")]
    if extra: k_ps.append(_p(3, tuple(int(s) for s in extra[0].shape), "u32"))
    if ggml_type == _gguf.IQ4_XS:
      k = _gguf._iq4_linear_f16_wmma_kernel(*k_ps, out_features=local_out, in_features=in_f, kgroups=kgroups, lanes=lanes)
    elif ggml_type in (_gguf.Q4_K, _gguf.Q5_K):
      k = _gguf._q5_linear_f16_wmma_kernel(*k_ps, out_features=local_out, in_features=in_f, ggml_type=ggml_type, kgroups=kgroups, lanes=lanes)
    else:
      k = _gguf._quant_linear_f16_wmma_kernel(*k_ps, out_features=local_out, in_features=in_f, ggml_type=ggml_type, kgroups=kgroups, lanes=lanes)
  else:
    x_tok_sh, g = int(prod(x_shard[:-1])), in_f // 32
    k_ps = [_p(0, sh_out, "f32"), _p(1, (w_len,), w_dt), _p(2, (x_tok_sh, g, 8), "u32"), _p(3, (x_tok_sh, g), "f32"), _p(4, (x_tok_sh, g, 2), "f32")]
    if extra: k_ps.append(_p(5, tuple(int(s) for s in extra[0].shape), "u32"))
    k = _gguf._quant_decode_kernel(*k_ps, out_features=local_out, in_features=in_f, ggml_type=ggml_type,
                                   cdna=cdna, group_rows=rows if groups > 1 else 0)
  res = out.after(k.call(out, p_w, *call_srcs))
  if len(out_shape) == 3: res = _sum_last(res)
  if splits > 1: res = _sum_first(res.reshape((splits, tokens, out_f)))
  if groups > 1: res = res.reshape((groups, rows, out_f)).permute((1, 0, 2))
  return res.reshape((*x_shape[:-1], out_f))

def build_moe_linear(*, target:str|tuple[int, ...], device:DeviceSpec="AMD", srcs:tuple[SrcDesc, ...],
                     ggml_type:int, sel_shape:tuple[int, int, int], in_features:int, local_out:int, global_out:int,
                     w_axis:int|None=None, ndev:int=1) -> UOp|None:
  """Quantized GGUF MoE expert projection. srcs: `(packed_weight, sel_flat_i32, x)`."""
  arch = parse_amd_target(target)
  if arch is None or len(srcs) != 3 or ggml_type not in _gguf.QUANT_SIZES: return None
  _, cdna = arch
  (_, w_shard, _, _), _, (x_shape, x_shard, _, x_dtype) = (
    cast(TensorDesc, srcs[0]), cast(TensorDesc, srcs[1]), cast(TensorDesc, srcs[2]))
  B, T, slots = sel_shape
  w_len, local_in, x_slots = int(prod(w_shard)), x_shard[-1], x_shape[-2]
  splits = in_features // local_in
  if (w_axis == 2) != (splits > 1) or x_slots not in (1, slots): return None
  w_dt = _word_dt(ggml_type)
  p_w = _p(0, (w_len,), w_dt, device, axis=0 if w_axis is not None else None)
  p_sel = _p(1, (B * T * slots,), "i32", device)
  p_x = _p(2, (B, T, x_slots, local_in), x_dtype, device, axis=3 if splits > 1 else None)
  x_rows = B * T * x_slots
  xq, xd, xs = _q8_quantize_uops(p_x, x_rows, local_in, (B, T, x_slots, local_in), x_dtype, device)
  chunks = (local_in + 1023) // 1024
  out_shape = (ndev * B * T * slots, global_out, splits * chunks)
  sh_out = tuple(s // (ndev if w_axis == 0 else splits if w_axis == 2 else (global_out // local_out))
                 if w_axis is not None and i == w_axis else s for i, s in enumerate(out_shape))
  out = UOp.alloc(sh_out, dtypes.float32, device=device, axis=w_axis)
  extra = ([_iq_grid_uop(ggml_type, device)]
           if ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ3_XXS, _gguf.IQ1_S, _gguf.IQ3_S, _gguf.IQ2_S) else [])
  g = local_in // 32
  k_ps = [_p(0, sh_out, "f32"), _p(1, (w_len,), w_dt), _p(2, (x_rows, g, 8), "u32"), _p(3, (x_rows, g), "f32"),
          _p(4, (x_rows, g, 2), "f32"), _p(5, (B * T * slots,), "i32")]
  if extra: k_ps.append(_p(6, tuple(int(s) for s in extra[0].shape), "u32"))
  k = _gguf._moe_decode_kernel(*k_ps, out_features=local_out, in_features=local_in, ggml_type=ggml_type,
                               x_share=slots // x_slots, cdna=cdna, ndev=ndev)
  res = _sum_last(out.after(k.call(out, p_w, xq, xd, xs, p_sel, *extra)))
  return res.reshape((ndev, B, T, slots, global_out) if ndev > 1 else (B, T, slots, global_out))

def build_flash_attention(*, target:str|tuple[int, ...], device:DeviceSpec="AMD", srcs:tuple[SrcDesc, ...],
                          valid_kv_len:int|None=None, valid_kv_offset:int=0, q_start_from_end:bool=False, decode:bool=False) -> UOp|None:
  """Flash Attention prefill / 2-pass decode on KV cache. srcs: `(q_f16, kv_f16[, valid_kv_sym][, q_len_sym])`."""
  arch = parse_amd_target(target)
  if arch is None or len(srcs) < 2: return None
  rdna4, cdna = arch
  (q_shape, q_shard, q_axis, _), (_, kv_shard, _, _) = cast(TensorDesc, srcs[0]), cast(TensorDesc, srcs[1])
  if len(q_shape) != 4 or len(kv_shard) != 5: return None
  B, H, T, D = q_shape
  sh_b, sh_h, sh_m, _ = q_shard
  _, _, sh_hkv, N_phys, _ = kv_shard
  group = sh_h // sh_hkv
  if D % 32 != 0: return None
  p_q = _p(0, q_shard, "f16", device, axis=q_axis)
  p_kv = _p(1, kv_shard, "f16", device, axis=2 if q_axis is not None else None)
  sym_descs = [cast(SymDesc, s) for s in srcs[2:]]
  sym_args: list[UOp] = []
  vkl_spec: int|SymDesc
  q_len_spec: SymDesc|None = None
  if valid_kv_len is None:
    if not sym_descs: return None
    vkl_spec = sym_descs[0]
    sym_args.append(_v(2, vkl_spec[0], vkl_spec[1], "valid_kv_len"))
    if len(sym_descs) > 1:
      q_len_spec = sym_descs[1]
      sym_args.append(_v(3, q_len_spec[0], q_len_spec[1], "q_len"))
  else:
    vkl_spec = valid_kv_len
    if sym_descs:
      q_len_spec = sym_descs[0]
      sym_args.append(_v(2, q_len_spec[0], q_len_spec[1], "q_len"))
  if decode:
    if not (D & (D - 1) == 0 and N_phys % 64 == 0 and group * ((D + 4) * 2 + 8) <= 65536): return None
    waves = 16
    while waves * group * ((D + 4) * 2 + 8) > 65536: waves //= 2
    if waves <= 0 or group > waves * 32: return None
    chunks = min(48, (N_phys + 63) // 64)
    partial = UOp.alloc((sh_b * sh_m, sh_h, chunks, D), dtypes.float32, device=device, axis=q_axis)
    stats = UOp.alloc((sh_b * sh_m, sh_h, chunks, 2), dtypes.float32, device=device, axis=q_axis)
    out = UOp.alloc(q_shard, dtypes.float32, device=device, axis=q_axis)
    vkl_p: int|UOp = (_v(4, vkl_spec[0], vkl_spec[1], "valid_kv_len") + valid_kv_offset
                      if isinstance(vkl_spec, tuple) else vkl_spec + valid_kv_offset)
    vkl_c: int|UOp = (_v(3, vkl_spec[0], vkl_spec[1], "valid_kv_len") + valid_kv_offset
                      if isinstance(vkl_spec, tuple) else vkl_spec + valid_kv_offset)
    kp = _amd._amd_flash_attention_decode_partial(_p(0, (sh_b * sh_m, sh_h, chunks, D), "f32"),
                                                  _p(1, (sh_b * sh_m, sh_h, chunks, 2), "f32"),
                                                  _p(2, (sh_b, sh_h, sh_m, D), "f16"),
                                                  _p(3, kv_shard, "f16"),
                                                  valid_kv_len=vkl_p, max_kv_len=N_phys, block_n=64, waves=waves)
    cp = kp.call(partial, stats, p_q, p_kv, *sym_args[:1])
    kc = _amd._amd_flash_decode_combine(_p(0, (sh_b, sh_h, sh_m, D), "f32"),
                                        _p(1, (sh_b * sh_m, sh_h, chunks, D), "f32"),
                                        _p(2, (sh_b * sh_m, sh_h, chunks, 2), "f32"), valid_kv_len=vkl_c)
    return out.after(kc.call(out, partial.after(cp), stats.after(cp), *sym_args[:1]))
  if cdna or not (D >= 64 and 2 * (64 * (D + 4) + D * 36) <= 65536 and N_phys % 32 == 0 and T % 32 == 0): return None
  out = UOp.alloc(q_shard, dtypes.float32, device=device, axis=q_axis).reshape((B * H, T, D))
  k_vkl_base: int|UOp = _v(3, vkl_spec[0], vkl_spec[1], "valid_kv_len") if isinstance(vkl_spec, tuple) else vkl_spec
  vkl: int|UOp
  qs: int|UOp|None
  if q_len_spec is not None:
    k_qlen: UOp = _v(3 + (1 if isinstance(vkl_spec, tuple) else 0), q_len_spec[0], q_len_spec[1], "q_len")
    if q_start_from_end: vkl, qs = k_vkl_base + valid_kv_offset, k_vkl_base + valid_kv_offset - k_qlen
    else: vkl, qs = k_vkl_base + valid_kv_offset + k_qlen, k_vkl_base + valid_kv_offset
  else:
    vkl, qs = k_vkl_base + valid_kv_offset, None
  k = _amd._amd_flash_attention(_p(0, (sh_b * sh_h, T, D), "f32"), _p(1, (sh_b * sh_h, T, D), "f16"),
                                _p(2, kv_shard, "f16"), valid_kv_len=vkl, q_start=qs, rdna4=rdna4)
  return out.after(k.call(out, p_q.reshape((B * H, T, D)), p_kv, *sym_args)).reshape((B, H, T, D))

def build_hc_coeffs(*, target:str|tuple[int, ...], device:DeviceSpec="AMD", srcs:tuple[SrcDesc, ...],
                    count:int, iters:int, eps:float, norm_eps:float) -> UOp|None:
  """Hyper-connection Sinkhorn coefficients. srcs: `(mix_f32, flat_f32, base, scale)`."""
  if parse_amd_target(target) is None or len(srcs) != 4 or count != 4: return None
  (mix_shape, _, _, _), (flat_shape, _, _, _), (_, _, _, base_dtype), (_, _, _, scale_dtype) = (
    cast(TensorDesc, srcs[0]), cast(TensorDesc, srcs[1]), cast(TensorDesc, srcs[2]), cast(TensorDesc, srcs[3]))
  tokens, features = mix_shape[0], flat_shape[-1]
  if features % 32 != 0: return None
  n = count * (2 + count)
  p_mix, p_flat = _p(0, (tokens, n), "f32", device), _p(1, (tokens, features), "f32", device)
  p_base, p_scale = _p(2, (n,), base_dtype, device), _p(3, (3,), scale_dtype, device)
  sq = _sum_last((p_flat * p_flat).reshape((tokens, 32, features // 32))).contiguous()
  out = UOp.alloc((tokens, n), dtypes.float32, device=device)
  k = _extra._hc_coeffs_kernel(_p(0, (tokens, n), "f32"), _p(1, (tokens, n), "f32"), _p(2, (tokens, 32), "f32"),
                               _p(3, (n,), base_dtype), _p(4, (3,), scale_dtype),
                               count=count, iters=iters, eps=eps, norm_eps=norm_eps, features=features)
  return out.after(k.call(out, p_mix, sq, p_base, p_scale))

def build_gated_delta_prefill(*, target:str|tuple[int, ...], device:DeviceSpec="AMD", srcs:tuple[SrcDesc, ...],
                              start_pos:int|None=None) -> UOp|None:
  """Fused Gated Delta Net recurrent scan (mutates state in place). srcs: `(q, k, v, beta, alpha, state[, start_pos_sym])`."""
  if parse_amd_target(target) is None or len(srcs) not in (6, 7): return None
  if any(cast(TensorDesc, s)[2] is not None for s in srcs[:6]): return None
  (q_shape, _, _, _), _, (v_shape, _, _, v_dtype), (_, _, _, beta_dtype), (alpha_shape, _, _, alpha_dtype), (_, _, _, state_dtype) = (
    cast(TensorDesc, srcs[0]), cast(TensorDesc, srcs[1]), cast(TensorDesc, srcs[2]),
    cast(TensorDesc, srcs[3]), cast(TensorDesc, srcs[4]), cast(TensorDesc, srcs[5]))
  batch, heads, tokens, key_dim = q_shape
  value_dim = v_shape[-1]
  if key_dim % 32 != 0 or value_dim % 4 != 0: return None
  p_q, p_k = _p(0, (batch, heads, tokens, key_dim), "f32", device), _p(1, (batch, heads, tokens, key_dim), "f32", device)
  p_v, p_beta = _p(2, (batch, heads, tokens, value_dim), v_dtype, device), _p(3, (batch, heads, tokens), beta_dtype, device)
  p_alpha, p_state = _p(4, alpha_shape, alpha_dtype, device), _p(5, (batch, heads, value_dim, key_dim), state_dtype, device)
  kq = _sum_last(p_q * p_k).contiguous()
  out = UOp.alloc((batch, heads, tokens, value_dim), _dt(v_dtype), device=device)
  sp_sym = cast(SymDesc, srcs[6]) if len(srcs) == 7 else None
  sym_args = [_v(6, sp_sym[0], sp_sym[1], "start_pos")] if sp_sym is not None else []
  sp: int|UOp|None = _v(8, sp_sym[0], sp_sym[1], "start_pos") if sp_sym is not None else start_pos
  k = _amd._gated_delta_prefill_kernel(_p(0, (batch, heads, tokens, value_dim), v_dtype), _p(1, (batch, heads, tokens, key_dim), "f32"),
                                       _p(2, (batch, heads, tokens, key_dim), "f32"), _p(3, (batch, heads, tokens, value_dim), v_dtype),
                                       _p(4, (batch, heads, tokens), beta_dtype), _p(5, alpha_shape, alpha_dtype),
                                       _p(6, (batch, heads, value_dim, key_dim), state_dtype), _p(7, (batch, heads, tokens), "f32"),
                                       start_pos=sp)
  return out.after(k.call(out, p_q, p_k, p_v, p_beta, p_alpha, p_state, kq, *sym_args))
