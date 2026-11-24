"""slopkernels: optional AMD RDNA3/RDNA4/CDNA kernel library exported strictly as tinygrad UIR text.

Narrow public API (pure primitive spec -> UIR text, zero runtime/Device/Tensor logic):
  - `list_kernels() -> tuple[str, ...]`
  - `get(name: str, /, **spec) -> str | None`
  - `compile_all(out_dir: str | pathlib.Path | None = None) -> dict[str, str]`

Logical-op contract (`dense_linear`, `quant_linear`, `moe_linear`, `flash_attention_op`,
`hc_coeffs_op`, `gated_delta_prefill_op`):
  Every logical op accepts `target`, `device`, `srcs` (a tuple of pure per-input descriptors matching
  the positional args passed to `parse_uir(uir).call_with_output(*srcs)`), plus pure semantic kwargs:
    - Tensor input descriptor: `(shape: tuple[int, ...], shard_shape: tuple[int, ...], axis: int | None, dtype: str)`
    - Symbolic scalar UOp descriptor: `(vmin: int, vmax: int)`
  See README.md for the full per-op schema table.
"""
from __future__ import annotations

import inspect
import pathlib
from collections.abc import Callable
from typing import Any

from tinygrad import UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.uop.ops import ParamArg
from tinygrad.uop.uir import render_uir

from slopkernels.src import amd as _amd
from slopkernels.src import amd_gguf as _gguf
from slopkernels.src import extra as _extra
from slopkernels.src import ops as _ops

_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
def _dt(d:str|DType) -> DType: return d if isinstance(d, DType) else _DTYPES[d]

def _p(slot:int, shape:tuple[int, ...], dt:str|DType, device:str|None=None, name:str|None=None) -> UOp:
  return UOp.param(slot, _dt(dt), shape, device, name=name)

def _v(slot:int, vmin:int, vmax:int, name:str|None=None) -> UOp:
  return UOp.variable(name or f"v{slot}", vmin, vmax).replace(arg=ParamArg(slot, dtypes.weakint, (vmin, vmax), name, AddrSpace.ALU, None))

def _sym(slot:int, spec:int|tuple[int, int], name:str) -> int|UOp:
  return spec if isinstance(spec, int) else _v(slot, spec[0], spec[1], name)

def _iq_grid_len(ggml_type:int) -> int:
  return {_gguf.IQ2_XXS: 512, _gguf.IQ2_XS: 1024, _gguf.IQ1_S: 4096,
          _gguf.IQ2_S: 2048, _gguf.IQ3_XXS: 256, _gguf.IQ3_S: 512}[ggml_type]

def _word_dt(ggml_type:int) -> DType:
  return dtypes.uint8 if ggml_type == _gguf.MXFP4 else dtypes.uint16 if ggml_type in _gguf.HALFWORD_QUANTS else dtypes.uint32

# ******** Kernel Registry (name -> (builder, canonical offline compile_all spec)) ********

_REGISTRY: dict[str, tuple[Callable[..., str|None], dict[str, Any]]] = {}

def _register(name:str, sample:dict[str, Any]):
  def deco(fn:Callable[..., str|None]) -> Callable[..., str|None]:
    _REGISTRY[name] = (fn, sample)
    return fn
  return deco

def _opt_uir(u:UOp|None) -> str|None: return None if u is None else render_uir(u)

# --- High-Level Logical Ops (return UIR graph or None when unsupported on target/shapes) ---

def _td(shape:tuple[int, ...], dtype:str, shard:tuple[int, ...]|None=None, axis:int|None=None) -> _ops.TensorDesc:
  return (shape, shard or shape, axis, dtype)

@_register("dense_linear", {"target": "gfx1201", "srcs": (_td((64, 128), "f16"), _td((1, 128), "f32"))})
def _dense_linear_op(**kw:Any) -> str|None: return _opt_uir(_ops.build_dense_linear(**kw))

@_register("quant_linear", {"target": "gfx1201",
                            "srcs": (_td((64 * 256 * 272 // 256 // 2,), "u16"), _td((1, 256), "f32")),
                            "ggml_type": 8, "in_features": 256, "out_features": 64, "groups": 1, "w_axis": None})
def _quant_linear_op(**kw:Any) -> str|None: return _opt_uir(_ops.build_quant_linear(**kw))

@_register("moe_linear", {"target": "gfx1201",
                          "srcs": (_td((8 * 64 * 256 * 210 // 256 // 2,), "u16"), _td((4,), "i32"), _td((1, 1, 1, 256), "f32")),
                          "ggml_type": 14, "sel_shape": (1, 1, 4), "in_features": 256,
                          "local_out": 64, "global_out": 64, "w_axis": None, "ndev": 1})
def _moe_linear_op(**kw:Any) -> str|None: return _opt_uir(_ops.build_moe_linear(**kw))

@_register("flash_attention_op", {"target": "gfx1201",
                                  "srcs": (_td((1, 4, 1, 64), "f16"), _td((2, 1, 2, 128, 64), "f16"), (1, 128)),
                                  "decode": True})
def _flash_attention_op(**kw:Any) -> str|None: return _opt_uir(_ops.build_flash_attention(**kw))

@_register("hc_coeffs_op", {"target": "gfx1201",
                            "srcs": (_td((2, 24), "f32"), _td((2, 128), "f32"), _td((24,), "f16"), _td((3,), "f16")),
                            "count": 4, "iters": 2, "eps": 1e-6, "norm_eps": 1e-6})
def _hc_coeffs_op(**kw:Any) -> str|None: return _opt_uir(_ops.build_hc_coeffs(**kw))

@_register("gated_delta_prefill_op", {"target": "gfx1201",
                                      "srcs": (_td((1, 4, 8, 64), "f32"), _td((1, 4, 8, 64), "f32"), _td((1, 4, 8, 64), "f32"),
                                               _td((1, 4, 8), "f32"), _td((1, 4, 8), "f32"), _td((1, 4, 64, 64), "f32"))})
def _gated_delta_prefill_op(**kw:Any) -> str|None: return _opt_uir(_ops.build_gated_delta_prefill(**kw))

# --- Dense / Quantized Linear & MoE ---

@_register("f16_gemv", {"in_features": 128, "out_features": 64, "tokens": 1})
def _f16_gemv(in_features:int, out_features:int, tokens:int=1, weight_dtype:str="f16",
              in_dtype:str="f32", out_dtype:str="f32", has_bias:bool=False, bias_dtype:str|None=None) -> str:
  ps = [_p(0, (tokens, out_features), out_dtype), _p(1, (out_features, in_features), weight_dtype), _p(2, (tokens, in_features), in_dtype)]
  if has_bias: ps.append(_p(3, (out_features,), bias_dtype or weight_dtype))
  return render_uir(_amd._amd_f16_gemv_kernel(*ps, in_features=in_features, out_features=out_features, tokens=tokens))

@_register("f16_wmma", {"out_features": 64, "in_features": 128, "tokens": 16})
def _f16_wmma(out_features:int, in_features:int, tokens:int, weight_dtype:str="f16", out_dtype:str="f32",
              kgroups:int=2, lanes:int=32, valid_tokens_max:int|None=None) -> str:
  ps = [_p(0, (tokens, out_features), out_dtype), _p(1, (out_features, in_features), weight_dtype), _p(2, (tokens, in_features), "f16")]
  vt = _v(3, 1, valid_tokens_max, "valid_tokens") if valid_tokens_max is not None else None
  return render_uir(_gguf._amd_f16_wmma_kernel(*ps, out_features=out_features, in_features=in_features,
                                               kgroups=kgroups, lanes=lanes, valid_tokens=vt))

@_register("q8_quantize", {"tokens": 1, "in_features": 256})
def _q8_quantize(tokens:int, in_features:int, in_shape:tuple[int, ...]|None=None, in_dtype:str="f32") -> str:
  g = in_features // 32
  return render_uir(_gguf._q8_quantize_kernel(_p(0, (tokens, g, 8), "u32"), _p(1, (tokens, g), "f32"),
                                              _p(2, (tokens, g, 2), "f32"), _p(3, in_shape or (tokens, in_features), in_dtype),
                                              tokens=tokens, in_features=in_features))

@_register("q8k_quantize", {"tokens": 1, "in_features": 256})
def _q8k_quantize(tokens:int, in_features:int, in_dtype:str="f32") -> str:
  return render_uir(_extra._q8k_quantize_kernel(_p(0, (tokens, in_features // 32, 8), "u32"), _p(1, (tokens, in_features // 256), "f32"),
                                                _p(2, (tokens, in_features), in_dtype), tokens=tokens, in_features=in_features))

@_register("quant_decode", {"out_features": 64, "in_features": 256, "ggml_type": 8})
def _quant_decode(out_features:int, in_features:int, ggml_type:int, tokens:int=1, cdna:bool=False, chunks:int=1,
                  group_rows:int=0, raw_len:int|None=None, out_shape:tuple[int, ...]|None=None) -> str:
  w_dt = _word_dt(ggml_type)
  nwords = raw_len if raw_len is not None else out_features * in_features * _gguf.QUANT_SIZES[ggml_type] // 256 // w_dt.itemsize
  g = in_features // 32
  o_shp = out_shape if out_shape is not None else ((tokens, out_features) if chunks == 1 else (tokens, out_features, chunks))
  ps = [_p(0, o_shp, "f32"), _p(1, (nwords,), w_dt), _p(2, (tokens, g, 8), "u32"), _p(3, (tokens, g), "f32"), _p(4, (tokens, g, 2), "f32")]
  if ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ3_XXS, _gguf.IQ1_S, _gguf.IQ3_S, _gguf.IQ2_S):
    ps.append(_p(5, (_iq_grid_len(ggml_type),), "u32"))
  return render_uir(_gguf._quant_decode_kernel(*ps, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                                               cdna=cdna, group_rows=group_rows))

@_register("quant_linear_wmma", {"out_features": 64, "in_features": 256, "ggml_type": 8, "tokens": 16})
def _quant_linear_wmma(out_features:int, in_features:int, ggml_type:int, tokens:int, *, kgroups:int=2, lanes:int=32,
                       raw_len:int|None=None, out_shape:tuple[int, ...]|None=None, out_dtype:str="f32",
                       x_shape:tuple[int, ...]|None=None, rdna4:bool=False, valid_tokens:int|tuple[int, int]|None=None) -> str:
  w_dt = _word_dt(ggml_type)
  nwords = raw_len if raw_len is not None else out_features * in_features * _gguf.QUANT_SIZES[ggml_type] // 256 // w_dt.itemsize
  o_shp = out_shape if out_shape is not None else (tokens, out_features)
  ps = [_p(0, o_shp, out_dtype), _p(1, (nwords,), w_dt), _p(2, x_shape or (tokens, in_features), "f16")]
  if ggml_type == _gguf.IQ4_XS: ps.append(_p(len(ps), (256,), "u32"))
  elif ggml_type in (_gguf.IQ3_XXS, _gguf.IQ2_S): ps.append(_p(len(ps), (512 if ggml_type == _gguf.IQ3_XXS else 4096,), "u32"))
  elif ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ1_S, _gguf.IQ3_S): ps.append(_p(len(ps), (_iq_grid_len(ggml_type),), "u32"))
  vt = _sym(len(ps), valid_tokens, "valid_tokens") if valid_tokens is not None else None
  if ggml_type == _gguf.IQ4_XS:
    return render_uir(_gguf._iq4_linear_f16_wmma_kernel(*ps, out_features=out_features, in_features=in_features,
                                                        kgroups=kgroups, lanes=lanes, valid_tokens=vt))
  if ggml_type in (_gguf.Q4_K, _gguf.Q5_K):
    return render_uir(_gguf._q5_linear_f16_wmma_kernel(*ps, out_features=out_features, in_features=in_features,
                                                       ggml_type=ggml_type, kgroups=kgroups, lanes=lanes, valid_tokens=vt))
  return render_uir(_gguf._quant_linear_f16_wmma_kernel(*ps, out_features=out_features, in_features=in_features,
                                                        ggml_type=ggml_type, kgroups=kgroups, lanes=lanes, valid_tokens=vt))

@_register("quant_decode_multi_proj", {"out_features_tuple": (64, 64), "in_features": 256, "ggml_type": 8})
def _quant_decode_multi_proj(out_features_tuple:tuple[int, ...], in_features:int, ggml_type:int, tokens:int=1, cdna:bool=False) -> str:
  w_dt = _word_dt(ggml_type)
  g = in_features // 32
  ps = [_p(i, (tokens, o), "f32") for i, o in enumerate(out_features_tuple)]
  for o in out_features_tuple:
    nwords = o * in_features * _gguf.QUANT_SIZES[ggml_type] // 256 // w_dt.itemsize
    ps.append(_p(len(ps), (nwords,), w_dt))
  ps.extend([_p(len(ps), (tokens, g, 8), "u32"), _p(len(ps) + 1, (tokens, g), "f32"), _p(len(ps) + 2, (tokens, g, 2), "f32")])
  if ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ3_XXS, _gguf.IQ1_S, _gguf.IQ3_S, _gguf.IQ2_S):
    ps.append(_p(len(ps), (_iq_grid_len(ggml_type),), "u32"))
  return render_uir(_gguf._quant_decode_multi_proj_kernel(*ps, out_features_tuple=out_features_tuple, in_features=in_features,
                                                          ggml_type=ggml_type, tokens=tokens, cdna=cdna))

@_register("moe_decode", {"out_shape": (4, 128, 1), "raw_len": 32 * 128 * 256 * 144 // 256 // 4,
                          "x_rows": 1, "sel_len": 4, "out_features": 128, "in_features": 256, "ggml_type": 40, "x_share": 4})
def _moe_decode(out_shape:tuple[int, ...], raw_len:int, x_rows:int, sel_len:int, out_features:int, in_features:int, ggml_type:int,
                x_share:int, *, cdna:bool=False, ndev:int=1, has_scale:bool=False, num_experts:int=0) -> str:
  w_dt = _word_dt(ggml_type)
  g = in_features // 32
  ps = [_p(0, out_shape, "f32"), _p(1, (raw_len,), w_dt), _p(2, (x_rows, g, 8), "u32"), _p(3, (x_rows, g), "f32"),
        _p(4, (x_rows, g, 2), "f32"), _p(5, (sel_len,), "i32")]
  if has_scale: ps.append(_p(len(ps), (num_experts,), "f16"))
  if ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ3_XXS, _gguf.IQ1_S, _gguf.IQ3_S, _gguf.IQ2_S):
    ps.append(_p(len(ps), (_iq_grid_len(ggml_type),), "u32"))
  return render_uir(_gguf._moe_decode_kernel(*ps, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                                             x_share=x_share, cdna=cdna, ndev=ndev, has_scale=has_scale))

@_register("moe_decode_situ_q8", {"num_experts": 32, "out_features": 256, "in_features": 256, "ggml_type": 8,
                                  "slots": 4, "situ": (4.0, 4.0), "x_share": 4})
def _moe_decode_situ_q8(num_experts:int, out_features:int, in_features:int, ggml_type:int, slots:int, situ:tuple[float, float],
                        *, x_share:int=8, cdna:bool=False, has_scale:bool=False) -> str:
  w_dt = _word_dt(ggml_type)
  nwords = num_experts * out_features * in_features * _gguf.QUANT_SIZES[ggml_type] // 256 // w_dt.itemsize
  h, g_in, g_out = out_features // 2, in_features // 32, (out_features // 2) // 32
  x_rows = slots // x_share
  ps = [_p(0, (slots, h), "f32"), _p(1, (slots, g_out, 8), "u32"), _p(2, (slots, g_out), "f32"), _p(3, (slots, g_out, 2), "f32"),
        _p(4, (nwords,), w_dt), _p(5, (x_rows, g_in, 8), "u32"), _p(6, (x_rows, g_in), "f32"), _p(7, (x_rows, g_in, 2), "f32"),
        _p(8, (slots,), "i32")]
  if has_scale: ps.append(_p(len(ps), (num_experts,), "f16"))
  if ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ3_XXS, _gguf.IQ1_S, _gguf.IQ3_S, _gguf.IQ2_S):
    ps.append(_p(len(ps), (_iq_grid_len(ggml_type),), "u32"))
  return render_uir(_gguf._moe_decode_situ_q8_kernel(*ps, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                                                     x_share=x_share, cdna=cdna, has_scale=has_scale, situ=situ))

@_register("moe_decode_down_sum", {"num_experts": 32, "out_features": 256, "in_features": 128, "ggml_type": 40, "slots": 4})
def _moe_decode_down_sum(num_experts:int, out_features:int, in_features:int, ggml_type:int, slots:int,
                         *, cdna:bool=False, has_scale:bool=False, has_add:bool=False) -> str:
  w_dt = _word_dt(ggml_type)
  nwords = num_experts * out_features * in_features * _gguf.QUANT_SIZES[ggml_type] // 256 // w_dt.itemsize
  g_in = in_features // 32
  ps = [_p(0, (1, out_features), "f32"), _p(1, (nwords,), w_dt), _p(2, (slots, g_in, 8), "u32"), _p(3, (slots, g_in), "f32"),
        _p(4, (slots, g_in, 2), "f32"), _p(5, (slots,), "i32"), _p(6, (slots,), "f32")]
  if has_scale: ps.append(_p(len(ps), (num_experts,), "f16"))
  if has_add: ps.append(_p(len(ps), (out_features,), "f32"))
  if ggml_type in (_gguf.IQ2_XXS, _gguf.IQ2_XS, _gguf.IQ3_XXS, _gguf.IQ1_S, _gguf.IQ3_S, _gguf.IQ2_S):
    ps.append(_p(len(ps), (_iq_grid_len(ggml_type),), "u32"))
  return render_uir(_gguf._moe_decode_down_sum_kernel(*ps, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                                                      slots=slots, cdna=cdna, has_scale=has_scale, has_add=has_add))

# --- Attention & Recurrent ---

@_register("flash_attention", {"bh": 4, "m": 32, "d": 64, "b": 1, "h_kv": 2, "physical_n": 128,
                               "valid_kv_len": (1, 128), "rel_shape": (32, 16), "has_sink": True, "has_gate": True})
def _flash_attention(bh:int, m:int, d:int, b:int, h_kv:int, physical_n:int, *, valid_kv_len:int|tuple[int, int],
                     q_start:int|tuple[int, int]|None=None, valid_m:int|tuple[int, int]|None=None,
                     is_q4:bool=False, rdna4:bool=True, window:int=0, rel_shape:tuple[int, int]|None=None,
                     has_sink:bool=False, has_gate:bool=False, ring:bool=False, out_dtype:str="f32") -> str:
  h = bh // b
  o_shp = (b, m, h * d) if has_gate else (bh, m, d)
  c_shp = (2, b, h_kv, physical_n, 17 if is_q4 else d)
  ps: list[UOp] = [_p(0, o_shp, out_dtype), _p(1, (bh, m, d), "f16"), _p(2, c_shp, "u32" if is_q4 else "f16")]
  if rel_shape is not None:
    E, L = rel_shape
    ps.extend([_p(len(ps), (bh, m, L), "f16"), _p(len(ps) + 1, (E, L), "f16")])
  if has_sink: ps.append(_p(len(ps), (bh, m), "f32"))
  if has_gate: ps.append(_p(len(ps), (b, m, h * d), "f16"))
  slot = len(ps)
  def next_sym(spec:int|tuple[int, int]|None, name:str) -> int|UOp|None:
    nonlocal slot
    if spec is None or isinstance(spec, int): return spec
    u = _v(slot, spec[0], spec[1], name)
    slot += 1
    return u
  vkl = next_sym(valid_kv_len, "valid_kv_len")
  assert vkl is not None
  qs = next_sym(q_start, "q_start")
  vm = next_sym(valid_m, "valid_m")
  return render_uir(_amd._amd_flash_attention(*ps, valid_kv_len=vkl, q_start=qs, valid_m=vm, rdna4=rdna4, window=window,
                                              has_rel=rel_shape is not None, has_sink=has_sink, has_gate=has_gate, ring=ring))

@_register("flash_attention_decode_partial", {"b": 1, "h": 4, "m": 1, "d": 64, "h_kv": 2, "physical_n": 128,
                                              "chunks": 2, "max_kv_len": 128, "valid_kv_len": (1, 128), "rel_shape": (32, 16)})
def _flash_attention_decode_partial(b:int, h:int, m:int, d:int, h_kv:int, physical_n:int, chunks:int, max_kv_len:int, *,
                                    valid_kv_len:int|tuple[int, int], block_n:int=64, waves:int=16, window:int=0,
                                    is_q4:bool=False, rel_shape:tuple[int, int]|None=None, ring:bool=False) -> str:
  ps: list[UOp] = [_p(0, (b * m, h, chunks, d), "f32"), _p(1, (b * m, h, chunks, 2), "f32"),
                   _p(2, (b, h, m, d), "f16"), _p(3, (2, b, h_kv, physical_n, 17 if is_q4 else d), "u32" if is_q4 else "f16")]
  if rel_shape is not None:
    E, L = rel_shape
    ps.extend([_p(4, (b, h, m, L), "f16"), _p(5, (E, L), "f16")])
  vkl = _sym(len(ps), valid_kv_len, "valid_kv_len")
  return render_uir(_amd._amd_flash_attention_decode_partial(*ps, valid_kv_len=vkl, max_kv_len=max_kv_len, block_n=block_n,
                                                             waves=waves, window=window, has_rel=rel_shape is not None, ring=ring))

@_register("flash_decode_combine", {"b": 1, "h": 4, "m": 1, "d": 64, "chunks": 2, "valid_kv_len": (1, 128), "has_sink": True, "has_gate": True})
def _flash_decode_combine(b:int, h:int, m:int, d:int, chunks:int, *, valid_kv_len:int|tuple[int, int], window:int=0,
                          has_sink:bool=False, has_gate:bool=False, out_dtype:str="f32") -> str:
  o_shp = (b, m, h * d) if has_gate else (b, h, m, d)
  ps: list[UOp] = [_p(0, o_shp, out_dtype), _p(1, (b * m, h, chunks, d), "f32"), _p(2, (b * m, h, chunks, 2), "f32")]
  if has_sink: ps.append(_p(len(ps), (b, h, m), "f32"))
  if has_gate: ps.append(_p(len(ps), (b, m, h * d), "f32"))
  vkl = _sym(len(ps), valid_kv_len, "valid_kv_len")
  return render_uir(_amd._amd_flash_decode_combine(*ps, valid_kv_len=vkl, window=window, has_sink=has_sink, has_gate=has_gate))

@_register("flash_decode_combine_q8", {"b": 1, "h": 4, "m": 1, "d": 64, "chunks": 2, "valid_kv_len": (1, 128), "has_sink": True})
def _flash_decode_combine_q8(b:int, h:int, m:int, d:int, chunks:int, *, valid_kv_len:int|tuple[int, int],
                             window:int=0, has_sink:bool=False) -> str:
  bt, groups = b * m, (h * d) // 32
  ps: list[UOp] = [_p(0, (b, m, h * d), "f32"), _p(1, (bt, groups, 8), "u32"), _p(2, (bt, groups), "f32"), _p(3, (bt, groups, 2), "f32"),
                   _p(4, (bt, h, chunks, d), "f32"), _p(5, (bt, h, chunks, 2), "f32")]
  if has_sink: ps.append(_p(len(ps), (b, h, m), "f32"))
  ps.append(_p(len(ps), (b, m, h * d), "f32"))
  vkl = _sym(len(ps), valid_kv_len, "valid_kv_len")
  return render_uir(_amd._amd_flash_decode_combine_q8(*ps, valid_kv_len=vkl, window=window, has_sink=has_sink))

@_register("gated_delta_prefill", {"batch": 1, "heads": 4, "tokens": 8, "value_dim": 64, "key_dim": 64, "alpha_shape": (1, 4, 8)})
def _gated_delta_prefill(batch:int, heads:int, tokens:int, value_dim:int, key_dim:int, *,
                         v_dtype:str="f32", beta_dtype:str="f32", alpha_shape:tuple[int, ...], alpha_dtype:str="f32",
                         state_dtype:str="f32", kq_dtype:str="f32", start_pos:int|tuple[int, int]|None=None) -> str:
  ps = [_p(0, (batch, heads, tokens, value_dim), v_dtype), _p(1, (batch, heads, tokens, key_dim), "f32"),
        _p(2, (batch, heads, tokens, key_dim), "f32"), _p(3, (batch, heads, tokens, value_dim), v_dtype),
        _p(4, (batch, heads, tokens), beta_dtype), _p(5, alpha_shape, alpha_dtype),
        _p(6, (batch, heads, value_dim, key_dim), state_dtype), _p(7, (batch, heads, tokens), kq_dtype)]
  sp = _sym(8, start_pos, "start_pos") if start_pos is not None else None
  return render_uir(_amd._gated_delta_prefill_kernel(*ps, start_pos=sp))

# --- Residual, Conv, Norm, Routing, Activation ---

@_register("attn_res_mix", {"b": 1, "t": 1, "n_streams": 2, "d": 256, "q": 1, "eps": 1e-6, "scale": 1 / 16, "normed": True})
def _attn_res_mix(b:int, t:int, n_streams:int, d:int, q:int, eps:float, scale:float, normed:bool,
                  out_dtype:str="f32", valid_t_max:int|None=None) -> str:
  vt = _v(3, 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_attn_res_mix_kernel(_p(0, (b, t, q, d), out_dtype), _p(1, (b, t, n_streams, d), "f32"), _p(2, (q, d), "f16"),
                                                  valid_t=vt, N=n_streams, Q=q, D=d, eps=eps, scale=scale, normed=normed))

@_register("attn_res_mix_streams", {"b": 1, "t": 1, "d": 256, "q": 1, "n_streams": 2, "eps": 1e-6, "scale": 1 / 16,
                                    "normed": True, "stream_dtypes": ("f16", "f32"), "pre_normed": 1})
def _attn_res_mix_streams(b:int, t:int, d:int, q:int, n_streams:int, eps:float, scale:float, normed:bool,
                          stream_dtypes:tuple[str, ...], pre_normed:int=0, emit_normed:bool=False,
                          out_dtype:str="f32", valid_t_max:int|None=None) -> str:
  ps: list[UOp] = [_p(0, (b, t, q, d), out_dtype)]
  if emit_normed: ps.append(_p(1, (b, t, d), "f16"))
  ps.append(_p(len(ps), (q, d), "f16"))
  for dt in stream_dtypes: ps.append(_p(len(ps), (b, t, d), dt))
  vt = _v(len(ps), 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_attn_res_mix_streams_kernel(*ps, valid_t=vt, N=n_streams, Q=q, D=d, eps=eps, scale=scale,
                                                          normed=normed, pre_normed=pre_normed, emit_normed=emit_normed))

@_register("attn_res_update", {"b": 1, "t": 1, "n_streams": 2, "d": 256, "new_block": True})
def _attn_res_update(b:int, t:int, n_streams:int, d:int, new_block:bool, valid_t_max:int|None=None) -> str:
  out_n = n_streams + 1 if new_block else n_streams
  vt = _v(3, 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_attn_res_update_kernel(_p(0, (b, t, out_n, d), "f32"), _p(1, (b, t, n_streams, d), "f32"),
                                                     _p(2, (b, t, d), "f32"), valid_t=vt, N=n_streams, D=d, new_block=new_block))

@_register("rel_short_conv", {"b": 1, "t": 1, "c": 128, "k": 4, "n_kv_heads": 2, "inv_scale": 0.5, "start_pos_max": 127, "has_res": True})
def _rel_short_conv(b:int, t:int, c:int, k:int, n_kv_heads:int, inv_scale:float, *, start_pos_max:int,
                    has_dyn:bool=False, has_res:bool=False, has_hist:bool=False, hist_len:int=0,
                    in_dtype:str="f32", out_dtype:str="f32") -> str:
  ps: list[UOp] = [_p(0, (b, t, c), out_dtype), _p(1, (b, k - 1, c), "f32"), _p(2, (b, t, c), in_dtype), _p(3, (c, k), "f16")]
  if has_dyn: ps.append(_p(len(ps), (b, t, n_kv_heads * k), "f32"))
  if has_res: ps.append(_p(len(ps), (b, t, c), "f32"))
  if has_hist: ps.append(_p(len(ps), (b, hist_len, c), "f32"))
  sp = _v(len(ps), 0, start_pos_max, "start_pos")
  return render_uir(_amd._amd_rel_short_conv_kernel(*ps, valid_t=t, K=k, C=c, H=n_kv_heads, inv_scale=inv_scale, start_pos=sp,
                                                    has_dyn=has_dyn, has_res=has_res, has_hist=has_hist))

@_register("loop_ctrl_down_decode", {"bt": 1, "d": 256, "r": 32})
def _loop_ctrl_down_decode(bt:int, d:int, r:int, *, h_q:int=1, h_idx:int=0, m_q:int=4, m_idx:int=0,
                           w_dtype:str="f16", h_dtype:str="f32", m_dtype:str="f32") -> str:
  ps = [_p(0, (bt, r), "f32"), _p(1, (bt, h_q, d), h_dtype), _p(2, (bt, m_q, d), m_dtype), _p(3, (r, 2 * d), w_dtype)]
  return render_uir(_amd._amd_loop_ctrl_down_decode_kernel(*ps, BT=bt, D=d, R=r, h_q=h_q, h_idx=h_idx, m_q=m_q, m_idx=m_idx))

@_register("loop_ctrl_apply_decode", {"bt": 1, "d": 256, "r": 32})
def _loop_ctrl_apply_decode(bt:int, d:int, r:int, *, h_q:int=1, h_idx:int=0, m_q:int=4, m_idx:int=0,
                            w_dtype:str="f16", h_dtype:str="f32", m_dtype:str="f32") -> str:
  ps = [_p(0, (bt, d), "f32"), _p(1, (bt, h_q, d), h_dtype), _p(2, (bt, m_q, d), m_dtype),
        _p(3, (bt, r), "f32"), _p(4, (d, r), w_dtype), _p(5, (d, r), w_dtype)]
  return render_uir(_amd._amd_loop_ctrl_apply_decode_kernel(*ps, BT=bt, D=d, R=r, h_q=h_q, h_idx=h_idx, m_q=m_q, m_idx=m_idx))

@_register("moe_router_topk", {"b": 1, "t": 1, "e": 32, "k": 4})
def _moe_router_topk(b:int, t:int, e:int, k:int, *, has_bias:bool=True, normalize:bool=True, scale:float=1.0,
                     valid_t_max:int|None=None) -> str:
  ps: list[UOp] = [_p(0, (b * t, k), "i32"), _p(1, (b * t, k), "f32"), _p(2, (b, t, e), "f32")]
  if has_bias: ps.append(_p(3, (e,), "f16"))
  vt = _v(len(ps), 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_moe_router_topk_kernel(*ps, valid_t=vt, E=e, K=k, has_bias=has_bias, normalize=normalize, scale=scale))

@_register("rmsnorm", {"b": 1, "t": 1, "d": 256, "eps": 1e-6, "q8": True})
def _rmsnorm(b:int, t:int, d:int, eps:float, *, in_dtype:str="f32", out_dtype:str="f32", q_cnt:int=1, q_idx:int=0,
             valid_t_max:int|None=None, q8:bool=False) -> str:
  bt = b * t
  in_shp = (b, t, q_cnt, d) if q_cnt > 1 else (b, t, d)
  if q8:
    g = d // 32
    ps = [_p(0, (b, t, d), out_dtype), _p(1, (bt, g, 8), "u32"), _p(2, (bt, g), "f32"), _p(3, (bt, g, 2), "f32"),
          _p(4, in_shp, in_dtype), _p(5, (d,), "f16")]
    return render_uir(_amd._amd_rmsnorm_q8_kernel(*ps, BT=bt, D=d, eps=eps, q_cnt=q_cnt, q_idx=q_idx))
  ps = [_p(0, (b, t, d), out_dtype), _p(1, in_shp, in_dtype), _p(2, (d,), "f16")]
  vt = _v(len(ps), 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_rmsnorm_kernel(*ps, valid_t=vt, D=d, eps=eps, q_cnt=q_cnt, q_idx=q_idx))

@_register("rel_qr_prep", {"b": 1, "t": 1, "h": 4, "hd": 64, "l": 16, "qk_clip": 8.0, "rel_clip": 8.0,
                           "full": True, "floor": 16.0, "alpha": 0.1, "start_pos_max": 127})
def _rel_qr_prep(b:int, t:int, h:int, hd:int, l:int, qk_clip:float, rel_clip:float, *, full:bool, floor:float, alpha:float,
                 start_pos_max:int, valid_t_max:int|None=None, in_dtype:str="f32") -> str:
  ps = [_p(0, (b, h, t, hd), "f16"), _p(1, (b, h, t, l), "f16"), _p(2, (b, t, h, hd), in_dtype), _p(3, (h, hd), "f16"),
        _p(4, (b, t, h, l), in_dtype), _p(5, (h, l), "f16")]
  sp = _v(len(ps), 0, start_pos_max, "start_pos")
  vt = _v(len(ps) + 1, 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_rel_qr_prep_kernel(*ps, valid_t=vt, start_pos=sp, H=h, D=hd, L=l, qk_clip=qk_clip, rel_clip=rel_clip,
                                                 full=full, floor=floor, alpha=alpha))

@_register("rel_kv_cache_update", {"b": 1, "t": 1, "h_kv": 2, "hd": 128, "max_ctx": 128, "qk_clip": 8.0, "eps": 1e-6,
                                   "start_pos_max": 127, "q4": True})
def _rel_kv_cache_update(b:int, t:int, h_kv:int, hd:int, max_ctx:int, qk_clip:float, eps:float, *, start_pos_max:int,
                         valid_t_max:int|None=None, q4:bool=False, ring:bool=False, in_dtype:str="f32") -> str:
  c_shp = (2, b, h_kv, max_ctx, 17 if q4 else hd)
  ps = [_p(0, c_shp, "u32" if q4 else "f16"), _p(1, (b, t, h_kv, hd), in_dtype), _p(2, (b, t, h_kv, hd), in_dtype)]
  sp = _v(len(ps), 0, start_pos_max, "start_pos")
  vt = _v(len(ps) + 1, 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  fn = _amd._amd_rel_kv_cache_q4_kernel if q4 else _amd._amd_rel_kv_cache_kernel
  return render_uir(fn(*ps, valid_t=vt, start_pos=sp, H_KV=h_kv, D=hd, qk_clip=qk_clip, eps=eps, ring=ring))

@_register("moe_sum_slots", {"b": 1, "t": 1, "slots": 4, "d": 256})
def _moe_sum_slots(b:int, t:int, slots:int, d:int, *, has_add:bool=False, in_dtype:str="f32", out_dtype:str="f32",
                   valid_t_max:int|None=None) -> str:
  ps = [_p(0, (b, t, d), out_dtype), _p(1, (b * t * slots, d), in_dtype)]
  if has_add: ps.append(_p(2, (b, t, d), in_dtype))
  vt = _v(len(ps), 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_moe_sum_slots_kernel(*ps, valid_t=vt, slots=slots, D=d, has_add=has_add))

@_register("situ", {"b": 1, "t": 1, "d": 256, "b0": 4.0, "b1": 4.0})
def _situ(b:int, t:int, d:int, b0:float, b1:float, in_dtype:str="f32", out_dtype:str="f32", valid_t_max:int|None=None) -> str:
  vt = _v(3, 1, valid_t_max, "valid_t") if valid_t_max is not None else t
  return render_uir(_amd._amd_situ_kernel(_p(0, (b, t, d), out_dtype), _p(1, (b, t, d), in_dtype), _p(2, (b, t, d), in_dtype),
                                          valid_t=vt, D=d, b0=b0, b1=b1))

# --- Extra AMD Kernels (PR #19170: FWHT, Q8/IQ4_XS embed, Q8 pack/unpack/sum, gather/scatter, argmax, nucleus, router, mHC) ---

@_register("fwht", {"n": 4, "d": 128, "pre": True})
def _fwht(n:int, d:int, *, pre:bool=True, in_dtype:str="f32", out_dtype:str="f32") -> str:
  return render_uir(_extra._fwht_kernel(_p(0, (n, d), out_dtype), _p(1, (n, d), in_dtype), _p(2, (d,), "f32"), pre=pre))

@_register("rows_gather", {"n": 4, "p": 64, "f": 256})
def _rows_gather(n:int, p:int, f:int, *, in_dtype:str="f16", out_dtype:str="f32") -> str:
  return render_uir(_extra._rows_gather_kernel(_p(0, (n, f), out_dtype), _p(1, (p, f), in_dtype), _p(2, (n,), "i32")))

@_register("rows_scatter", {"n": 4, "p": 64, "f": 256})
def _rows_scatter(n:int, p:int, f:int, *, in_dtype:str="f32", out_dtype:str="f16") -> str:
  return render_uir(_extra._rows_scatter_kernel(_p(0, (p, f), out_dtype), _p(1, (n, f), in_dtype), _p(2, (n,), "i32")))

@_register("q8_embed", {"r": 4, "vocab": 64, "d": 256})
def _q8_embed(r:int, vocab:int, d:int, *, out_dtype:str="f32") -> str:
  return render_uir(_extra._q8_embed_kernel(_p(0, (r, d), out_dtype), _p(1, (vocab * (d // 32) * 34,), "u8"), _p(2, (r,), "i32")))

@_register("iq4xs_embed", {"r": 4, "vocab": 64, "d": 256})
def _iq4xs_embed(r:int, vocab:int, d:int, *, out_dtype:str="f32") -> str:
  return render_uir(_extra._iq4xs_embed_kernel(_p(0, (r, d), out_dtype), _p(1, (vocab * (d // 256) * 136,), "u8"), _p(2, (r,), "i32")))

@_register("q8_pack", {"t": 2, "d": 256})
def _q8_pack(t:int, d:int, *, in_dtype:str="f32", has_add:bool=False, row0:int=0, in_rows:int|None=None) -> str:
  g = d // 32
  ps = [_p(0, (t, g, 9), "u32"), _p(1, (in_rows or t, d), in_dtype)]
  if has_add: ps.append(_p(2, (t, g, 9), "u32"))
  return render_uir(_extra._q8_pack_kernel(*ps, row0=row0))

@_register("q8_unpack", {"t": 2, "d": 256})
def _q8_unpack(t:int, d:int, *, n_parts:int=1, out_dtype:str="f16") -> str:
  g, rows = d // 32, t // n_parts
  ps = [_p(0, (t, d), out_dtype)] + [_p(1 + i, (rows, g, 9), "u32") for i in range(n_parts)]
  return render_uir(_extra._q8_unpack_kernel(*ps))

@_register("q8_sum", {"t": 2, "d": 256, "n_packed": 1})
def _q8_sum(t:int, d:int, *, n_packed:int=1, in_dtype:str="f32", out_dtype:str="f32", row0:int=0, in_rows:int|None=None) -> str:
  g = d // 32
  ps = [_p(0, (t, d), out_dtype), _p(1, (in_rows or t, d), in_dtype)] + [_p(2 + i, (t, g, 9), "u32") for i in range(n_packed)]
  return render_uir(_extra._q8_sum_kernel(*ps, row0=row0))

@_register("local_argmax", {"b": 2, "v": 1024})
def _local_argmax(b:int, v:int, *, in_dtype:str="f32") -> str:
  return render_uir(_extra._local_argmax_kernel(_p(0, (b, 2), "f32"), _p(1, (b, v), in_dtype), _p(2, (1,), "i32")))

@_register("argmax_chunks", {"b": 2, "v": 1024, "chunks": 4})
def _argmax_chunks(b:int, v:int, chunks:int, *, in_dtype:str="f32") -> str:
  return render_uir(_extra._argmax_chunks_kernel(_p(0, (b, chunks, 2), "f32"), _p(1, (b, v), in_dtype), _p(2, (1,), "i32")))

@_register("argmax_final", {"b": 2, "chunks": 4})
def _argmax_final(b:int, chunks:int) -> str:
  return render_uir(_extra._argmax_final_kernel(_p(0, (b, 2), "f32"), _p(1, (b, chunks, 2), "f32")))

@_register("nucleus_round", {"b": 1, "c_blocks": 4, "c": 256, "g": 16})
def _nucleus_round(b:int, c_blocks:int, c:int, g:int) -> str:
  return render_uir(_extra._nucleus_round_kernel(_p(0, (1, b, g), "f32"), _p(1, (b, 1, c_blocks, c), "f32"),
                                                 _p(2, (b, 1, c_blocks, c), "f32"), _p(3, (b, g), "f32")))

@_register("moe_softmax_router", {"tokens": 2, "e": 64, "k": 4})
def _moe_softmax_router(tokens:int, e:int, k:int, *, scale:float=1.0, table_width:int=0, in_dtype:str="f32") -> str:
  w = table_width or 1
  ps = [_p(0, (tokens, k * w), "i32"), _p(1, (tokens, k * w), "f32"), _p(2, (tokens, e), in_dtype)]
  if table_width: ps.append(_p(3, (e, table_width), "i32"))
  return render_uir(_extra._moe_softmax_router_kernel(*ps, k=k, scale=scale))

@_register("hc_coeffs", {"tokens": 2, "groups": 32, "features": 1024})
def _hc_coeffs(tokens:int, groups:int, features:int, *, count:int=4, iters:int=2, eps:float=1e-6, norm_eps:float=1e-6,
               base_dtype:str="f16", scale_dtype:str="f16") -> str:
  n = count * (2 + count)
  return render_uir(_extra._hc_coeffs_kernel(_p(0, (tokens, n), "f32"), _p(1, (tokens, n), "f32"), _p(2, (tokens, groups), "f32"),
                                             _p(3, (n,), base_dtype), _p(4, (3,), scale_dtype),
                                             count=count, iters=iters, eps=eps, norm_eps=norm_eps, features=features))

# ******** Narrow Public API ********

def list_kernels() -> tuple[str, ...]:
  """Return sorted names of all registered slopkernels."""
  return tuple(sorted(_REGISTRY))

def get(name:str, /, **spec:Any) -> str|None:
  """Compile a named slopkernel specification into pure tinygrad UIR text, or None if unsupported."""
  if name not in _REGISTRY: raise KeyError(f"unknown slopkernel {name!r}; available: {list_kernels()}")
  return _REGISTRY[name][0](**spec)

def compile_all(out_dir:str|pathlib.Path|None=None, target:str|tuple[int, ...]|None=None) -> dict[str, str]:
  """Compile every registered slopkernel with its canonical spec into pure UIR text (optionally writing `<name>.uir` files)."""
  out: dict[str, str] = {}
  dst = pathlib.Path(out_dir) if out_dir is not None else None
  if dst is not None: dst.mkdir(parents=True, exist_ok=True)
  arch = _ops.parse_amd_target(target) if target is not None else None
  for name in list_kernels():
    fn, sample = _REGISTRY[name]
    spec = dict(sample)
    if arch is not None:
      rdna4, cdna = arch
      if "target" in spec: spec["target"] = target
      params = inspect.signature(fn).parameters
      if "cdna" in params: spec["cdna"] = cdna
      if "rdna4" in params: spec["rdna4"] = rdna4
      if "kgroups" in params: spec["kgroups"] = 4 if cdna else 2 if rdna4 else 1
      if "lanes" in params: spec["lanes"] = 64 if cdna else 32
    uir = fn(**spec)
    assert uir is not None, f"canonical sample for {name!r} returned None"
    out[name] = uir
    if dst is not None: (dst / f"{name}.uir").write_text(uir + "\n")
  return out

__all__ = ["compile_all", "get", "list_kernels"]
