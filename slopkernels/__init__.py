"""slopkernels: optional AMD RDNA3/RDNA4/CDNA templated UIR kernels (always round-tripped through slopkernels/uir/)."""
from __future__ import annotations

import functools, os, pathlib, threading
from collections.abc import Callable
from dataclasses import replace

from tinygrad import Context, UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.uop.ops import Ops, ParamArg, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

from slopkernels.src import amd as _amd, amd_gguf as _gguf

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
_IQ_GRID_LENS = {_gguf.IQ2_XXS: 512, _gguf.IQ2_XS: 1024, _gguf.IQ1_S: 4096, _gguf.IQ2_S: 2048, _gguf.IQ3_XXS: 256, _gguf.IQ3_S: 512}
_WMMA_TBL_LENS = {**_IQ_GRID_LENS, _gguf.IQ4_XS: 256, _gguf.IQ3_XXS: 512, _gguf.IQ2_S: 4096}

def _wdt(t:int) -> DType: return dtypes.uint8 if t == _gguf.MXFP4 else dtypes.uint16 if t in _gguf.HALFWORD_QUANTS else dtypes.uint32
def _p(slot:int, shape:tuple[int|UOp, ...], dt:str|DType) -> UOp: return UOp.param(slot, dt if isinstance(dt, DType) else _DTYPES[dt], shape)
def _v(slot:int, name:str) -> UOp:
  return UOp.variable(name, 0, 1 << 20).replace(arg=ParamArg(slot, dtypes.weakint, (0, 1 << 20), name, AddrSpace.ALU, None))
def _t(name:str, lo:int|float=1, hi:int|float=1 << 20, dt:DType=dtypes.weakint) -> UOp: return UOp.template(name, (lo, hi), dtype=dt)

def _b_q8(dt:str) -> UOp:
  tok, g = _t("tokens"), _t("in_groups")
  return _gguf._q8_quantize_kernel(_p(0, (tok, g, 8), "u32"), _p(1, (tok, g), "f32"), _p(2, (tok, g, 2), "f32"),
                                   _p(3, (tok, g * 32), dt), tok, g * 32)

def _b_gemv(wdt:str, xdt:str, vc:int) -> UOp:
  tok, ib, outf = _t("tokens"), _t("in_blks"), _t("out_features")
  inf = ib * (32 * vc)
  return _amd._amd_f16_gemv_kernel(_p(0, (tok, outf), "f32"), _p(1, (outf, inf), wdt), _p(2, (tok, inf), xdt),
                                   in_features=inf, out_features=outf, tokens=tok, val_chunk=vc)

def _b_qgemv(qt:int, cdna:bool) -> UOp:
  tok, inf, outf, rl = _t("tokens"), _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
  ch, gr, g = _t("chunks", 1, 1024), _t("group_rows", 0), inf // 32
  gl = (_p(5, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
  return _gguf._quant_decode_kernel(_p(0, (tok, outf, ch), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, g, 8), "u32"),
                                    _p(3, (tok, g), "f32"), _p(4, (tok, g, 2), "f32"), *gl,
                                    out_features=outf, in_features=inf, ggml_type=qt, cdna=cdna, group_rows=gr)

def _b_moe(qt:int, cdna:bool) -> UOp:
  inf, outf, rl, ch = _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30), _t("chunks", 1, 1024)
  rows, xrows, xs, nd, g = _t("rows"), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64), inf // 32
  gl = (_p(6, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
  return _gguf._moe_decode_kernel(_p(0, (rows, outf, ch), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (xrows, g, 8), "u32"),
                                  _p(3, (xrows, g), "f32"), _p(4, (xrows, g, 2), "f32"), _p(5, (rows,), "i32"), *gl,
                                  out_features=outf, in_features=inf, ggml_type=qt, x_share=xs, cdna=cdna, ndev=nd)

def _b_gemm(qt:int, kg:int, ln:int) -> UOp:
  tok, inf, outf, rl = _t("tokens"), _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
  tbl = (_p(3, (_WMMA_TBL_LENS[qt],), "u32"),) if qt in _WMMA_TBL_LENS else ()
  return _gguf._quant_linear_f16_wmma_kernel(_p(0, (tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, inf), "f16"), *tbl,
                                             out_features=outf, in_features=inf, ggml_type=qt, kgroups=kg, lanes=ln)

def _b_fdc() -> UOp:
  B, H, C, db, voff = _t("B", 1, 64), _t("H", 1, 256), _t("C", 1, 48), _t("d_blks", 1, 8), _t("valid_kv_offset", 0)
  return _amd._amd_flash_decode_combine(_p(0, (B, H, 1, db * 64), "f32"), _p(1, (B, H, C, db * 64), "f32"),
                                        _p(2, (B, H, C, 2), "f32"), valid_kv_len=_v(3, "valid_kv_len") + voff)

def _b_fdp(G:int, D:int) -> UOp:
  B, H_KV, C, nb, voff, waves = _t("B", 1, 64), _t("H_KV", 1, 256), _t("C", 1, 48), _t("n_blks", 1, 65536), _t("valid_kv_offset", 0), 16
  while waves * G * ((D + 4) * 2 + 8) > 65536: waves //= 2
  return _amd._amd_flash_attention_decode_partial(_p(0, (B, H_KV * G, C, D), "f32"), _p(1, (B, H_KV * G, C, 2), "f32"),
    _p(2, (B, H_KV * G, 1, D), "f16"), _p(3, (2, B, H_KV, nb * 64, D), "f16"),
    valid_kv_len=_v(4, "valid_kv_len") + voff, max_kv_len=nb * 64, block_n=64, waves=waves, group=G)

def _b_fa(D:int, rdna4:bool) -> UOp:
  B, H_KV, nb, voff = _t("B", 1, 64), _t("H_KV", 1, 256), _t("n_blks", 1, 65536), _t("valid_kv_offset", 0)
  BH, mb, qfe, vkl, ql = _t("BH", 1, 4096), _t("m_blks", 1, 4096), _t("q_start_from_end", 0, 1), _v(3, "valid_kv_len"), _v(4, "q_len")
  return _amd._amd_flash_attention(_p(0, (BH, mb * 32, D), "f32"), _p(1, (BH, mb * 32, D), "f16"), _p(2, (2, B, H_KV, nb * 32, D), "f16"),
    valid_kv_len=qfe.eq(1).where(vkl + voff, vkl + voff + ql), q_start=qfe.eq(1).where(vkl + voff - ql, vkl + voff), rdna4=rdna4)

def _b_hc(bdt:str, sdt:str) -> UOp:
  tok, feats, iters = _t("tokens"), _t("features", 32), _t("iters", 1, 64)
  eps, neps = _t("eps", 0.0, 1.0, dtypes.float32), _t("norm_eps", 0.0, 1.0, dtypes.float32)
  return _amd._hc_coeffs_kernel(_p(0, (tok, 24), "f32"), _p(1, (tok, 24), "f32"), _p(2, (tok, 32), "f32"),
                                _p(3, (24,), bdt), _p(4, (3,), sdt), 4, iters, eps, neps, feats)

def _b_gd(K:int, ca:bool) -> UOp:
  tok, ba, hd, vt = _t("tokens"), _t("batch", 1, 64), _t("heads", 1, 256), _t("v_tiles", 1, 1024)
  return _amd._gated_delta_prefill_kernel(_p(0, (ba, hd, tok, vt * 4), "f32"), _p(1, (ba, hd, tok, K), "f32"),
    _p(2, (ba, hd, tok, K), "f32"), _p(3, (ba, hd, tok, vt * 4), "f32"), _p(4, (ba, hd, tok), "f32"),
    _p(5, (ba, hd, tok, K) if ca else (ba, hd, tok), "f32"), _p(6, (ba, hd, vt * 4, K), "f32"),
    _p(7, (ba, hd, tok), "f32"), start_pos=_v(8, "start_pos"))

_BUILDERS: dict[str, Callable[[], UOp]] = {
  **{f"amd_q8_quantize_{dt}": functools.partial(_b_q8, dt) for dt in ("f16", "bf16", "f32")},
  **{f"amd_gemv_{w}_{x}{s}": functools.partial(_b_gemv, w, x, vc)
     for w, x in (("f16", "f32"), ("f32", "f16"), ("bf16", "f32"), ("f32", "bf16"), ("f32", "f32")) for vc, s in ((4, ""), (1, "_v1"))},
  **{f"{a}_gemv_{qn}": functools.partial(_b_qgemv, qt, cdna)
     for qt, qn in _gguf.QUANT_NAMES.items() for a, cdna in (("rdna", False), ("cdna", True))},
  **{f"{a}_moe_{qn}": functools.partial(_b_moe, qt, cdna) for qt, qn in _gguf.QUANT_NAMES.items() for a, cdna in (("rdna", False), ("cdna", True))},
  **{f"{a}_gemm_{qn}_f16": functools.partial(_b_gemm, qt, kg, ln)
     for qt, qn in _gguf.QUANT_NAMES.items() for a, kg, ln in (("rdna4", 2, 32), ("rdna3", 1, 32), ("cdna", 4, 64))},
  "amd_flash_decode_combine": _b_fdc,
  **{f"amd_flash_decode_partial_g{G}_d{D}": functools.partial(_b_fdp, G, D) for G in (1, 2, 4, 8) for D in (64, 128, 256)},
  **{f"{a}_flash_attn_d{D}": functools.partial(_b_fa, D, r4) for D in (64, 128, 256, 320) for a, r4 in (("rdna4", True), ("rdna3", False))},
  **{f"amd_hc_coeffs_{b}_{s}": functools.partial(_b_hc, b, s) for b in ("f16", "f32") for s in ("f16", "f32")},
  **{f"amd_gated_delta_k{K}_{s}": functools.partial(_b_gd, K, ca) for K in (32, 64, 128, 256) for ca, s in ((False, "a1"), (True, "ak"))},
}

def _emit(name:str, fn:Callable[[], UOp]) -> tuple[pathlib.Path, str]:
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  with Context(TRACK_MATCH_STATS=0, SPEC=0):
    u = graph_rewrite(fn(), symbolic_simple, enter_calls=True)
    if allocs := [x for x in u.toposort(enter_calls=True) if x.op is Ops.ALLOC]:
      u = u.substitute({b: b.replace(arg=replace(b.arg, slot=-1 - i)) for i, b in enumerate(allocs)}, enter_calls=True)
      u = u.substitute({b.replace(arg=replace(b.arg, slot=-1 - i)): b.replace(arg=replace(b.arg, slot=i))
                        for i, b in enumerate(allocs)}, enter_calls=True)
    uir = render_uir(u, color=False)
  dst = UIR_DIR / f"{name}.uir"
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_bytes((uir + "\n").encode("utf-8"))
  os.replace(tmp, dst)
  return dst, uir

def generate_all() -> dict[str, str]:
  """Generate all static templated `.uir` files fresh on disk in `slopkernels/uir/`."""
  return {name: _emit(name, fn)[1] for name, fn in sorted(_BUILDERS.items())}

@functools.cache
def get(name:str) -> UOp|None:
  """Generate `slopkernels/uir/<name>.uir` fresh on disk and return the parsed templated UOp."""
  if (fn := _BUILDERS.get(name)) is None: return None
  with Context(SPEC=0): return parse_uir(_emit(name, fn)[0].read_text(encoding="utf-8"))

__all__ = ["generate_all", "get"]
