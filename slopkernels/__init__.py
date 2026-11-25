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
_WMMA_TBL_LENS = {**_IQ_GRID_LENS, _gguf.IQ3_XXS: 512, _gguf.IQ2_S: 4096}

_DIRECT_RPW = {_gguf.Q8_0: 4, _gguf.Q2_K: 2, _gguf.Q3_K: 2, _gguf.Q4_K: 2, _gguf.IQ1_S: 16,
               _gguf.IQ4_NL: 2, _gguf.IQ3_S: 16, _gguf.IQ2_S: 16, _gguf.MXFP4: 2}
_MOE_RPW = {_gguf.Q8_0: 4, _gguf.Q4_K: 2, _gguf.IQ1_S: 16, _gguf.IQ4_XS: 2, _gguf.MXFP4: 2}

def _wdt(t:int) -> DType: return dtypes.uint8 if t == _gguf.MXFP4 else dtypes.uint16 if t in _gguf.HALFWORD_QUANTS else dtypes.uint32
def _p(slot:int, shape:tuple[int|UOp, ...], dt:str|DType) -> UOp: return UOp.param(slot, dt if isinstance(dt, DType) else _DTYPES[dt], shape)
def _v(slot:int, name:str) -> UOp:
  return UOp.variable(name, 0, 1 << 20).replace(arg=ParamArg(slot, dtypes.weakint, (0, 1 << 20), name, AddrSpace.ALU, None))
def _t(name:str, lo:int|float=1, hi:int|float=1 << 20, dt:DType=dtypes.weakint) -> UOp: return UOp.template(name, (lo, hi), dtype=dt)
def _rpw(outf:UOp, t:int) -> UOp: return ((outf >= 1024) & (outf % t).eq(0)).where(t, 1) if t > 1 else UOp.const(1, dtypes.weakint)

def _b_q8(dt:str) -> UOp:
  tok, g = _t("tokens"), _t("in_groups")
  return _gguf._q8_quantize_kernel(_p(0, (tok, g, 8), "u32"), _p(1, (tok, g), "f32"), _p(2, (tok, g, 2), "f32"),
                                   _p(3, (tok, g * 32), dt), tok, g * 32)

def _b_gemv(wdt:str, xdt:str) -> UOp:
  tok, inf, outf = _t("tokens"), _t("in_features", 32), _t("out_features")
  vc = (inf % 256).eq(0).where(8, (inf % 128).eq(0).where(4, 1)) if _DTYPES[wdt].itemsize == 2 else (inf % 128).eq(0).where(4, 1)
  ku = ((inf // (32 * vc)) % 2).eq(0).where(2, 1)
  return _amd._amd_f16_gemv_kernel(_p(0, (tok, outf), "f32"), _p(1, (outf, inf), wdt), _p(2, (tok, inf), xdt),
                                   in_blks=inf // (32 * vc * ku), out_features=outf, tokens=tok, val_chunk=vc, k_unroll=ku)

def _b_qgemv(qt:int, cdna:bool) -> UOp:
  tok, inf, outf, rl, gr = _t("tokens"), _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30), _t("group_rows", 0)
  wv, g = ((inf + 1023) // 1024).minimum(32), inf // 32
  gl = (_p(5, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
  return _gguf._quant_decode_kernel(_p(0, (tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, g, 8), "u32"),
                                    _p(3, (tok, g), "f32"), _p(4, (tok, g, 2), "f32"), *gl,
                                    out_features=outf, in_features=inf, waves=wv, ggml_type=qt, cdna=cdna, group_rows=gr)

def _b_qgemv_m(qt:int, cdna:bool) -> UOp:
  tok, inf, outf, rl = _t("tokens", 2, 8), _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
  wv, g = ((inf + 1023) // 1024).minimum(32), inf // 32
  return _gguf._quant_decode_m_kernel(_p(0, (tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, g, 8), "u32"),
                                      _p(3, (tok, g), "f32"), _p(4, (tok, g, 2), "f32"),
                                      tokens=tok, out_features=outf, in_features=inf, waves=wv, ggml_type=qt, cdna=cdna)

def _b_qgemv_direct(qt:int, xdt:str) -> UOp:
  inf, outf, rl = _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
  gl = (_p(3, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
  return _gguf._quant_direct_gemv_kernel(_p(0, (1, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (1, inf), xdt), *gl,
                                         out_features=outf, in_features=inf, waves=((inf + 511) // 512).minimum(32),
                                         rows_per_wg=_rpw(outf, _DIRECT_RPW.get(qt, 1)), ggml_type=qt)

def _b_moe(qt:int, cdna:bool) -> UOp:
  inf, outf, rl = _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
  rows, xrows, xs, nd = _t("rows"), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64)
  wv, g = ((inf + 1023) // 1024).minimum(32), inf // 32
  gl = (_p(6, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
  return _gguf._moe_decode_kernel(_p(0, (rows, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (xrows, g, 8), "u32"),
                                  _p(3, (xrows, g), "f32"), _p(4, (xrows, g, 2), "f32"), _p(5, (rows,), "i32"), *gl,
                                  out_features=outf, in_features=inf, waves=wv, ggml_type=qt, x_share=xs, cdna=cdna, ndev=nd)

def _b_moe_direct(qt:int) -> UOp:
  inf, outf, rl = _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
  rows, xrows, xs, nd = _t("rows"), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64)
  gl = (_p(4, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
  return _gguf._moe_direct_kernel(_p(0, (rows, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (xrows, inf), "f32"), _p(3, (rows,), "i32"), *gl,
                                  out_features=outf, in_features=inf, waves=((inf + 511) // 512).minimum(32),
                                  rows_per_wg=_rpw(outf, _MOE_RPW.get(qt, 1)), ggml_type=qt, x_share=xs, ndev=nd)

def _b_gemm(qt:int, kg:int, ln:int) -> UOp:
  tok, inf, outf, rl = _t("tokens", 16), _t("in_features", 32), _t("out_features", 16), _t("raw_len", 1, 1 << 30)
  mt = ((tok % 128).eq(0) & ((tok // 128) * (outf // 16) >= 512)).where(8,
       ((tok % 64).eq(0) & ((tok // 64) * (outf // 16) >= 512)).where(4,
       ((tok % 32).eq(0) & ((tok // 32) * (outf // 16) >= 256)).where(2, 1)))
  tbl = (_p(3, (_WMMA_TBL_LENS[qt],), "u32"),) if qt in _WMMA_TBL_LENS else ()
  return _gguf._quant_linear_f16_wmma_kernel(_p(0, (tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, inf), "f16"), *tbl,
                                             m_blks=tok // (mt * 16), m_tiles=mt, out_features=outf, in_features=inf,
                                             ggml_type=qt, kgroups=kg, lanes=ln)

def _b_gemm_2d(qt:int, rdna4:bool) -> UOp:
  tok, outf, inf, rl, sk = _t("tokens", 32), _t("out_features", 64), _t("in_features", 64), _t("raw_len", 1, 1 << 30), _t("split_k", 1, 16)
  c256 = (tok % 256).eq(0) & (outf % 128).eq(0) if qt != _gguf.Q4_K else UOp.const(False)
  c128 = (tok >= 256) & (tok % 128).eq(0) & (outf % 128).eq(0)
  wm = c256.where(4, c128.where(2, (tok % 128).eq(0).where(2, (tok % 64).eq(0).where(2, 1))))
  wn, tm, tn = c256.where(2, c128.where(4, 2)), c256.where(4, c128.where(4, (tok % 128).eq(0).where(4, 2))), c256.where(4, 2)
  tbl = (_p(3, (_WMMA_TBL_LENS[qt],), "u32"),) if qt in _WMMA_TBL_LENS else ()
  return _gguf._quant_linear_2d_wmma_kernel(_p(0, (sk * tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, inf), "f16"), *tbl,
                                            m_blks=tok // (wm * tm * 16), n_blks=outf // (wn * tn * 16), wm=wm, wn=wn, tm=tm, tn=tn,
                                            in_features=inf, split_k=sk, ggml_type=qt, rdna4=rdna4)

def _b_fdc() -> UOp:
  B, H, C, db, voff, sk = _t("B", 1, 64), _t("H", 1, 256), _t("C", 1, 48), _t("d_blks", 1, 8), _t("valid_kv_offset", 0), _t("step_keys", 1, 64)
  return _amd._amd_flash_decode_combine(_p(0, (B, H, 1, db * 64), "f32"), _p(1, (B, H, C, db * 64), "f16"),
                                        _p(2, (B, H, C, 2), "f32"), valid_kv_len=_v(3, "valid_kv_len") + voff, step_keys=sk)

def _b_fdp() -> UOp:
  B, H_KV, G, db, C, nb = _t("B", 1, 64), _t("H_KV", 1, 256), _t("G", 1, 8), _t("d_blks", 1, 8), _t("C", 1, 48), _t("n_blks", 1, 65536)
  voff, waves, D = _t("valid_kv_offset", 0), _t("waves", 1, 16), db * 64
  return _amd._amd_flash_attention_decode_partial(_p(0, (B, H_KV * G, C, D), "f16"), _p(1, (B, H_KV * G, C, 2), "f32"),
    _p(2, (B, H_KV * G, 1, D), "f16"), _p(3, (2, B, H_KV, nb * 64, D), "f16"),
    valid_kv_len=_v(4, "valid_kv_len") + voff, d_blks=db, group=G, waves=waves)

def _b_fa(rdna4:bool) -> UOp:
  B, H_KV, db, N, voff = _t("B", 1, 64), _t("H_KV", 1, 256), _t("d_blks", 1, 8), _t("N", 32, 65536 * 64), _t("valid_kv_offset", 0)
  BH, M, qfe, D = _t("BH", 1, 4096), _t("M", 32, 65536), _t("q_start_from_end", 0, 1), db * 64
  gqa_w = ((db <= 2) & ((BH // (B * H_KV)) % 4).eq(0) & (db.eq(2) | (N % 64).eq(0))).where(4, 1) if rdna4 else UOp.const(1, dtypes.weakint)
  wm = gqa_w.eq(4).where(8, (M % 64).eq(0).where(4, 2))
  bnt = ((db <= 2) & (N % 64).eq(0) & (gqa_w.eq(4) & db.eq(2)).logical_not()).where(4, 2) if rdna4 else UOp.const(2, dtypes.weakint)
  vkl, ql = _v(3, "valid_kv_len"), _v(4, "q_len")
  return _amd._amd_flash_attention(_p(0, (BH, M, D), "f32"), _p(1, (BH, M, D), "f16"), _p(2, (2, B, H_KV, N, D), "f16"),
    valid_kv_len=qfe.eq(1).where(vkl + voff, vkl + voff + ql), q_start=qfe.eq(1).where(vkl + voff - ql, vkl + voff),
    d_blks=db, m_blks=M // ((wm // gqa_w) * 16), waves_m=wm, bn_tiles=bnt, gqa_w=gqa_w, rdna4=rdna4)

def _b_hc(bdt:str, sdt:str) -> UOp:
  tok, feats, iters = _t("tokens"), _t("features", 32), _t("iters", 1, 64)
  eps, neps = _t("eps", 0.0, 1.0, dtypes.float32), _t("norm_eps", 0.0, 1.0, dtypes.float32)
  return _amd._hc_coeffs_kernel(_p(0, (tok, 24), "f32"), _p(1, (tok, 24), "f32"), _p(2, (tok, feats), "f32"),
                                _p(3, (24,), bdt), _p(4, (3,), sdt), 4, ((feats + 511) // 512).maximum(1).minimum(8), iters, eps, neps, feats)

def _b_gd() -> UOp:
  tok, ba, hd, vt, kb, ak = _t("tokens"), _t("batch", 1, 64), _t("heads", 1, 256), _t("v_tiles", 1, 1024), _t("k_blks", 1, 8), _t("alpha_k", 1, 256)
  K = kb * 32
  return _amd._gated_delta_prefill_kernel(_p(0, (ba, hd, tok, vt * 4), "f32"), _p(1, (ba, hd, tok, K), "f32"),
    _p(2, (ba, hd, tok, K), "f32"), _p(3, (ba, hd, tok, vt * 4), "f32"), _p(4, (ba, hd, tok), "f32"),
    _p(5, (ba, hd, tok, ak), "f32"), _p(6, (ba, hd, vt * 4, K), "f32"),
    _p(7, (ba, hd, tok), "f32"), k_blks=kb, alpha_k=ak, start_pos=_v(8, "start_pos"))

def _b_gd_wmma() -> UOp:
  ba, hd, tok, db = _t("batch", 1, 64), _t("heads", 1, 256), _t("tokens"), _t("d_blks", 1, 2)
  D = db * 64
  return _amd._gated_delta_prefill_wmma_kernel(_p(0, (ba, hd, tok, D), "f32"), _p(1, (ba, hd, tok, D), "f32"),
    _p(2, (ba, hd, tok, D), "f32"), _p(3, (ba, hd, tok, D), "f32"), _p(4, (ba, hd, tok), "f32"),
    _p(5, (ba, hd, tok), "f32"), _p(6, (ba, hd, D, D), "f32"), d_blks=db, start_pos=_v(7, "start_pos"))

def _b_r4_qrow() -> UOp:
  mt, kg = _t("m_tiles", 1, 4096), _t("k_groups", 8, 4096)
  return _gguf._rdna4_qrow_tiled_kernel(_p(0, (mt, kg, 2, 16, 4), "u32"), _p(1, (mt * 16,), "f32"), _p(2, (mt * 16, kg * 32), "f16"),
                                        m_tiles=mt, k_groups=kg, waves=(kg // 16).maximum(1).minimum(8))

def _b_r4_gemm_fp8(qt:int) -> UOp:
  mb, nb, kp, rl = _t("m_blks", 1, 4096), _t("n_blks", 1, 4096), _t("k_pairs", 1, 4096), _t("raw_len", 1, 1 << 30)
  gl = (_p(4, (_IQ_GRID_LENS[qt],), "u32"),) if qt in (_gguf.IQ2_S, _gguf.IQ3_S) else ()
  return _gguf._rdna4_gemm_fp8_kernel(_p(0, (mb * 128, nb * 64), "f32"), _p(1, (rl,), _wdt(qt)),
                                      _p(2, (mb * 8, kp * 8, 2, 16, 4), "u32"), _p(3, (mb * 128,), "f32"), *gl,
                                      m_blks=mb, n_blks=nb, k_pairs=kp, ggml_type=qt)

_BUILDERS: dict[str, Callable[[], UOp]] = {
  **{f"amd_q8_quantize_{dt}": functools.partial(_b_q8, dt) for dt in ("f16", "bf16", "f32")},
  **{f"amd_gemv_{w}_{x}": functools.partial(_b_gemv, w, x)
     for w, x in (("f16", "f32"), ("f32", "f16"), ("bf16", "f32"), ("f32", "bf16"), ("f32", "f32"))},
  **{f"{a}_gemv_{qn}": functools.partial(_b_qgemv, qt, cdna)
     for qt, qn in _gguf.QUANT_NAMES.items() for a, cdna in (("rdna", False), ("cdna", True))},
  **{f"{a}_gemv_m_{_gguf.QUANT_NAMES[qt]}": functools.partial(_b_qgemv_m, qt, cdna)
     for qt in _gguf.DP4A_M_QUANTS for a, cdna in (("rdna", False), ("cdna", True))},
  **{f"rdna_gemv_direct_{_gguf.QUANT_NAMES[qt]}_{xdt}": functools.partial(_b_qgemv_direct, qt, xdt)
     for qt in _gguf.DIRECT_GEMV_QUANTS for xdt in ("f32", "f16")},
  **{f"{a}_moe_{qn}": functools.partial(_b_moe, qt, cdna) for qt, qn in _gguf.QUANT_NAMES.items() for a, cdna in (("rdna", False), ("cdna", True))},
  **{f"rdna_moe_direct_{_gguf.QUANT_NAMES[qt]}": functools.partial(_b_moe_direct, qt) for qt in _gguf.DIRECT_MOE_QUANTS},
  **{f"{a}_gemm_{qn}_f16": functools.partial(_b_gemm, qt, kg, ln)
     for qt, qn in _gguf.QUANT_NAMES.items() for a, kg, ln in (("rdna4", 2, 32), ("rdna3", 1, 32), ("cdna", 4, 64))},
  **{f"{a}_gemm_{qn}_2d": functools.partial(_b_gemm_2d, qt, rdna4)
     for qt, qn in _gguf.QUANT_NAMES.items() for a, rdna4 in (("rdna4", True), ("rdna3", False))},
  "rdna4_fp8_qrow_tiled": _b_r4_qrow,
  **{f"rdna4_gemm_{_gguf.QUANT_NAMES[qt]}_fp8": functools.partial(_b_r4_gemm_fp8, qt) for qt in _gguf.RDNA4_FP8_QUANTS},
  "amd_flash_decode_combine": _b_fdc,
  "amd_flash_decode_partial": _b_fdp,
  "rdna4_flash_attn": functools.partial(_b_fa, True),
  "rdna3_flash_attn": functools.partial(_b_fa, False),
  **{f"amd_hc_coeffs_{b}_{s}": functools.partial(_b_hc, b, s) for b in ("f16", "f32") for s in ("f16", "f32")},
  "amd_gated_delta": _b_gd,
  "rdna4_gated_delta_wmma": _b_gd_wmma,
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
