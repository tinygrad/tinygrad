"""slopkernels: optional AMD RDNA3/RDNA4/CDNA templated UIR kernels (always round-tripped through slopkernels/uir/)."""
from __future__ import annotations

import functools, os, pathlib, threading
from collections.abc import Callable
from dataclasses import replace

from tinygrad import Context, UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.uop.ops import AxisType, Ops, ParamArg, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

from slopkernels.src import amd as _amd, amd_gguf as _gguf

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"
_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
_IQ_GRID_LENS = {_gguf.IQ2_XXS: 512, _gguf.IQ2_XS: 1024, _gguf.IQ1_S: 4096, _gguf.IQ2_S: 2048, _gguf.IQ3_XXS: 256, _gguf.IQ3_S: 512}
_DIRECT_GRID_LENS = {**_IQ_GRID_LENS, _gguf.IQ3_XXS: 512}
_WMMA_TBL_LENS = {**_IQ_GRID_LENS, _gguf.IQ4_XS: 256}
_R3_WMMA_TBL_LENS = {**_WMMA_TBL_LENS, _gguf.IQ3_XXS: 512}

def _wdt(t:int) -> DType: return dtypes.uint8 if t == _gguf.MXFP4 else dtypes.uint16 if t in _gguf.HALFWORD_QUANTS else dtypes.uint32
def _p(slot:int, shape:tuple[int|UOp, ...], dt:str|DType) -> UOp: return UOp.param(slot, dt if isinstance(dt, DType) else _DTYPES[dt], shape)
def _v(slot:int, name:str) -> UOp:
  return UOp.variable(name, 0, 1 << 20).replace(arg=ParamArg(slot, dtypes.weakint, (0, 1 << 20), name, AddrSpace.ALU, None))
def _t(name:str, lo:int|float=1, hi:int|float=1 << 20, dt:DType=dtypes.weakint) -> UOp: return UOp.template(name, (lo, hi), dtype=dt)

def _b_gemv(wdt:str, xdt:str) -> UOp:
  tok, ib, outf, vc = _t("tokens"), _t("in_blks"), _t("out_features"), _t("val_chunk", 1, 4)
  inf = ib * (32 * vc)
  args = (_p(0, (tok, outf), "f32"), _p(1, (outf, inf), wdt), _p(2, (tok, inf), xdt))
  k_base = _amd._amd_f16_gemv_kernel(*args, in_blks=ib, out_features=outf, tokens=tok, val_chunk=vc)
  if wdt != "f16": return k_base
  k_lds = _amd._amd_f16_gemv_kernel(*args, in_blks=ib, out_features=outf, tokens=tok, val_chunk=vc, lds_mt=True)
  return UOp(Ops.WHERE, src=((tok > 1) & (tok <= 8) & (inf % 1024).eq(0) & (outf >= 1024) & (outf % 8).eq(0), k_lds, k_base))

def _b_qgemv(qt:int, xdt:str) -> UOp:
  tok, inf, outf, rl, gr = _t("tokens"), _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 40), _t("group_rows", 0)
  gl = (_p(3, (_DIRECT_GRID_LENS[qt],), "u32"),) if qt in _DIRECT_GRID_LENS else ()
  p_out, p_raw, p_x = _p(0, (tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, inf), xdt)
  t_eff = (gr > 0).where(gr, tok)
  use_lds = (t_eff >= 2) & (t_eff <= 12) & (inf >= 1024) & (outf % 4).eq(0)
  def mk(al4:bool) -> UOp:
    b = functools.partial(_gguf._direct_gemv_body, p_out, p_raw, p_x, gl, outf, inf, qt, aligned4=al4, group_rows=gr)
    return UOp(Ops.WHERE, src=(use_lds, b(x_lds=True), _gguf._select_direct_variants(b, inf, outf, qt)))
  return UOp(Ops.WHERE, src=((rl % 4).eq(0), mk(True), mk(False))) if qt == _gguf.MXFP4 else mk(True)

def _b_moe(qt:int, xdt:str) -> UOp:
  inf, outf, rl = _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 40)
  rows, xrows, xs, nd = _t("rows"), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64)
  gl = (_p(4, (_DIRECT_GRID_LENS[qt],), "u32"),) if qt in _DIRECT_GRID_LENS else ()
  p_out, p_raw, p_x, p_sel = _p(0, (rows, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (xrows, inf), xdt), _p(3, (rows,), "i32")
  expert_words = outf * inf * _gguf.QUANT_SIZES[qt] // 256 // p_raw.dtype.itemsize
  local_experts = p_raw.shape[0] // expert_words
  def expert(row:UOp) -> UOp: return p_sel[row].cast(dtypes.weakint) - UOp.range(nd, 10, AxisType.DEVICE) * local_experts
  def mk(al4:bool) -> UOp:
    b = functools.partial(_gguf._direct_gemv_body, p_out, p_raw, p_x, gl, outf, inf, qt, aligned4=al4,
                          gate=lambda row, base: base + expert(row).maximum(0).minimum(local_experts - 1) * expert_words,
                          xtoken=lambda row: row // xs, name="moe_", row_mask=lambda row: (expert(row) >= 0) & (expert(row) < local_experts))
    k_dec = _gguf._select_direct_variants(b, inf, outf, qt)
    rpw_pre = (outf % 4).eq(0).where(4, (outf % 2).eq(0).where(2, 1))
    k_pre = UOp(Ops.WHERE, src=(inf <= 2048, b(rpw=rpw_pre, w1_only=True, stream_w=False), b(rpw=rpw_pre, wk_cap=4, stream_w=False)))
    return UOp(Ops.WHERE, src=(rows >= 16, k_pre, k_dec))
  return UOp(Ops.WHERE, src=((rl % 4).eq(0), mk(True), mk(False))) if qt == _gguf.MXFP4 else mk(True)

def _b_gemm(qt:int, kg:int, ln:int) -> UOp:
  mb, mt, inf, outf, rl = _t("m_blks"), _t("m_tiles", 1, 8), _t("in_features", 32), _t("out_features", 16), _t("raw_len", 1, 1 << 40)
  tok = mb * (mt * 16)
  p_out, p_raw, p_x = _p(0, (tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, inf), "f16")
  tbl_map = _R3_WMMA_TBL_LENS if kg == 1 else _WMMA_TBL_LENS
  tbl = (_p(3, (tbl_map[qt],), "u32"),) if qt in tbl_map else ()
  if kg == 1:  # RDNA3: Direct-A + Double-Buffered-B 2D FP16 WMMA + in-WG K-split (wk=2,4)
    c_lg = (tok % 128).eq(0) & (outf % 64).eq(0) & (inf % 128).eq(0) & (((tok // 128) * (outf // 64)) >= 192)
    c_md = (tok >= 64) & (tok % 64).eq(0) & (outf % 32).eq(0) & (inf % 256).eq(0) & (((tok // 64) * (outf // 32)) >= 96)
    wm = c_lg.where(4, c_md.where(2, 1))
    tm, tn = c_lg.where(2, c_md.where(2, (tok % 32).eq(0).where(2, 1))), c_lg.where(4, c_md.where(2, 1))
    bk = c_lg.where(64, c_md.where(128, (inf % 64).eq(0).where(64, 32)))
    k1 = _gguf._rdna3_gemm_2d_kernel(p_out, p_raw, p_x, *tbl, m_blks=tok // (wm * tm * 16), n_blks=outf // (tn * 16),
                                     wm=wm, wn=UOp.const(1, dtypes.weakint), tm=tm, tn=tn, bk=bk, out_features=outf, in_features=inf,
                                     ggml_type=qt, wk=1, wk_split=False)
    use_w4 = (tok % 32).eq(0) & (inf >= 256) & (inf % 256).eq(0) & ((tok.eq(32)) | ((tok * outf) <= 128 * 1024))
    use_w2 = (tok >= 128) & (tok % 64).eq(0) & (outf % 64).eq(0) & (inf >= 8192) & (inf % 128).eq(0) & (outf > 4096)
    k_wk = _gguf._rdna3_gemm_2d_kernel(p_out, p_raw, p_x, *tbl, m_blks=use_w4.where(tok // 32, tok // 64),
                                       n_blks=use_w4.where(outf // 16, outf // 64), wm=UOp.const(1, dtypes.weakint),
                                       wn=use_w4.where(1, 2), tm=use_w4.where(2, 4), tn=use_w4.where(1, 2),
                                       bk=use_w4.where(256, 128), out_features=outf, in_features=inf,
                                       ggml_type=qt, wk=use_w4.where(4, 2), wk_split=True)
    return UOp(Ops.WHERE, src=(use_w4 | use_w2, k_wk, k1))
  k_f16 = _gguf._quant_linear_f16_wmma_kernel(p_out, p_raw, p_x, *tbl, m_blks=mb, m_tiles=mt, out_features=outf, in_features=inf,
                                              ggml_type=qt, kgroups=kg, lanes=ln)
  if kg != 2: return k_f16
  k_zlds = _gguf._rdna4_zlds_f16_kernel(p_out, p_raw, p_x, *tbl, tok=tok, out_features=outf, in_features=inf, ggml_type=qt)
  sel = UOp(Ops.WHERE, src=((inf >= 1024) & (inf % 1024).eq(0) & (outf >= 64) & (outf % 32).eq(0) & (tok >= 16), k_zlds, k_f16))
  if qt not in _gguf.OTF_FP8_QUANTS: return sel
  k_128 = _gguf._rdna4_gemm_fp8_kernel(p_out, p_raw, p_x, *tbl, tok=tok, out_features=outf, in_features=inf, ggml_type=qt,
                                       wm=4, wn=4, tm=2, tn=2, double_buf=False, lanes_a=8 if qt == _gguf.Q4_K else 4)
  k_64x128 = _gguf._rdna4_gemm_fp8_kernel(p_out, p_raw, p_x, *tbl, tok=tok, out_features=outf, in_features=inf, ggml_type=qt,
                                          wm=4, wn=4, tm=1, tn=2, double_buf=True, lanes_a=8)
  can = (inf >= 512) & (inf % 256).eq(0) & (outf >= 2048) & (outf % 128).eq(0) & (tok >= 64)
  sel = UOp(Ops.WHERE, src=(can & (tok >= 128 if qt == _gguf.Q8_0 else tok >= 64), k_64x128, sel))
  return UOp(Ops.WHERE, src=(can & ((tok >= 256) | ((tok >= 128) & ((outf >= 8192) | ((outf > 4096) & (inf >= 8192))))), k_128, sel))

def _b_fdc() -> UOp:
  B, H, C, db, voff = _t("B", 1, 64), _t("H", 1, 256), _t("C", 1, 48), _t("d_blks", 1, 8), _t("valid_kv_offset", 0)
  return _amd._amd_flash_decode_combine(_p(0, (B, H, 1, db * 64), "f32"), _p(1, (B, H, C, db * 64), "f32"),
                                        _p(2, (B, H, C, 2), "f32"), valid_kv_len=_v(3, "valid_kv_len") + voff)

def _b_fdp() -> UOp:
  B, H_KV, G, db, C, nb = _t("B", 1, 64), _t("H_KV", 1, 256), _t("G", 1, 8), _t("d_blks", 1, 8), _t("C", 1, 48), _t("n_blks", 1, 65536)
  voff, waves, D = _t("valid_kv_offset", 0), _t("waves", 1, 16), db * 64
  args = (_p(0, (B, H_KV * G, C, D), "f32"), _p(1, (B, H_KV * G, C, 2), "f32"),
          _p(2, (B, H_KV * G, 1, D), "f16"), _p(3, (2, B, H_KV, nb * 64, D), "f16"))
  kw = dict(valid_kv_len=_v(4, "valid_kv_len") + voff, d_blks=db, group=G, waves=waves)
  k_base = _amd._amd_flash_attention_decode_partial(*args, **kw, g_split=1, lanes_per_tok=32)
  k_d256 = _amd._amd_flash_attention_decode_partial(*args, **kw, g_split=2, lanes_per_tok=32)
  k_d128 = _amd._amd_flash_attention_decode_partial(*args, **kw, g_split=2, lanes_per_tok=16)
  can_gs2 = (G % 2).eq(0) & (waves >= 2)
  return UOp(Ops.WHERE, src=(can_gs2 & db.eq(2), k_d128, UOp(Ops.WHERE, src=(can_gs2 & (db >= 4), k_d256, k_base))))

def _b_fa(rdna4:bool) -> UOp:
  B, H_KV, db, nb, voff = _t("B", 1, 64), _t("H_KV", 1, 256), _t("d_blks", 1, 8), _t("n_blks", 1, 65536), _t("valid_kv_offset", 0)
  BH, mb, qfe, vkl, ql, D = _t("BH", 1, 4096), _t("m_blks", 1, 4096), _t("q_start_from_end", 0, 1), _v(3, "valid_kv_len"), _v(4, "q_len"), db * 64
  args = (_p(0, (BH, mb * 32, D), "f32"), _p(1, (BH, mb * 32, D), "f16"), _p(2, (2, B, H_KV, nb * 32, D), "f16"))
  vkl_e, qs_e = qfe.eq(1).where(vkl + voff, vkl + voff + ql), qfe.eq(1).where(vkl + voff - ql, vkl + voff)
  k_sm = _amd._amd_flash_attention_smallm(*args, valid_kv_len=vkl_e, q_start=qs_e, d_blks=db, rdna4=rdna4)
  k_gen = _amd._amd_flash_attention(*args, valid_kv_len=vkl_e, q_start=qs_e, d_blks=db, rdna4=rdna4)
  return UOp(Ops.WHERE, src=(mb.eq(1) & (db <= 2) & (nb >= 4), k_sm, k_gen))

def _b_hc(bdt:str, sdt:str) -> UOp:
  tok, feats, iters = _t("tokens"), _t("features", 32), _t("iters", 1, 64)
  eps, neps = _t("eps", 0.0, 1.0, dtypes.float32), _t("norm_eps", 0.0, 1.0, dtypes.float32)
  return _amd._hc_coeffs_kernel(_p(0, (tok, 24), "f32"), _p(1, (tok, 24), "f32"), _p(2, (tok, feats), "f32"),
                                _p(3, (24,), bdt), _p(4, (3,), sdt), 4, iters, eps, neps)

def _b_gd(arch:str) -> UOp:
  tok, ba, hd, vt, kb, ak = _t("tokens"), _t("batch", 1, 64), _t("heads", 1, 256), _t("v_tiles", 1, 1024), _t("k_blks", 1, 8), _t("alpha_k", 1, 256)
  K = kb * 32
  return _amd._gated_delta_prefill_kernel(_p(0, (ba, hd, tok, vt * 4), "f32"), _p(1, (ba, hd, tok, K), "f32"), _p(2, (ba, hd, tok, K), "f32"),
    _p(3, (ba, hd, tok, vt * 4), "f32"), _p(4, (ba, hd, tok), "f32"), _p(5, (ba, hd, tok, ak), "f32"), _p(6, (ba, hd, vt * 4, K), "f32"),
    k_blks=kb, alpha_k=ak, start_pos=_v(7, "start_pos"), arch=arch)

_BUILDERS: dict[str, Callable[[], UOp]] = {
  **{f"amd_gemv_{w}_{x}": functools.partial(_b_gemv, w, x)
     for w, x in (("f16", "f32"), ("f32", "f16"), ("bf16", "f32"), ("f32", "bf16"), ("f32", "f32"))},
  **{f"amd_gemv_{qn}_{dt}": functools.partial(_b_qgemv, qt, dt) for qt, qn in _gguf.QUANT_NAMES.items() for dt in ("f16", "bf16", "f32")},
  **{f"amd_moe_{qn}_{dt}": functools.partial(_b_moe, qt, dt) for qt, qn in _gguf.QUANT_NAMES.items() for dt in ("f16", "bf16", "f32")},
  **{f"{a}_gemm_{qn}_f16": functools.partial(_b_gemm, qt, kg, ln)
     for qt, qn in _gguf.QUANT_NAMES.items() for a, kg, ln in (("rdna4", 2, 32), ("rdna3", 1, 32), ("cdna", 4, 64))},
  "amd_flash_decode_combine": _b_fdc, "amd_flash_decode_partial": _b_fdp,
  "rdna4_flash_attn": functools.partial(_b_fa, True), "rdna3_flash_attn": functools.partial(_b_fa, False),
  **{f"amd_hc_coeffs_{b}_{s}": functools.partial(_b_hc, b, s) for b in ("f16", "f32") for s in ("f16", "f32")},
  **{f"{a}_gated_delta": functools.partial(_b_gd, a) for a in ("rdna4", "rdna3", "amd")},
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
