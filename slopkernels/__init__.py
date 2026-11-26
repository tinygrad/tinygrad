"""slopkernels: optional AMD RDNA3/RDNA4/CDNA templated UIR kernels (always round-tripped through tinygrad/llm/kernels/)."""
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

UIR_DIR = pathlib.Path(__file__).resolve().parent.parent / "tinygrad" / "llm" / "kernels"
_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}

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

def _b_qgemv() -> UOp:
  qt, tok, inf, outf = _t("qtype", 8, 40), _t("tokens"), _t("in_features", 32), _t("out_features")
  rb, gl, gr = _t("raw_bytes", 1, 1 << 30), _t("grid_len", 1, 4096), _t("group_rows", 0)
  use_sel, xrows, xs, nd = _t("use_sel", 0, 1), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64)
  p_out, p_raw, p_x = _p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (xrows, inf), "f32")
  p_sel, p_grid = _p(3, (use_sel.eq(1).where(tok, 1),), "i32"), _p(4, (gl,), "u32")
  is_moe = use_sel.eq(1)
  expert_bytes = outf * inf * _gguf._qsize(qt) // 256
  local_experts = (rb // expert_bytes).maximum(1)
  def expert(row:UOp) -> UOp: return p_sel[row].cast(dtypes.weakint) - UOp.range(nd, 10, AxisType.DEVICE) * local_experts
  w1 = (inf <= 1024) | (is_moe.logical_not() & tok.eq(1) & (outf <= 1024) & (inf <= 4096)) | (is_moe & (tok >= 16) & (inf <= 2048))
  is_rpw2 = _gguf._qswitch(qt, [(q, UOp.const(q in _gguf._RPW2_QUANTS, dtypes.bool)) for q in _gguf.QUANT_NAMES])
  rpw_dec = is_rpw2.where((inf <= 1024).where(1, (outf % 2).eq(0).where(2, 1)), 1)
  rpw_pre = (outf % 4).eq(0).where(4, (outf % 2).eq(0).where(2, 1))
  rpw = (is_moe & (tok >= 16)).where(rpw_pre, rpw_dec)
  wk_cap = (is_moe & (tok >= 16)).where(4, rpw_dec.eq(2).where(8, 16))
  def _gate(row:UOp, base:UOp, q:int, isz:int) -> UOp:
    ew = outf * inf * (_gguf.QUANT_SIZES[q] // isz) // 256
    le = ((rb // isz) // ew).maximum(1)
    ex = p_sel[row].cast(dtypes.weakint) - UOp.range(nd, 10, AxisType.DEVICE) * le
    return is_moe.where(base + ex.maximum(0).minimum(le - 1) * ew, base)
  return _gguf._direct_gemv_body(
    p_out, p_raw, p_x, (p_grid,), outf, inf, qt, rpw=rpw, wk_cap=wk_cap, w1_cond=w1,
    aligned4=False, lds_grid=True, group_rows=gr, gate=_gate,
    xtoken=lambda row: row // xs, row_mask=lambda row: is_moe.logical_not() | ((expert(row) >= 0) & (expert(row) < local_experts)))

def _b_quant_q8_1(xdt:str) -> UOp:
  tok, inf = _t("tokens"), _t("in_features", 32)
  return _gguf._quant_q8_1_kernel(_p(0, (tok, inf // 32, 8), "u32"), _p(1, (tok, inf // 32, 3), "f32"), _p(2, (tok, inf), xdt), tok, inf)

def _b_sdot4() -> UOp:
  qt, tok, inf, outf = _t("qtype", 8, 40), _t("tokens"), _t("in_features", 32), _t("out_features")
  rb, gl, gr = _t("raw_bytes", 1, 1 << 30), _t("grid_len", 1, 4096), _t("group_rows", 0)
  p_out, p_raw = _p(0, (tok, outf), "f32"), _p(1, (rb,), "u8")
  p_xq, p_xds, p_grid = _p(2, (tok, inf // 32, 8), "u32"), _p(3, (tok, inf // 32, 3), "f32"), _p(4, (gl,), "u32")
  is_iq3x = qt.eq(_gguf.IQ3_XXS)
  wk = is_iq3x.where((inf >= 2048).where(2, 1), 1)
  wr = is_iq3x.where((outf % 4).eq(0).where(4, 1), (outf % 8).eq(0).where(8, (outf % 4).eq(0).where(4, 1)))
  return _gguf._sdot4_gemv_kernel(p_out, p_raw, p_xq, p_xds, (p_grid,), outf, inf, qt, w_rows=wr, waves_k=wk, aligned4=False, group_rows=gr)

def _b_gemm(kg:int, ln:int) -> UOp:
  qt, mb, mt, inf, outf = _t("qtype", 8, 40), _t("m_blks"), _t("m_tiles", 1, 8), _t("in_features", 32), _t("out_features", 16)
  rb, gl = _t("raw_bytes", 1, 1 << 30), _t("grid_len", 1, 4096)
  tok = mb * (mt * 16)
  p_out, p_raw, p_x, p_grid = _p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (tok, inf), "f16"), _p(3, (gl,), "u32")
  if kg == 1:  # RDNA3: Direct-A + Double-Buffered-B 2D FP16 WMMA + in-WG K-split (wk=2,4)
    c_lg = (tok % 128).eq(0) & (outf % 64).eq(0) & (inf % 128).eq(0) & (((tok // 128) * (outf // 64)) >= 192)
    c_md = (tok >= 64) & (tok % 64).eq(0) & (outf % 32).eq(0) & (inf % 256).eq(0) & (((tok // 64) * (outf // 32)) >= 96)
    use_w4 = (tok % 32).eq(0) & (inf >= 256) & (inf % 256).eq(0) & ((tok.eq(32)) | ((tok * outf) <= 128 * 1024))
    use_w2 = (tok >= 128) & (tok % 64).eq(0) & (outf % 64).eq(0) & (inf >= 8192) & (inf % 128).eq(0) & (outf > 4096)
    use_wk = use_w4 | use_w2
    wm = use_wk.where(1, c_lg.where(4, c_md.where(2, 1)))
    wn = use_wk.where(use_w4.where(1, 2), 1)
    tm = use_wk.where(use_w4.where(2, 4), c_lg.where(2, c_md.where(2, (tok % 32).eq(0).where(2, 1))))
    tn = use_wk.where(use_w4.where(1, 2), c_lg.where(4, c_md.where(2, 1)))
    bk = use_wk.where(use_w4.where(256, 128), c_lg.where(64, c_md.where(128, (inf % 64).eq(0).where(64, 32))))
    wk = use_wk.where(use_w4.where(4, 2), 1)
    return _gguf._rdna3_gemm_2d_kernel(p_out, p_raw, p_x, p_grid, m_blks=tok // (wm * tm * 16), n_blks=outf // (wn * tn * 16),
                                       wm=wm, wn=wn, tm=tm, tn=tn, bk=bk, out_features=outf, in_features=inf, ggml_type=qt, wk=wk)
  if kg != 2:
    return _gguf._quant_linear_f16_wmma_kernel(p_out, p_raw, p_x, p_grid, m_blks=mb, m_tiles=mt, out_features=outf, in_features=inf,
                                               ggml_type=qt, kgroups=kg, lanes=ln)
  k_sb = inf // 256
  wk_z, tn_z = ((k_sb >= 4) & (k_sb % 4).eq(0)).where(4, ((k_sb >= 2) & (k_sb % 2).eq(0)).where(2, 1)), (outf >= 32).where(2, 1)
  k_zlds = _gguf._rdna4_zlds_f16_kernel(p_out, p_raw, p_x, p_grid, tok=tok, out_features=outf, in_features=inf,
                                        ggml_type=qt, waves_k=wk_z, tn=tn_z)
  f16_qts = tuple(sorted(_gguf.SMALL_BLOCK_QUANTS))
  k_f16 = _gguf._quant_linear_f16_wmma_kernel(p_out, p_raw, p_x, p_grid, m_blks=mb, m_tiles=mt, out_features=outf, in_features=inf,
                                              ggml_type=qt, kgroups=kg, lanes=ln, qtypes=f16_qts)
  return UOp(Ops.WHERE, src=((inf >= 256) & (inf % 256).eq(0) & (outf >= 16) & (tok >= 16), k_zlds, k_f16))

def _b_moe_route() -> UOp:
  mtot, maxt, ne, nd = _t("m_tot"), _t("max_tiles"), _t("num_experts", 1, 512), _t("ndev", 1, 64)
  return _gguf._amd_moe_route_kernel(_p(0, (mtot,), "i32"), _p(1, (maxt,), "i32"), _p(2, (maxt,), "i32"), _p(3, (maxt,), "i32"),
                                     _p(4, (1,), "i32"), _p(5, (mtot,), "i32"), num_experts=ne, ndev=nd)

def _b_quant_x_frag() -> UOp:
  mt, kt, xr, um, xs = _t("m_tiles"), _t("k_tiles"), _t("x_rows"), _t("use_moe", 0, 1), _t("x_share", 1, 1024)
  tl = _t("tok_len")
  return _gguf._rdna4_quant_x_frag_kernel(_p(0, (mt, kt * 8, 32, 2), "u32"), _p(1, (mt, kt, 16), "f32"), _p(2, (xr, kt * 128), "f16"),
                                          _p(3, (tl,), "i32"), _p(4, (mt,), "i32"), _p(5, (mt,), "i32"), _p(6, (1,), "i32"),
                                          use_moe=um, x_share=xs)

def _b_rdna4_gemm_fp8() -> UOp:
  qt, tok, mt, inf, outf = _t("qtype", 8, 40), _t("tokens"), _t("m_tiles"), _t("in_features", 128), _t("out_features", 64)
  rb, gl, um = _t("raw_bytes", 1, 1 << 30), _t("grid_len", 1, 4096), _t("use_moe", 0, 1)
  tl = _t("tok_len")
  return _gguf._rdna4_gemm_fp8_kernel(
    _p(0, (tok, outf), "f32"), _p(1, (rb,), "u8"), _p(2, (mt, inf // 16, 32, 2), "u32"), _p(3, (mt, inf // 128, 16), "f32"),
    _p(4, (gl,), "u32"), _p(5, (tl,), "i32"), _p(6, (mt,), "i32"), _p(7, (mt,), "i32"), _p(8, (mt,), "i32"), _p(9, (1,), "i32"),
    out_features=outf, in_features=inf, ggml_type=qt, use_moe=um, use_128=tok >= 256, qtypes=tuple(sorted(_gguf.OTF_FP8_QUANTS)))

def _b_fdc() -> UOp:
  B, H, C, db, voff = _t("B", 1, 64), _t("H", 1, 256), _t("C", 1, 64), _t("d_blks", 1, 8), _t("valid_kv_offset", 0)
  return _amd._amd_flash_decode_combine(_p(0, (B, H, 1, db * 64), "f32"), _p(1, (B, H, C, db * 64), "f32"),
                                        _p(2, (B, H, C, 2), "f32"), valid_kv_len=_v(3, "valid_kv_len") + voff)

def _b_fdp() -> UOp:
  B, H_KV, G, db, C, nb = _t("B", 1, 64), _t("H_KV", 1, 256), _t("G", 1, 8), _t("d_blks", 1, 8), _t("C", 1, 64), _t("n_blks", 1, 65536)
  voff, waves, D = _t("valid_kv_offset", 0), _t("waves", 1, 16), db * 64
  args = (_p(0, (B, H_KV * G, C, D), "f32"), _p(1, (B, H_KV * G, C, 2), "f32"),
          _p(2, (B, H_KV * G, 1, D), "f16"), _p(3, (2, B, H_KV, nb * 64, D), "f16"))
  can_gs2 = (G % 2).eq(0) & (waves >= 2)
  gs = (can_gs2 & (db.eq(2) | (db >= 4))).where(2, 1)
  lpt = (db.eq(1) & (waves * G <= 112)).where(8, (db.eq(2) & ((waves // gs) * G <= 112)).where(16, 32))
  return _amd._amd_flash_attention_decode_partial(*args, valid_kv_len=_v(4, "valid_kv_len") + voff, d_blks=db, group=G, waves=waves,
                                                  g_split=gs, lanes_per_tok=lpt)

def _b_fa(rdna4:bool) -> UOp:
  B, H_KV, db, nb, voff = _t("B", 1, 64), _t("H_KV", 1, 256), _t("d_blks", 1, 8), _t("n_blks", 1, 65536), _t("valid_kv_offset", 0)
  BH, mb, qfe, vkl, ql, D = _t("BH", 1, 4096), _t("m_blks", 1, 4096), _t("q_start_from_end", 0, 1), _v(3, "valid_kv_len"), _v(4, "q_len"), db * 64
  args = (_p(0, (BH, mb * 32, D), "f32"), _p(1, (BH, mb * 32, D), "f16"), _p(2, (2, B, H_KV, nb * 32, D), "f16"))
  vkl_e, qs_e = qfe.eq(1).where(vkl + voff, vkl + voff + ql), qfe.eq(1).where(vkl + voff - ql, vkl + voff)
  k_sm = _amd._amd_flash_attention_smallm(*args, valid_kv_len=vkl_e, q_start=qs_e, d_blks=db, rdna4=rdna4)
  k_gen = _amd._amd_flash_attention(*args, valid_kv_len=vkl_e, q_start=qs_e, d_blks=db, rdna4=rdna4)
  return UOp(Ops.WHERE, src=(mb.eq(1) & (db <= 2) & (nb >= 4), k_sm, k_gen))

def _b_hc() -> UOp:
  tok, feats, iters = _t("tokens"), _t("features", 32), _t("iters", 1, 64)
  eps, neps = _t("eps", 0.0, 1.0, dtypes.float32), _t("norm_eps", 0.0, 1.0, dtypes.float32)
  return _amd._hc_coeffs_kernel(_p(0, (tok, 24), "f32"), _p(1, (tok, 24), "f32"), _p(2, (tok, feats), "f32"),
                                _p(3, (24,), "f32"), _p(4, (3,), "f32"), 4, iters, eps, neps)

def _b_gd(arch:str) -> UOp:
  tok, ba, hd, vt, kb, ak = _t("tokens"), _t("batch", 1, 64), _t("heads", 1, 256), _t("v_tiles", 1, 1024), _t("k_blks", 1, 8), _t("alpha_k", 1, 256)
  K = kb * 32
  return _amd._gated_delta_prefill_kernel(_p(0, (ba, hd, tok, vt * 4), "f32"), _p(1, (ba, hd, tok, K), "f32"), _p(2, (ba, hd, tok, K), "f32"),
    _p(3, (ba, hd, tok, vt * 4), "f32"), _p(4, (ba, hd, tok), "f32"), _p(5, (ba, hd, tok, ak), "f32"), _p(6, (ba, hd, vt * 4, K), "f32"),
    k_blks=kb, alpha_k=ak, start_pos=_v(7, "start_pos"), arch=arch)

_BUILDERS: dict[str, Callable[[], UOp]] = {
  **{f"amd_gemv_{w}_f32": functools.partial(_b_gemv, w, "f32") for w in ("f16", "bf16", "f32")},
  "amd_gemv_quant_f32": _b_qgemv,
  "amd_quant_q8_1_f32": functools.partial(_b_quant_q8_1, "f32"),
  "amd_sdot4": _b_sdot4,
  **{f"{a}_gemm_f16": functools.partial(_b_gemm, kg, ln) for a, kg, ln in (("rdna4", 2, 32), ("rdna3", 1, 32), ("cdna", 4, 64))},
  "amd_moe_route": _b_moe_route,
  "rdna4_quant_x_frag": _b_quant_x_frag,
  "rdna4_gemm_fp8": _b_rdna4_gemm_fp8,
  "amd_flash_decode_combine": _b_fdc, "amd_flash_decode_partial": _b_fdp,
  "rdna4_flash_attn": functools.partial(_b_fa, True), "rdna3_flash_attn": functools.partial(_b_fa, False),
  "amd_hc_coeffs": _b_hc,
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
  """Generate all static templated `.uir` files fresh on disk in `tinygrad/llm/kernels/`."""
  return {name: _emit(name, fn)[1] for name, fn in sorted(_BUILDERS.items())}

@functools.cache
def get(name:str) -> UOp|None:
  """Generate `tinygrad/llm/kernels/<name>.uir` fresh on disk and return the parsed templated UOp."""
  if (fn := _BUILDERS.get(name)) is None: return None
  with Context(SPEC=0): return parse_uir(_emit(name, fn)[0].read_text(encoding="utf-8"))

__all__ = ["generate_all", "get"]
