"""Static catalog of templated LLM UIR kernels (no string parsing; finite enum of named kernels)."""
from __future__ import annotations

import functools, os, pathlib, threading
from collections.abc import Callable
from dataclasses import replace

from tinygrad import Context, UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.uop.ops import Ops, ParamArg, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import parse_uir, render_uir

from slopkernels.src import amd as _amd, amd_gguf as _gguf, extra as _extra

_DTYPES: dict[str, DType] = {v.name: v for v in vars(type(dtypes)).values() if isinstance(v, DType)}
_IQ_GRID_LENS = {_gguf.IQ2_XXS: 512, _gguf.IQ2_XS: 1024, _gguf.IQ1_S: 4096, _gguf.IQ2_S: 2048, _gguf.IQ3_XXS: 256, _gguf.IQ3_S: 512}
_WMMA_TBL_LENS = {**_IQ_GRID_LENS, _gguf.IQ4_XS: 256, _gguf.IQ3_XXS: 512, _gguf.IQ2_S: 4096}

def _wdt(t:int) -> DType: return dtypes.uint8 if t == _gguf.MXFP4 else dtypes.uint16 if t in _gguf.HALFWORD_QUANTS else dtypes.uint32
def _p(slot:int, shape:tuple[int|UOp, ...], dt:str|DType) -> UOp: return UOp.param(slot, dt if isinstance(dt, DType) else _DTYPES[dt], shape)
def _v(slot:int, name:str) -> UOp:
  return UOp.variable(name, 0, 1 << 20).replace(arg=ParamArg(slot, dtypes.weakint, (0, 1 << 20), name, AddrSpace.ALU, None))
def _t(name:str, lo:int|float=1, hi:int|float=1 << 20, dt:DType=dtypes.weakint) -> UOp: return UOp.template(name, (lo, hi), dtype=dt)

def _canonical_uir(u:UOp) -> str:
  with Context(TRACK_MATCH_STATS=0, SPEC=0): u = graph_rewrite(u, symbolic_simple, enter_calls=True)
  allocs = [x for x in u.toposort(enter_calls=True) if x.op is Ops.ALLOC]
  if allocs: u = u.substitute({b: b.replace(arg=replace(b.arg, slot=i)) for i, b in enumerate(allocs)}, enter_calls=True)
  return render_uir(u, color=False)

@functools.cache
def _llm_builders() -> dict[str, Callable[[], UOp]]:
  """Build a static map of every named templated LLM kernel to a zero-arg UOp builder."""
  cat: dict[str, Callable[[], UOp]] = {}
  def _regs():
    tok, inf, outf, rl = _t("tokens"), _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
    ch, gr = _t("chunks", 1, 1024), _t("group_rows", 0)
    return tok, inf, outf, rl, ch, gr
  for dt in ("f16", "bf16", "f32"):
    def _q8(dt:str=dt) -> UOp:
      tok, inf, *_ = _regs()
      g = inf // 32
      return _gguf._q8_quantize_kernel(_p(0, (tok, g, 8), "u32"), _p(1, (tok, g), "f32"), _p(2, (tok, g, 2), "f32"),
                                       _p(3, (tok, inf), dt), tok, inf)
    cat[f"amd_q8_quantize_{dt}"] = _q8
  for wdt, xdt in (("f16", "f32"), ("f32", "f16"), ("bf16", "f32"), ("f32", "bf16"), ("f32", "f32")):
    for vc, suf in ((4, ""), (1, "_v1")):
      def _gemv(wdt:str=wdt, xdt:str=xdt, vc:int=vc) -> UOp:
        tok, inf, outf, *_ = _regs()
        return _amd._amd_f16_gemv_kernel(_p(0, (tok, outf), "f32"), _p(1, (outf, inf), wdt), _p(2, (tok, inf), xdt),
                                         in_features=inf, out_features=outf, tokens=tok, val_chunk=vc)
      cat[f"amd_gemv_{wdt}_{xdt}{suf}"] = _gemv
  for qt, qn in _gguf.QUANT_NAMES.items():
    for arch, cdna in (("rdna", False), ("cdna", True)):
      def _qgemv(qt:int=qt, cdna:bool=cdna) -> UOp:
        tok, inf, outf, rl, ch, gr = _regs()
        wd, g = _wdt(qt), inf // 32
        gl = (_p(5, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
        return _gguf._quant_decode_kernel(_p(0, (tok, outf, ch), "f32"), _p(1, (rl,), wd), _p(2, (tok, g, 8), "u32"),
                                          _p(3, (tok, g), "f32"), _p(4, (tok, g, 2), "f32"), *gl,
                                          out_features=outf, in_features=inf, ggml_type=qt, cdna=cdna, group_rows=gr)
      cat[f"{arch}_gemv_{qn}"] = _qgemv
      def _moe(qt:int=qt, cdna:bool=cdna) -> UOp:
        _, inf, outf, rl, ch, _ = _regs()
        wd, g = _wdt(qt), inf // 32
        rows, xrows, xs, nd = _t("rows"), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64)
        mgl = (_p(6, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
        return _gguf._moe_decode_kernel(_p(0, (rows, outf, ch), "f32"), _p(1, (rl,), wd), _p(2, (xrows, g, 8), "u32"),
                                        _p(3, (xrows, g), "f32"), _p(4, (xrows, g, 2), "f32"), _p(5, (rows,), "i32"), *mgl,
                                        out_features=outf, in_features=inf, ggml_type=qt, x_share=xs, cdna=cdna, ndev=nd)
      cat[f"{arch}_moe_{qn}"] = _moe
    for arch, kg, ln in (("rdna4", 2, 32), ("rdna3", 1, 32), ("cdna", 4, 64)):
      def _gemm(qt:int=qt, kg:int=kg, ln:int=ln) -> UOp:
        tok, inf, outf, rl, *_ = _regs()
        tbl = (_p(3, (_WMMA_TBL_LENS[qt],), "u32"),) if qt in _WMMA_TBL_LENS else ()
        return _gguf._quant_linear_f16_wmma_kernel(_p(0, (tok, outf), "f32"), _p(1, (rl,), _wdt(qt)), _p(2, (tok, inf), "f16"), *tbl,
                                                   out_features=outf, in_features=inf, ggml_type=qt, kgroups=kg, lanes=ln)
      cat[f"{arch}_gemm_{qn}_f16"] = _gemm
  def _fdc() -> UOp:
    B, H, C, d_blks, voff = _t("B", 1, 64), _t("H", 1, 256), _t("C", 1, 48), _t("d_blks", 1, 8), _t("valid_kv_offset", 0)
    return _amd._amd_flash_decode_combine(_p(0, (B, H, 1, d_blks * 64), "f32"), _p(1, (B, H, C, d_blks * 64), "f32"),
                                          _p(2, (B, H, C, 2), "f32"), valid_kv_len=_v(3, "valid_kv_len") + voff)
  cat["amd_flash_decode_combine"] = _fdc
  for G in (1, 2, 4, 8):
    for D in (64, 128, 256):
      def _fdp(G:int=G, D:int=D) -> UOp:
        B, H_KV, C, n_blks, voff = _t("B", 1, 64), _t("H_KV", 1, 256), _t("C", 1, 48), _t("n_blks", 1, 65536), _t("valid_kv_offset", 0)
        waves = 16
        while waves * G * ((D + 4) * 2 + 8) > 65536: waves //= 2
        return _amd._amd_flash_attention_decode_partial(_p(0, (B, H_KV * G, C, D), "f32"), _p(1, (B, H_KV * G, C, 2), "f32"),
          _p(2, (B, H_KV * G, 1, D), "f16"), _p(3, (2, B, H_KV, n_blks * 64, D), "f16"),
          valid_kv_len=_v(4, "valid_kv_len") + voff, max_kv_len=n_blks * 64, block_n=64, waves=waves, group=G)
      cat[f"amd_flash_decode_partial_g{G}_d{D}"] = _fdp
  for D in (64, 128, 256):
    for arch, rdna4 in (("rdna4", True), ("rdna3", False)):
      def _fa(D:int=D, rdna4:bool=rdna4) -> UOp:
        B, H_KV, n_blks, voff = _t("B", 1, 64), _t("H_KV", 1, 256), _t("n_blks", 1, 65536), _t("valid_kv_offset", 0)
        BH, m_blks, qfe, vkl, ql = _t("BH", 1, 4096), _t("m_blks", 1, 4096), _t("q_start_from_end", 0, 1), _v(3, "valid_kv_len"), _v(4, "q_len")
        return _amd._amd_flash_attention(_p(0, (BH, m_blks * 32, D), "f32"), _p(1, (BH, m_blks * 32, D), "f16"),
          _p(2, (2, B, H_KV, n_blks * 32, D), "f16"), valid_kv_len=qfe.eq(1).where(vkl + voff, vkl + voff + ql),
          q_start=qfe.eq(1).where(vkl + voff - ql, vkl + voff), rdna4=rdna4)
      cat[f"{arch}_flash_attn_d{D}"] = _fa
  for bdt in ("f16", "f32"):
    for sdt in ("f16", "f32"):
      def _hc(bdt:str=bdt, sdt:str=sdt) -> UOp:
        tok, feats, iters = _t("tokens"), _t("features", 32), _t("iters", 1, 64)
        eps, neps = _t("eps", 0.0, 1.0, dtypes.float32), _t("norm_eps", 0.0, 1.0, dtypes.float32)
        return _extra._hc_coeffs_kernel(_p(0, (tok, 24), "f32"), _p(1, (tok, 24), "f32"), _p(2, (tok, 32), "f32"),
                                        _p(3, (24,), bdt), _p(4, (3,), sdt), 4, iters, eps, neps, feats)
      cat[f"amd_hc_coeffs_{bdt}_{sdt}"] = _hc
  for K in (32, 64, 128, 256):
    for ca, suf in ((False, "a1"), (True, "ak")):
      def _gd(K:int=K, ca:bool=ca) -> UOp:
        tok, ba, hd, vt = _t("tokens"), _t("batch", 1, 64), _t("heads", 1, 256), _t("v_tiles", 1, 1024)
        return _amd._gated_delta_prefill_kernel(_p(0, (ba, hd, tok, vt * 4), "f32"), _p(1, (ba, hd, tok, K), "f32"),
          _p(2, (ba, hd, tok, K), "f32"), _p(3, (ba, hd, tok, vt * 4), "f32"), _p(4, (ba, hd, tok), "f32"),
          _p(5, (ba, hd, tok, K) if ca else (ba, hd, tok), "f32"), _p(6, (ba, hd, vt * 4, K), "f32"),
          _p(7, (ba, hd, tok), "f32"), start_pos=_v(8, "start_pos"))
      cat[f"amd_gated_delta_k{K}_{suf}"] = _gd
  return cat

UIR_DIR = pathlib.Path(__file__).resolve().parent / "uir"

def _atomic_write(dst:pathlib.Path, raw:bytes) -> None:
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_bytes(raw)
  os.replace(tmp, dst)

def generate_all() -> dict[str, str]:
  """Generate all static templated LLM `.uir` files on disk in `slopkernels/uir/` (color-free, always fresh)."""
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  out: dict[str, str] = {}
  with Context(SPEC=0):
    for name, fn in sorted(_llm_builders().items()):
      uir = _canonical_uir(fn())
      _atomic_write(UIR_DIR / f"{name}.uir", (uir + "\n").encode("utf-8"))
      out[name] = uir
  return out

@functools.cache
def get_llm_kernel(name:str) -> UOp|None:
  """Generate `slopkernels/uir/<name>.uir` fresh on disk and return the parsed UOp (always round-tripped through disk)."""
  if (fn := _llm_builders().get(name)) is None: return None
  UIR_DIR.mkdir(parents=True, exist_ok=True)
  with Context(SPEC=0):
    _atomic_write(p := UIR_DIR / f"{name}.uir", (_canonical_uir(fn()) + "\n").encode("utf-8"))
    return parse_uir(p.read_text(encoding="utf-8"))
