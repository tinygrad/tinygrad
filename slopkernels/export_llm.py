"""Offline exporter for templated LLM UIR kernels (`tinygrad/llm/kernels/*.uir`)."""
from __future__ import annotations

import os, pathlib, threading
from dataclasses import replace

from tinygrad import Context, UOp, dtypes
from tinygrad.dtype import AddrSpace, DType
from tinygrad.uop.ops import Ops, ParamArg, graph_rewrite
from tinygrad.uop.symbolic import symbolic_simple
from tinygrad.uop.uir import render_uir

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

def llm_kernel_catalog() -> dict[str, UOp]:
  """Build all templated LLM kernel UOps (named like `rdna4_gemm_iq4_xs_f16`)."""
  cat: dict[str, UOp] = {}
  tok, inf, outf, rl = _t("tokens"), _t("in_features", 32), _t("out_features"), _t("raw_len", 1, 1 << 30)
  ch, gr = _t("chunks", 1, 1024), _t("group_rows", 0)
  for dt in ("f16", "bf16", "f32"):
    g = inf // 32
    cat[f"amd_q8_quantize_{dt}"] = _gguf._q8_quantize_kernel(_p(0, (tok, g, 8), "u32"), _p(1, (tok, g), "f32"),
                                                             _p(2, (tok, g, 2), "f32"), _p(3, (tok, inf), dt), tok, inf)
  for wdt, xdt in (("f16", "f32"), ("f32", "f16"), ("bf16", "f32"), ("f32", "bf16"), ("f32", "f32")):
    for vc, suf in ((4, ""), (1, "_v1")):
      cat[f"amd_gemv_{wdt}_{xdt}{suf}"] = _amd._amd_f16_gemv_kernel(_p(0, (tok, outf), "f32"), _p(1, (outf, inf), wdt), _p(2, (tok, inf), xdt),
                                                                    in_features=inf, out_features=outf, tokens=tok, val_chunk=vc)
  for qt, qn in _gguf.QUANT_NAMES.items():
    wd, g = _wdt(qt), inf // 32
    gl = (_p(5, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
    for arch, cdna in (("rdna", False), ("cdna", True)):
      cat[f"{arch}_gemv_{qn}"] = _gguf._quant_decode_kernel(_p(0, (tok, outf, ch), "f32"), _p(1, (rl,), wd), _p(2, (tok, g, 8), "u32"),
                                                            _p(3, (tok, g), "f32"), _p(4, (tok, g, 2), "f32"), *gl,
                                                            out_features=outf, in_features=inf, ggml_type=qt, cdna=cdna, group_rows=gr)
    tbl = (_p(3, (_WMMA_TBL_LENS[qt],), "u32"),) if qt in _WMMA_TBL_LENS else ()
    for arch, kg, ln in (("rdna4", 2, 32), ("rdna3", 1, 32), ("cdna", 4, 64)):
      cat[f"{arch}_gemm_{qn}_f16"] = _gguf._quant_linear_f16_wmma_kernel(_p(0, (tok, outf), "f32"), _p(1, (rl,), wd), _p(2, (tok, inf), "f16"), *tbl,
                                                                         out_features=outf, in_features=inf, ggml_type=qt, kgroups=kg, lanes=ln)
    rows, xrows, xs, nd = _t("rows"), _t("x_rows"), _t("x_share", 1, 1024), _t("ndev", 1, 64)
    mgl = (_p(6, (_IQ_GRID_LENS[qt],), "u32"),) if qt in _IQ_GRID_LENS else ()
    for arch, cdna in (("rdna", False), ("cdna", True)):
      cat[f"{arch}_moe_{qn}"] = _gguf._moe_decode_kernel(_p(0, (rows, outf, ch), "f32"), _p(1, (rl,), wd), _p(2, (xrows, g, 8), "u32"),
                                                         _p(3, (xrows, g), "f32"), _p(4, (xrows, g, 2), "f32"), _p(5, (rows,), "i32"), *mgl,
                                                         out_features=outf, in_features=inf, ggml_type=qt, x_share=xs, cdna=cdna, ndev=nd)
  B, H, H_KV, C = _t("B", 1, 64), _t("H", 1, 256), _t("H_KV", 1, 256), _t("C", 1, 48)
  n_blks, d_blks, voff = _t("n_blks", 1, 65536), _t("d_blks", 1, 8), _t("valid_kv_offset", 0)
  cat["amd_flash_decode_combine"] = _amd._amd_flash_decode_combine(_p(0, (B, H, 1, d_blks * 64), "f32"), _p(1, (B, H, C, d_blks * 64), "f32"),
                                                                   _p(2, (B, H, C, 2), "f32"), valid_kv_len=_v(3, "valid_kv_len") + voff)
  for G in (1, 2, 4, 8):
    for D in (64, 128, 256):
      waves = 16
      while waves * G * ((D + 4) * 2 + 8) > 65536: waves //= 2
      cat[f"amd_flash_decode_partial_g{G}_d{D}"] = _amd._amd_flash_attention_decode_partial(
        _p(0, (B, H_KV * G, C, D), "f32"), _p(1, (B, H_KV * G, C, 2), "f32"), _p(2, (B, H_KV * G, 1, D), "f16"),
        _p(3, (2, B, H_KV, n_blks * 64, D), "f16"), valid_kv_len=_v(4, "valid_kv_len") + voff, max_kv_len=n_blks * 64, block_n=64, waves=waves,
        group=G)
  BH, m_blks, qfe = _t("BH", 1, 4096), _t("m_blks", 1, 4096), _t("q_start_from_end", 0, 1)
  vkl, ql = _v(3, "valid_kv_len"), _v(4, "q_len")
  for D in (64, 128, 256):
    for arch, rdna4 in (("rdna4", True), ("rdna3", False)):
      cat[f"{arch}_flash_attn_d{D}"] = _amd._amd_flash_attention(
        _p(0, (BH, m_blks * 32, D), "f32"), _p(1, (BH, m_blks * 32, D), "f16"), _p(2, (2, B, H_KV, n_blks * 32, D), "f16"),
        valid_kv_len=qfe.eq(1).where(vkl + voff, vkl + voff + ql), q_start=qfe.eq(1).where(vkl + voff - ql, vkl + voff), rdna4=rdna4)
  feats, iters, eps, neps = _t("features", 32), _t("iters", 1, 64), _t("eps", 0.0, 1.0, dtypes.float32), _t("norm_eps", 0.0, 1.0, dtypes.float32)
  for bdt in ("f16", "f32"):
    for sdt in ("f16", "f32"):
      cat[f"amd_hc_coeffs_{bdt}_{sdt}"] = _extra._hc_coeffs_kernel(_p(0, (tok, 24), "f32"), _p(1, (tok, 24), "f32"), _p(2, (tok, 32), "f32"),
                                                                   _p(3, (24,), bdt), _p(4, (3,), sdt), 4, iters, eps, neps, feats)
  ba, hd, vt = _t("batch", 1, 64), _t("heads", 1, 256), _t("v_tiles", 1, 1024)
  for K in (32, 64, 128, 256):
    for ca, suf in ((False, "a1"), (True, "ak")):
      cat[f"amd_gated_delta_k{K}_{suf}"] = _amd._gated_delta_prefill_kernel(
        _p(0, (ba, hd, tok, vt * 4), "f32"), _p(1, (ba, hd, tok, K), "f32"), _p(2, (ba, hd, tok, K), "f32"), _p(3, (ba, hd, tok, vt * 4), "f32"),
        _p(4, (ba, hd, tok), "f32"), _p(5, (ba, hd, tok, K) if ca else (ba, hd, tok), "f32"), _p(6, (ba, hd, vt * 4, K), "f32"),
        _p(7, (ba, hd, tok), "f32"), start_pos=_v(8, "start_pos"))
  return cat

def _atomic_write(dst:pathlib.Path, raw:bytes) -> None:
  tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
  tmp.write_bytes(raw)
  os.replace(tmp, dst)

DEFAULT_LLM_KERNELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "tinygrad" / "llm" / "kernels"

def generate_all() -> None:
  """Generate all templated LLM `.uir` files on disk in `tinygrad/llm/kernels/` (race-safe, color-free)."""
  out_dir = DEFAULT_LLM_KERNELS_DIR
  out_dir.mkdir(parents=True, exist_ok=True)
  def _run() -> None:
    with Context(SPEC=0):
      for name, u in llm_kernel_catalog().items():
        raw = (_canonical_uir(u) + "\n").encode("utf-8")
        if not (dst := out_dir / f"{name}.uir").is_file() or dst.read_bytes() != raw: _atomic_write(dst, raw)
  try:
    import fcntl
    with open(out_dir / ".lock", "w") as lf:
      fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
      _run()
  except ImportError: _run()
