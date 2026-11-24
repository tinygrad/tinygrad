"""End-to-end LLM kernel benchmark suite for slopkernels on AMD GPUs (e.g. 7900XTX, 9060XT).

Benchmarks the exact logical op UIRs that `tinygrad.llm` dispatches (`quant_linear` GEMV & GEMM
across GGUF encodings, `moe_linear` across GGUF encodings, `dense_linear` GEMV, `flash_attention_op`,
`gated_delta_prefill_op`, `hc_coeffs_op`) using hardware GPU timestamps and L2 cache flushes.

Usage:
  DEV=AMD python3 -m slopkernels.bench
  DEV=AMD python3 -m slopkernels.bench --filter quant_gemv --iters 10
  DEV="MOCK+AMD" python3 -m slopkernels.bench --smoke --warmup 1 --iters 1
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

if str(root := pathlib.Path(__file__).resolve().parent.parent) not in sys.path: sys.path.insert(0, str(root))

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.dtype import DType
from tinygrad.engine.realize import ExecContext, compile_linear, link_linear, pm_exec
from tinygrad.helpers import unwrap
from tinygrad.uop.uir import parse_uir

import slopkernels

# GGUF quant type metadata: (ggml_type, block_bytes per 256 weights, packed word dtype)
_QUANT_META: dict[str, tuple[int, int, DType]] = {
  "Q8_0": (8, 272, dtypes.uint16),
  "Q6_K": (14, 210, dtypes.uint16),
  "Q5_K": (13, 176, dtypes.uint32),
  "Q4_K": (12, 144, dtypes.uint32),
  "Q3_K": (11, 110, dtypes.uint16),
  "Q2_K": (10, 84, dtypes.uint16),
  "IQ4_XS": (23, 136, dtypes.uint32),
  "IQ4_NL": (20, 144, dtypes.uint16),
  "IQ3_S": (21, 110, dtypes.uint16),
  "IQ3_XXS": (18, 98, dtypes.uint16),
  "IQ2_S": (22, 82, dtypes.uint16),
  "IQ2_XS": (17, 74, dtypes.uint16),
  "IQ2_XXS": (16, 66, dtypes.uint16),
  "IQ1_S": (19, 50, dtypes.uint16),
  "MXFP4": (39, 136, dtypes.uint8),
}

@dataclass(frozen=True)
class BenchResult:
  name: str
  median_us: float
  min_us: float
  gb_s: float
  tflops: float

def _desc(s:Tensor|UOp) -> tuple:
  if isinstance(s, Tensor):
    return (tuple(int(d) for d in s.shape), tuple(int(d) for d in s.uop.max_shard_shape), s.uop.axis, s.dtype.name)
  return (int((u:=s.unbind_all()[0]).vmin), int(u.vmax))

def _run_op(uir:str, *srcs:Tensor|UOp) -> Tensor:
  return Tensor(parse_uir(uir).call_with_output(*(s.uop if isinstance(s, Tensor) else s for s in srcs)))

def _slop_call(name:str, target:tuple[int, ...], dev:str, *srcs:Tensor|UOp, **attrs:object) -> Callable[[], Tensor]:
  uir = slopkernels.get(name, target=target, device=dev, srcs=tuple(_desc(s) for s in srcs), **attrs)
  assert uir is not None, f"slopkernels.get({name!r}) returned None"
  return lambda: _run_op(uir, *srcs)

def _arch_info(dev:str) -> tuple[tuple[int, ...], bool]:
  raw = getattr(Device[dev], "target", None)
  target = tuple(int(x) for x in raw) if isinstance(raw, tuple) else (11, 0, 0)
  return target, target[0] == 9

def _packed_words(rows:int, cols:int, qname:str) -> tuple[int, int, DType]:
  qtype, blk_bytes, w_dt = _QUANT_META[qname]
  return qtype, (rows * cols * blk_bytes // 256) // w_dt.itemsize, w_dt

def _filled_packed(w_len:int, w_dt:DType) -> Tensor:
  # Valid non-zero packed pattern across all GGUF formats (f16 0x2111 ≈ 0.0088, MXFP4 e8m0 0x7e = 0.5) to avoid zero-page compression.
  val = 0x7e if w_dt == dtypes.uint8 else 0x2111 if w_dt == dtypes.uint16 else 0x21112111
  return Tensor.full((w_len,), val, dtype=w_dt).contiguous().realize()

def _make_cache_flusher(flush_mb:int) -> Callable[[], None]:
  """Build a compiled read-only GPU kernel that streams `flush_mb` MB through L1/L2/Infinity Cache (MALL)
  to evict cached weights without leaving dirty cache lines that would trigger writebacks during the timed kernel."""
  if flush_mb <= 0: return lambda: None
  n = flush_mb * 1024 * 1024 // 4
  src = Tensor.ones(n, dtype=dtypes.float32).contiguous().realize()
  sink = src.sum()
  linked = link_linear(compile_linear(sink.schedule_linear(), beam=0, profile=False, cache=True))
  ctx = ExecContext({}, update_stats=False, wait=True, cache=True)
  def _flush() -> None:
    for raw in linked.src: pm_exec.rewrite(raw.without_after, ctx)
  _flush()
  return _flush

def _time_fn(fn:Callable[[], Tensor], warmup:int, iters:int, flush_fn:Callable[[], None]) -> tuple[float, float]:
  out = fn()
  linked = link_linear(compile_linear(out.schedule_linear(), beam=0, profile=True, cache=True))
  ctx = ExecContext({}, update_stats=False, wait=True, cache=True)
  def _run_once() -> float:
    return sum(sum(map(unwrap, pm_exec.rewrite(c, ctx)[c.arg.hcq is not None:])) for raw in linked.src for c in (raw.without_after,))
  for _ in range(max(1, warmup)): _run_once()
  first = float(out.flatten()[:1].float().item())
  if not math.isfinite(first): raise RuntimeError(f"non-finite benchmark output: {first}")
  times_us: list[float] = []
  for _ in range(iters):
    flush_fn()
    times_us.append(_run_once() * 1e6)
  return statistics.median(times_us), min(times_us)

def build_benchmarks(dev:str, smoke:bool=False, fast:bool=False) -> list[tuple[str, int, int, Callable[[], Callable[[], Tensor]]]]:
  """Return list of (name, io_bytes, flops, setup_fn) matching how `tinygrad.llm` uses `slopkernels`."""
  target, cdna = _arch_info(dev)
  N, K, FFN = (128, 256, 256) if smoke else (4096, 4096, 14336)
  M_PREFILL = 16 if smoke else 128
  M_PREFILL_LG = 32 if smoke else 512
  E, TOPK, MOE_N, MOE_K = (8, 4, 128, 256) if smoke else (64, 8, 2048, 4096)
  ATTN_H, ATTN_HKV, ATTN_D, ATTN_N, ATTN_M = (4, 2, 64, 128, 32) if smoke else (32, 8, 128, 2048, 256)
  ATTN_N_LONG = 256 if smoke else 8192
  cases: list[tuple[str, int, int, Callable[[], Callable[[], Tensor]]]] = []

  # 1. Quantized GGUF GEMV (decode M=1, M=4 via quant_linear: q8_quantize + quant_decode + split-K reduce)
  gemv_quants = ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if smoke else (
    ("Q8_0", "Q6_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_S", "IQ1_S", "MXFP4") if fast else
    ("Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ1_S", "MXFP4"))
  for qname in gemv_quants:
    qtype, w_len, w_dt = _packed_words(N, K, qname)
    def _mk_qgemv(qtype:int=qtype, w_len:int=w_len, w_dt:DType=w_dt, m:int=1, n:int=N) -> Callable[[], Tensor]:
      return _slop_call("quant_linear", target, dev, _filled_packed(w_len, w_dt), Tensor.ones(m, K, dtype=dtypes.float32).realize(),
                        ggml_type=qtype, in_features=K, out_features=n, groups=1)
    cases.append((f"quant_gemv_{qname}_1x{N}x{K}", w_len * w_dt.itemsize + (K + N) * 4, 2 * N * K, _mk_qgemv))

  if not smoke:
    extra_gemvs = (("Q4_K", 1, FFN), ("Q8_0", 1, FFN), ("Q4_K", 4, N)) if fast else (
      ("Q4_K", 1, FFN), ("Q8_0", 1, FFN), ("MXFP4", 1, FFN), ("Q4_K", 4, N))
    for qname, m, n in extra_gemvs:
      qtype, w_len, w_dt = _packed_words(n, K, qname)
      def _mk_qgemv_lg(qtype:int=qtype, w_len:int=w_len, w_dt:DType=w_dt, m:int=m, n:int=n) -> Callable[[], Tensor]:
        return _slop_call("quant_linear", target, dev, _filled_packed(w_len, w_dt), Tensor.ones(m, K, dtype=dtypes.float32).realize(),
                          ggml_type=qtype, in_features=K, out_features=n, groups=1)
      cases.append((f"quant_gemv_{qname}_{m}x{n}x{K}", w_len * w_dt.itemsize + m * (K + n) * 4, 2 * m * n * K, _mk_qgemv_lg))

  # 2. Quantized GGUF GEMM (prefill M=128, M=512 via quant_linear WMMA across GGUF encodings)
  gemm_quants = ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if (smoke or fast) else (
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_S", "IQ2_S", "MXFP4")
  for qname in gemm_quants:
    qtype, w_len, w_dt = _packed_words(N, K, qname)
    def _mk_qgemm(qtype:int=qtype, w_len:int=w_len, w_dt:DType=w_dt, m:int=M_PREFILL) -> Callable[[], Tensor]:
      return _slop_call("quant_linear", target, dev, _filled_packed(w_len, w_dt), Tensor.ones(m, K, dtype=dtypes.float16).realize(),
                        ggml_type=qtype, in_features=K, out_features=N, groups=1)
    cases.append((f"quant_gemm_{qname}_{M_PREFILL}x{N}x{K}",
                  w_len * w_dt.itemsize + M_PREFILL * K * 2 + M_PREFILL * N * 4, 2 * M_PREFILL * N * K, _mk_qgemm))

  if not smoke:
    for qname in (("Q4_K", "IQ4_XS") if fast else ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4")):
      qtype, w_len, w_dt = _packed_words(N, K, qname)
      def _mk_qgemm_lg(qtype:int=qtype, w_len:int=w_len, w_dt:DType=w_dt) -> Callable[[], Tensor]:
        return _slop_call("quant_linear", target, dev, _filled_packed(w_len, w_dt), Tensor.ones(M_PREFILL_LG, K, dtype=dtypes.float16).realize(),
                          ggml_type=qtype, in_features=K, out_features=N, groups=1)
      cases.append((f"quant_gemm_{qname}_{M_PREFILL_LG}x{N}x{K}",
                    w_len * w_dt.itemsize + M_PREFILL_LG * K * 2 + M_PREFILL_LG * N * 4, 2 * M_PREFILL_LG * N * K, _mk_qgemm_lg))

  # 3. Quantized GGUF MoE GEMV (via moe_linear: q8_quantize + moe_decode + split-K reduce)
  moe_quants = ("Q4_K",) if smoke else (("Q8_0", "Q4_K", "IQ4_XS") if fast else ("Q8_0", "Q4_K", "IQ4_XS", "IQ1_S", "MXFP4"))
  for qname in moe_quants:
    qtype, per_e_words, w_dt = _packed_words(MOE_N, MOE_K, qname)
    def _mk_moe(qtype:int=qtype, raw_len:int=E * per_e_words, w_dt:DType=w_dt) -> Callable[[], Tensor]:
      return _slop_call("moe_linear", target, dev, _filled_packed(raw_len, w_dt), Tensor(list(range(TOPK)), dtype=dtypes.int32).realize(),
                        Tensor.ones(1, 1, 1, MOE_K, dtype=dtypes.float32).realize(), ggml_type=qtype, sel_shape=(1, 1, TOPK),
                        in_features=MOE_K, local_out=MOE_N, global_out=MOE_N)
    cases.append((f"moe_gemv_{qname}_E{E}k{TOPK}_{MOE_N}x{MOE_K}", TOPK * per_e_words * w_dt.itemsize, 2 * TOPK * MOE_N * MOE_K, _mk_moe))

  # 4. Dense FP16 GEMV (via dense_linear, decode M=1, M=4)
  for m, n in (((1, FFN),) if fast else ((1, N), (1, FFN), (4, N))):
    def _mk_dense(m:int=m, n:int=n) -> Callable[[], Tensor]:
      return _slop_call("dense_linear", target, dev, Tensor.ones(n, K, dtype=dtypes.float16).realize(),
                        Tensor.ones(m, K, dtype=dtypes.float32).realize())
    cases.append((f"dense_gemv_f16_{m}x{n}x{K}", n * K * 2 + m * (K + n) * 4, 2 * m * n * K, _mk_dense))

  # 5. Flash Attention (prefill + 2-pass decode via flash_attention_op)
  if not cdna:
    def _mk_fa_prefill() -> Callable[[], Tensor]:
      return _slop_call("flash_attention_op", target, dev, Tensor.ones(1, ATTN_H, ATTN_M, ATTN_D, dtype=dtypes.float16).realize(),
                        Tensor.ones(2, 1, ATTN_HKV, ATTN_N, ATTN_D, dtype=dtypes.float16).realize(), valid_kv_len=ATTN_N, decode=False)
    fa_flops = 4 * ATTN_H * ATTN_M * ATTN_N * ATTN_D
    fa_bytes = (ATTN_H * ATTN_M * ATTN_D + 2 * ATTN_HKV * ATTN_N * ATTN_D) * 2 + ATTN_H * ATTN_M * ATTN_D * 4
    cases.append((f"flash_attn_prefill_H{ATTN_H}_M{ATTN_M}_N{ATTN_N}_D{ATTN_D}", fa_bytes, fa_flops, _mk_fa_prefill))

  for ctx in ((ATTN_N,) if smoke else (ATTN_N, ATTN_N_LONG)):
    def _mk_fa_dec(ctx:int=ctx) -> Callable[[], Tensor]:
      return _slop_call("flash_attention_op", target, dev, Tensor.ones(1, ATTN_H, 1, ATTN_D, dtype=dtypes.float16).realize(),
                        Tensor.ones(2, 1, ATTN_HKV, ctx, ATTN_D, dtype=dtypes.float16).realize(), valid_kv_len=ctx, decode=True)
    dec_bytes = 2 * ATTN_HKV * ctx * ATTN_D * 2 + ATTN_H * ATTN_D * 6
    cases.append((f"flash_attn_decode_H{ATTN_H}_N{ctx}_D{ATTN_D}", dec_bytes, 4 * ATTN_H * ctx * ATTN_D, _mk_fa_dec))

  # 6. Gated Delta Net Prefill & mHC Coeffs
  gd_h, gd_t, gd_d = (2, 8, 32) if smoke else (16, 64, 128)
  def _mk_gd() -> Callable[[], Tensor]:
    return _slop_call("gated_delta_prefill_op", target, dev,
                      Tensor.ones(1, gd_h, gd_t, gd_d, dtype=dtypes.float32).realize(),
                      Tensor.ones(1, gd_h, gd_t, gd_d, dtype=dtypes.float32).realize(),
                      Tensor.ones(1, gd_h, gd_t, gd_d, dtype=dtypes.float32).realize(),
                      Tensor.zeros(1, gd_h, gd_t, dtype=dtypes.float32).realize(),
                      Tensor.ones(1, gd_h, gd_t, dtype=dtypes.float32).realize(),
                      Tensor.zeros(1, gd_h, gd_d, gd_d, dtype=dtypes.float32).realize())
  cases.append((f"gated_delta_prefill_H{gd_h}_T{gd_t}_D{gd_d}", (4 * gd_h * gd_t * gd_d + 2 * gd_h * gd_d * gd_d) * 4,
                6 * gd_h * gd_t * gd_d * gd_d, _mk_gd))

  def _mk_hc() -> Callable[[], Tensor]:
    t_h = 2 if smoke else 32
    return _slop_call("hc_coeffs_op", target, dev, Tensor.ones(t_h, 24).realize(), Tensor.ones(t_h, K).realize(),
                      Tensor.zeros(24, dtype=dtypes.float16).realize(), Tensor.ones(3, dtype=dtypes.float16).realize(),
                      count=4, iters=2, eps=1e-6, norm_eps=1e-6)
  cases.append((f"hc_coeffs_{2 if smoke else 32}x{K}", (2 if smoke else 32) * K * 4, 0, _mk_hc))

  return cases

def main() -> None:
  ap = argparse.ArgumentParser(description="Benchmark end-to-end LLM slopkernels via UIR on AMD GPUs")
  ap.add_argument("--warmup", type=int, default=2, help="Warmup iterations per op")
  ap.add_argument("--iters", type=int, default=5, help="Timed iterations per op")
  ap.add_argument("--filter", type=str, default="", help="Regex filter on benchmark name")
  ap.add_argument("--smoke", action="store_true", help="Use small shapes for fast emulator verification")
  ap.add_argument("--fast", action="store_true", help="Representative ~25-op LLM subset for <90s CI budget")
  ap.add_argument("--flush-mb", type=int, default=256, help="MB of clean read-only VRAM traffic per iter to evict L2/Infinity Cache (0=disable)")
  ap.add_argument("--json", type=str, default="", help="Optional path to write JSON results")
  args = ap.parse_args()

  dev = Device.DEFAULT
  target = getattr(Device[dev], "target", None)
  flush_mb = min(args.flush_mb, 2) if args.smoke else args.flush_mb
  flush_fn = _make_cache_flusher(flush_mb)
  t_start = time.perf_counter()
  print(f"slopkernels LLM benchmark | DEV={dev} | target={target} | warmup={args.warmup} iters={args.iters} flush={flush_mb}MB smoke={args.smoke}")
  print(f"{'Op':<46} {'Median (us)':>12} {'Min (us)':>10} {'GB/s':>9} {'TFLOPS':>9}")
  print("-" * 90)

  pat = re.compile(args.filter) if args.filter else None
  results: list[BenchResult] = []
  for name, io_bytes, flops, setup_fn in build_benchmarks(dev, smoke=args.smoke, fast=args.fast):
    if pat is not None and not pat.search(name): continue
    step_fn = setup_fn()
    med_us, min_us = _time_fn(step_fn, args.warmup, args.iters, flush_fn)
    gb_s = (io_bytes / 1e9) / (med_us / 1e6) if io_bytes > 0 and med_us > 0 else 0.0
    tflops = (flops / 1e12) / (med_us / 1e6) if flops > 0 and med_us > 0 else 0.0
    r = BenchResult(name=name, median_us=med_us, min_us=min_us, gb_s=gb_s, tflops=tflops)
    results.append(r)
    print(f"{r.name:<46} {r.median_us:>12.1f} {r.min_us:>10.1f} {r.gb_s:>9.1f} {r.tflops:>9.2f}")

  print("-" * 90)
  print(f"Benchmarked {len(results)} LLM ops in {time.perf_counter() - t_start:.1f}s.")
  if args.json:
    with open(args.json, "w") as f:
      json.dump([r.__dict__ for r in results], f, indent=2)

if __name__ == "__main__":
  main()
