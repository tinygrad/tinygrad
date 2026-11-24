"""End-to-end LLM kernel benchmark & correctness suite for slopkernels on AMD GPUs (e.g. 7900XTX, 9060XT).

Benchmarks and numerically verifies the exact logical op UIRs that `tinygrad.llm` dispatches
(`quant_linear` GEMV & GEMM across GGUF encodings, `moe_linear` across GGUF encodings,
`dense_linear` GEMV, `flash_attention_op`, `gated_delta_prefill_op`, `hc_coeffs_op`) using
hardware GPU timestamps and clean read-only L2/Infinity Cache flushes.

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

import numpy as np

if str(root := pathlib.Path(__file__).resolve().parent.parent) not in sys.path: sys.path.insert(0, str(root))

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.dtype import DType
from tinygrad.engine.realize import ExecContext, compile_linear, link_linear, pm_exec
from tinygrad.helpers import prod, unwrap
from tinygrad.llm.gguf import ggml_data_to_tensor
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

VerifyFn = Callable[[Tensor], None]
SetupFn = Callable[[], tuple[Callable[[], Tensor], VerifyFn]]

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

def _nonuniform(shape:tuple[int, ...], dtype:DType, dev:str, scale:float=0.5, freq:float=0.017, phase:float=0.3) -> Tensor:
  t = ((Tensor.arange(prod(shape), dtype=dtypes.float32) + Tensor.zeros(1, dtype=dtypes.float32, device=dev)) * freq + phase).sin() * scale
  return t.cast(dtype).reshape(shape).contiguous().realize()

def _make_packed(rows:int, cols:int, qname:str, dev:str, seed:int=7) -> tuple[Tensor, Tensor]:
  """Generate non-uniform valid packed weights for any GGUF quant by tiling 16 distinct valid rows on GPU.
  Returns `(packed_words, first_tile_u8)` where `first_tile_u8` has shape `(tile_rows, row_bytes)` for fast reference checks."""
  qtype, blk_bytes, w_dt = _QUANT_META[qname]
  row_bytes = cols * blk_bytes // 256
  tile_rows = min(16, rows)
  rng = np.random.default_rng(seed + qtype)
  if qtype == 39:  # MXFP4: 17 bytes per 32 weights -> byte 0 is e8m0 exponent, bytes 1..16 are packed fp4 nibbles
    blocks = rng.integers(0, 256, size=(tile_rows, cols // 32, 17), dtype=np.uint8)
    blocks[:, :, 0] = rng.integers(0x7b, 0x80, size=(tile_rows, cols // 32), dtype=np.uint8)
    tile_np = blocks.reshape(tile_rows, row_bytes)
  else:
    # Bytes in [0x20, 0x37] keep every little-endian f16 header scale finite in ~[0.008, 0.49] across all GGUF block layouts
    tile_np = rng.integers(0x20, 0x38, size=(tile_rows, row_bytes), dtype=np.uint8)
  tile_u8 = Tensor(tile_np, device=dev).realize()
  np_dt = {dtypes.uint8: np.uint8, dtypes.uint16: np.uint16, dtypes.uint32: np.uint32}[w_dt]
  tile_w = Tensor(np.ascontiguousarray(tile_np).view(np_dt), device=dev).realize()
  reps = rows // tile_rows
  row_words = row_bytes // w_dt.itemsize
  if reps == 1: packed = tile_w.flatten().contiguous().realize()
  else: packed = tile_w.reshape(1, tile_rows, row_words).expand(reps, tile_rows, row_words).reshape(-1).contiguous().realize()
  return packed, tile_u8

def _assert_close(got:np.ndarray, ref:np.ndarray, rtol:float, atol:float) -> None:
  scale_atol = max(atol, rtol * float(np.max(np.abs(ref))) if ref.size else atol)
  np.testing.assert_allclose(got, ref, rtol=rtol, atol=scale_atol)

def _make_cache_flusher(flush_mb:int) -> Callable[[], None]:
  """Build a compiled read-only GPU kernel that streams `flush_mb` MB through L1/L2/Infinity Cache (MALL)
  to evict cached weights without leaving dirty cache lines that would trigger writebacks during the timed kernel."""
  if flush_mb <= 0: return lambda: None
  n = flush_mb * 1024 * 1024 // 4
  src = Tensor.ones(n, dtype=dtypes.float32).contiguous().realize()
  sink = src.sum()
  Device[Device.DEFAULT].synchronize()
  linked = link_linear(compile_linear(sink.schedule_linear(), beam=0, profile=False, cache=True))
  ctx = ExecContext({}, update_stats=False, wait=True, cache=True)
  keep = (src, sink, linked)
  def _flush() -> None:
    _ = keep
    for raw in linked.src: pm_exec.rewrite(raw.without_after, ctx)
  _flush()
  return _flush

def _time_fn(fn:Callable[[], Tensor], verify:VerifyFn|None, warmup:int, iters:int, flush_fn:Callable[[], None]) -> tuple[float, float]:
  Device[Device.DEFAULT].synchronize()
  out = fn()
  linked = link_linear(compile_linear(out.schedule_linear(), beam=0, profile=True, cache=True))
  ctx = ExecContext({}, update_stats=False, wait=True, cache=True)
  def _run_once() -> float:
    return sum(sum(map(unwrap, pm_exec.rewrite(c, ctx)[c.arg.hcq is not None:])) for raw in linked.src for c in (raw.without_after,))
  _run_once()
  if verify is not None: verify(out)
  else:
    first = float(out.flatten()[:1].float().item())
    if not math.isfinite(first): raise RuntimeError(f"non-finite benchmark output: {first}")
  for _ in range(max(0, warmup - 1)): _run_once()
  times_us: list[float] = []
  for _ in range(iters):
    flush_fn()
    times_us.append(_run_once() * 1e6)
  return statistics.median(times_us), min(times_us)

def build_benchmarks(dev:str, smoke:bool=False, fast:bool=False) -> list[tuple[str, int, int, SetupFn]]:
  """Return list of (name, io_bytes, flops, setup_fn) matching how `tinygrad.llm` uses `slopkernels`."""
  target, cdna = _arch_info(dev)
  N, K, FFN = (128, 256, 256) if smoke else (4096, 4096, 14336)
  M_PREFILL = 16 if smoke else 128
  M_PREFILL_LG = 32 if smoke else 512
  E, TOPK, MOE_N, MOE_K = (8, 4, 128, 256) if smoke else (64, 8, 2048, 4096)
  ATTN_H, ATTN_HKV, ATTN_D, ATTN_N, ATTN_M = (4, 2, 64, 128, 32) if smoke else (32, 8, 128, 2048, 256)
  ATTN_N_LONG = 256 if smoke else 8192
  CHECK_ROWS = 4
  cases: list[tuple[str, int, int, SetupFn]] = []

  def _make_qlinear_setup(qname:str, m:int, n:int, x_dt:DType) -> SetupFn:
    qtype = _QUANT_META[qname][0]
    def _setup() -> tuple[Callable[[], Tensor], VerifyFn]:
      w_packed, tile_u8 = _make_packed(n, K, qname, dev)
      x = _nonuniform((m, K), x_dt, dev, scale=0.5, freq=0.013)
      def _verify(out:Tensor) -> None:
        w4 = ggml_data_to_tensor(tile_u8[:CHECK_ROWS].flatten(), CHECK_ROWS * K, qtype).reshape(CHECK_ROWS, K)
        ref = (x.float() @ w4.T).numpy()
        _assert_close(out[:, :CHECK_ROWS].float().numpy(), ref, rtol=4e-2, atol=5e-2)
      return _slop_call("quant_linear", target, dev, w_packed, x, ggml_type=qtype, in_features=K, out_features=n, groups=1), _verify
    return _setup

  # 1. Quantized GGUF GEMV (decode M=1, M=4 via quant_linear: q8_quantize + quant_decode + split-K reduce)
  gemv_quants = ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if smoke else (
    ("Q8_0", "Q6_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_S", "IQ1_S", "MXFP4") if fast else
    ("Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ1_S", "MXFP4"))
  for qname in gemv_quants:
    _, w_len, w_dt = _packed_words(N, K, qname)
    cases.append((f"quant_gemv_{qname}_1x{N}x{K}", w_len * w_dt.itemsize + (K + N) * 4, 2 * N * K,
                  _make_qlinear_setup(qname, 1, N, dtypes.float32)))

  if not smoke:
    extra_gemvs = (("Q4_K", 1, FFN), ("Q8_0", 1, FFN), ("Q4_K", 4, N)) if fast else (
      ("Q4_K", 1, FFN), ("Q8_0", 1, FFN), ("MXFP4", 1, FFN), ("Q4_K", 4, N))
    for qname, m, n in extra_gemvs:
      _, w_len, w_dt = _packed_words(n, K, qname)
      cases.append((f"quant_gemv_{qname}_{m}x{n}x{K}", w_len * w_dt.itemsize + m * (K + n) * 4, 2 * m * n * K,
                    _make_qlinear_setup(qname, m, n, dtypes.float32)))

  # 2. Quantized GGUF GEMM (prefill M=128, M=512 via quant_linear WMMA across GGUF encodings)
  gemm_quants = ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if (smoke or fast) else (
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_S", "IQ2_S", "MXFP4")
  for qname in gemm_quants:
    _, w_len, w_dt = _packed_words(N, K, qname)
    cases.append((f"quant_gemm_{qname}_{M_PREFILL}x{N}x{K}",
                  w_len * w_dt.itemsize + M_PREFILL * K * 2 + M_PREFILL * N * 4, 2 * M_PREFILL * N * K,
                  _make_qlinear_setup(qname, M_PREFILL, N, dtypes.float16)))

  if not smoke:
    for qname in (("Q4_K", "IQ4_XS") if fast else ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4")):
      _, w_len, w_dt = _packed_words(N, K, qname)
      cases.append((f"quant_gemm_{qname}_{M_PREFILL_LG}x{N}x{K}",
                    w_len * w_dt.itemsize + M_PREFILL_LG * K * 2 + M_PREFILL_LG * N * 4, 2 * M_PREFILL_LG * N * K,
                    _make_qlinear_setup(qname, M_PREFILL_LG, N, dtypes.float16)))

  # 3. Quantized GGUF MoE GEMV (via moe_linear: q8_quantize + moe_decode + split-K reduce)
  moe_quants = ("Q4_K",) if smoke else (("Q8_0", "Q4_K", "IQ4_XS") if fast else ("Q8_0", "Q4_K", "IQ4_XS", "IQ1_S", "MXFP4"))
  for qname in moe_quants:
    qtype, per_e_words, w_dt = _packed_words(MOE_N, MOE_K, qname)
    def _mk_moe(qname:str=qname, qtype:int=qtype) -> tuple[Callable[[], Tensor], VerifyFn]:
      w_packed, tile_u8 = _make_packed(E * MOE_N, MOE_K, qname, dev, seed=19)
      sel = Tensor(list(range(TOPK)), dtype=dtypes.int32, device=dev).realize()
      x = _nonuniform((1, 1, 1, MOE_K), dtypes.float32, dev, scale=0.5, freq=0.011)
      def _verify(out:Tensor) -> None:
        # All experts share the 16-row tile layout (since MOE_N % 16 == 0); check first CHECK_ROWS of each selected expert.
        w4 = ggml_data_to_tensor(tile_u8[:CHECK_ROWS].flatten(), CHECK_ROWS * MOE_K, qtype).reshape(CHECK_ROWS, MOE_K)
        ref1 = (x.reshape(1, MOE_K) @ w4.T).numpy()[0]
        got = out[0, 0, :, :CHECK_ROWS].float().numpy()
        _assert_close(got, np.broadcast_to(ref1, got.shape), rtol=4e-2, atol=5e-2)
      return _slop_call("moe_linear", target, dev, w_packed, sel, x, ggml_type=qtype, sel_shape=(1, 1, TOPK),
                        in_features=MOE_K, local_out=MOE_N, global_out=MOE_N), _verify
    cases.append((f"moe_gemv_{qname}_E{E}k{TOPK}_{MOE_N}x{MOE_K}", TOPK * per_e_words * w_dt.itemsize, 2 * TOPK * MOE_N * MOE_K, _mk_moe))

  # 4. Dense FP16 GEMV (via dense_linear, decode M=1, M=4)
  for m, n in (((1, FFN),) if fast else ((1, N), (1, FFN), (4, N))):
    def _mk_dense(m:int=m, n:int=n) -> tuple[Callable[[], Tensor], VerifyFn]:
      w = _nonuniform((n, K), dtypes.float16, dev, scale=0.25, freq=0.019)
      x = _nonuniform((m, K), dtypes.float32, dev, scale=0.5, freq=0.023)
      def _verify(out:Tensor) -> None:
        ref = (x @ w[:CHECK_ROWS].float().T).numpy()
        _assert_close(out[:, :CHECK_ROWS].float().numpy(), ref, rtol=3e-3, atol=3e-3)
      return _slop_call("dense_linear", target, dev, w, x), _verify
    cases.append((f"dense_gemv_f16_{m}x{n}x{K}", n * K * 2 + m * (K + n) * 4, 2 * m * n * K, _mk_dense))

  # 5. Flash Attention (prefill + 2-pass decode via flash_attention_op)
  if not cdna:
    def _mk_fa_prefill() -> tuple[Callable[[], Tensor], VerifyFn]:
      q = _nonuniform((1, ATTN_H, ATTN_M, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.029)
      kv = _nonuniform((2, 1, ATTN_HKV, ATTN_N, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.031, phase=0.7)
      def _verify(out:Tensor) -> None:
        mask = Tensor.full((ATTN_M, ATTN_N), float("-inf"), dtype=dtypes.float32, device=dev).triu(ATTN_N - ATTN_M + 1)
        ref = q[:, :1].float().scaled_dot_product_attention(kv[0, :, :1].float(), kv[1, :, :1].float(), attn_mask=mask, enable_gqa=True).numpy()
        _assert_close(out[:, :1].float().numpy(), ref, rtol=5e-3, atol=5e-3)
      return _slop_call("flash_attention_op", target, dev, q, kv, valid_kv_len=ATTN_N, decode=False), _verify
    fa_flops = 4 * ATTN_H * ATTN_M * ATTN_N * ATTN_D
    fa_bytes = (ATTN_H * ATTN_M * ATTN_D + 2 * ATTN_HKV * ATTN_N * ATTN_D) * 2 + ATTN_H * ATTN_M * ATTN_D * 4
    cases.append((f"flash_attn_prefill_H{ATTN_H}_M{ATTN_M}_N{ATTN_N}_D{ATTN_D}", fa_bytes, fa_flops, _mk_fa_prefill))

  for ctx in ((ATTN_N,) if smoke else (ATTN_N, ATTN_N_LONG)):
    def _mk_fa_dec(ctx:int=ctx) -> tuple[Callable[[], Tensor], VerifyFn]:
      q = _nonuniform((1, ATTN_H, 1, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.037)
      kv = _nonuniform((2, 1, ATTN_HKV, ctx, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.041, phase=1.1)
      def _verify(out:Tensor) -> None:
        ref = q[:, :1].float().scaled_dot_product_attention(kv[0, :, :1].float(), kv[1, :, :1].float(), enable_gqa=True).numpy()
        _assert_close(out[:, :1].float().numpy(), ref, rtol=5e-3, atol=5e-3)
      return _slop_call("flash_attention_op", target, dev, q, kv, valid_kv_len=ctx, decode=True), _verify
    dec_bytes = 2 * ATTN_HKV * ctx * ATTN_D * 2 + ATTN_H * ATTN_D * 6
    cases.append((f"flash_attn_decode_H{ATTN_H}_N{ctx}_D{ATTN_D}", dec_bytes, 4 * ATTN_H * ctx * ATTN_D, _mk_fa_dec))

  # 6. Gated Delta Net Prefill & mHC Coeffs
  gd_h, gd_t, gd_d = (2, 8, 32) if smoke else (16, 64, 128)
  def _mk_gd() -> tuple[Callable[[], Tensor], VerifyFn]:
    q = _nonuniform((1, gd_h, gd_t, gd_d), dtypes.float32, dev, scale=0.05, freq=0.043)
    k = _nonuniform((1, gd_h, gd_t, gd_d), dtypes.float32, dev, scale=0.05, freq=0.047, phase=0.5)
    v = _nonuniform((1, gd_h, gd_t, gd_d), dtypes.float32, dev, scale=0.2, freq=0.053, phase=0.9)
    beta = (_nonuniform((1, gd_h, gd_t), dtypes.float32, dev, scale=0.05, freq=0.059) + 0.1).contiguous().realize()
    alpha = (_nonuniform((1, gd_h, gd_t), dtypes.float32, dev, scale=0.05, freq=0.061) + 0.9).contiguous().realize()
    state = Tensor.zeros(1, gd_h, gd_d, gd_d, dtype=dtypes.float32, device=dev).contiguous().realize()
    def _verify(out:Tensor) -> None:
      q0, k0, v0 = q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy()
      b0, a0 = beta[0, 0].numpy(), alpha[0, 0].numpy()
      st, ref = np.zeros((gd_d, gd_d), dtype=np.float32), np.empty((gd_t, gd_d), dtype=np.float32)
      for t in range(gd_t):
        st *= a0[t]
        delta = (v0[t] - st @ k0[t]) * b0[t]
        st += np.outer(delta, k0[t])
        ref[t] = st @ q0[t]
      _assert_close(out[0, 0].numpy(), ref, rtol=2e-3, atol=2e-4)
      _assert_close(state[0, 0].numpy(), st, rtol=2e-3, atol=2e-4)
    return _slop_call("gated_delta_prefill_op", target, dev, q, k, v, beta, alpha, state), _verify
  cases.append((f"gated_delta_prefill_H{gd_h}_T{gd_t}_D{gd_d}", (4 * gd_h * gd_t * gd_d + 2 * gd_h * gd_d * gd_d) * 4,
                6 * gd_h * gd_t * gd_d * gd_d, _mk_gd))

  def _mk_hc() -> tuple[Callable[[], Tensor], VerifyFn]:
    t_h = 2 if smoke else 32
    mix = _nonuniform((t_h, 24), dtypes.float32, dev, scale=0.2, freq=0.067)
    flat = _nonuniform((t_h, K), dtypes.float32, dev, scale=0.5, freq=0.071)
    base = _nonuniform((24,), dtypes.float16, dev, scale=0.2, freq=0.073)
    scale = (_nonuniform((3,), dtypes.float16, dev, scale=0.2, freq=0.079) + 1.0).cast(dtypes.float16).contiguous().realize()
    def _verify(out:Tensor) -> None:
      inv_r = (flat.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
      m_n = mix * inv_r
      pre = (m_n[:, :4] * scale[0].float() + base[:4].float()).sigmoid()
      post = (m_n[:, 4:8] * scale[1].float() + base[4:8].float()).sigmoid() * 2.0
      comb = (m_n[:, 8:] * scale[2].float() + base[8:].float()).reshape(t_h, 4, 4).softmax(-1) + 1e-6
      for i in range(3): comb = comb / (comb.sum(-1 if i % 2 else -2, keepdim=True) + 1e-6)
      ref = pre.cat(post, comb.reshape(t_h, 16), dim=-1).numpy()
      _assert_close(out.float().numpy(), ref, rtol=2e-3, atol=2e-3)
    return _slop_call("hc_coeffs_op", target, dev, mix, flat, base, scale, count=4, iters=2, eps=1e-6, norm_eps=1e-6), _verify
  cases.append((f"hc_coeffs_{2 if smoke else 32}x{K}", (2 if smoke else 32) * K * 4, 0, _mk_hc))

  return cases

def main() -> None:
  ap = argparse.ArgumentParser(description="Benchmark & verify end-to-end LLM slopkernels via UIR on AMD GPUs")
  ap.add_argument("--warmup", type=int, default=2, help="Warmup iterations per op")
  ap.add_argument("--iters", type=int, default=5, help="Timed iterations per op")
  ap.add_argument("--filter", type=str, default="", help="Regex filter on benchmark name")
  ap.add_argument("--smoke", action="store_true", help="Use small shapes for fast emulator verification")
  ap.add_argument("--fast", action="store_true", help="Representative ~26-op LLM subset for <90s CI budget")
  ap.add_argument("--no-verify", action="store_true", help="Skip numerical correctness checks")
  ap.add_argument("--flush-mb", type=int, default=256, help="MB of clean read-only VRAM traffic per iter to evict L2/Infinity Cache (0=disable)")
  ap.add_argument("--json", type=str, default="", help="Optional path to write JSON results")
  args = ap.parse_args()

  dev = Device.DEFAULT
  target = getattr(Device[dev], "target", None)
  flush_mb = min(args.flush_mb, 2) if args.smoke else args.flush_mb
  flush_fn = _make_cache_flusher(flush_mb)
  t_start = time.perf_counter()
  print(f"slopkernels LLM benchmark | DEV={dev} | target={target} | warmup={args.warmup} iters={args.iters} flush={flush_mb}MB "
        f"verify={not args.no_verify} smoke={args.smoke}", flush=True)
  print(f"{'Op':<46} {'Median (us)':>12} {'Min (us)':>10} {'GB/s':>9} {'TFLOPS':>9}", flush=True)
  print("-" * 90, flush=True)

  pat = re.compile(args.filter) if args.filter else None
  results: list[BenchResult] = []
  for name, io_bytes, flops, setup_fn in build_benchmarks(dev, smoke=args.smoke, fast=args.fast):
    if pat is not None and not pat.search(name): continue
    step_fn, verify_fn = setup_fn()
    med_us, min_us = _time_fn(step_fn, None if args.no_verify else verify_fn, args.warmup, args.iters, flush_fn)
    gb_s = (io_bytes / 1e9) / (med_us / 1e6) if io_bytes > 0 and med_us > 0 else 0.0
    tflops = (flops / 1e12) / (med_us / 1e6) if flops > 0 and med_us > 0 else 0.0
    r = BenchResult(name=name, median_us=med_us, min_us=min_us, gb_s=gb_s, tflops=tflops)
    results.append(r)
    print(f"{r.name:<46} {r.median_us:>12.1f} {r.min_us:>10.1f} {r.gb_s:>9.1f} {r.tflops:>9.2f}", flush=True)

  print("-" * 90, flush=True)
  print(f"Benchmarked & verified {len(results)} LLM ops in {time.perf_counter() - t_start:.1f}s.", flush=True)
  if args.json:
    with open(args.json, "w") as f:
      json.dump([r.__dict__ for r in results], f, indent=2)

if __name__ == "__main__":
  main()
