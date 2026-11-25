"""End-to-end LLM kernel benchmark & correctness suite for slopkernels on AMD GPUs (e.g. 7900XTX, 9060XT).

Usage:
  DEV=AMD python3 -m slopkernels.bench
  DEV=AMD python3 -m slopkernels.bench --filter quant_gemv --iters 10
  DEV="MOCK+AMD" python3 -m slopkernels.bench --smoke --warmup 1 --iters 1
"""
from __future__ import annotations

import argparse, functools, json, math, pathlib, re, statistics, struct, sys, time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

if str(root := pathlib.Path(__file__).resolve().parent.parent) not in sys.path: sys.path.insert(0, str(root))

from tinygrad import Device, Tensor, UOp, dtypes
from tinygrad.dtype import DType
from tinygrad.engine.realize import ExecContext, compile_linear, link_linear, pm_exec
from tinygrad.helpers import ceildiv, prod, unwrap
from tinygrad.llm.gguf import ggml_data_to_tensor

import slopkernels

_QUANT_META: dict[str, tuple[int, int, DType, str]] = {
  "Q8_0": (8, 272, dtypes.uint16, "q8_0"), "Q6_K": (14, 210, dtypes.uint16, "q6"), "Q5_K": (13, 176, dtypes.uint32, "q5_k"),
  "Q4_K": (12, 144, dtypes.uint32, "q4_k"), "Q3_K": (11, 110, dtypes.uint16, "q3_k"), "Q2_K": (10, 84, dtypes.uint16, "q2_k"),
  "IQ4_XS": (23, 136, dtypes.uint32, "iq4_xs"), "IQ4_NL": (20, 144, dtypes.uint16, "iq4_nl"), "IQ3_S": (21, 110, dtypes.uint16, "iq3_s"),
  "IQ3_XXS": (18, 98, dtypes.uint16, "iq3_xxs"), "IQ2_S": (22, 82, dtypes.uint16, "iq2_s"), "IQ2_XS": (17, 74, dtypes.uint16, "iq2_xs"),
  "IQ2_XXS": (16, 66, dtypes.uint16, "iq2_xxs"), "IQ1_S": (19, 50, dtypes.uint16, "iq1_s"), "MXFP4": (39, 136, dtypes.uint8, "mxfp4"),
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

def _arch_info(dev:str) -> tuple[tuple[int, ...], bool]:
  raw = getattr(Device[dev], "target", None)
  target = tuple(int(x) for x in raw) if isinstance(raw, tuple) else (11, 0, 0)
  return target, target[0] == 9

def _packed_words(rows:int, cols:int, qname:str) -> tuple[int, int, DType, str]:
  qtype, blk_bytes, w_dt, kname = _QUANT_META[qname]
  return qtype, (rows * cols * blk_bytes // 256) // w_dt.itemsize, w_dt, kname

def _nonuniform(shape:tuple[int, ...], dtype:DType, dev:str, scale:float=0.5, freq:float=0.017, phase:float=0.3) -> Tensor:
  t = ((Tensor.arange(prod(shape), dtype=dtypes.float32) + Tensor.zeros(1, dtype=dtypes.float32, device=dev)) * freq + phase).sin() * scale
  return t.cast(dtype).reshape(shape).contiguous().realize()

def _make_packed(rows:int, cols:int, qname:str, dev:str, seed:int=7) -> tuple[Tensor, Tensor]:
  qtype, blk_bytes, w_dt, _ = _QUANT_META[qname]
  row_bytes, tile_rows = cols * blk_bytes // 256, min(16, rows)
  rng = np.random.default_rng(seed + qtype)
  if qtype == 39:
    blocks = rng.integers(0, 256, size=(tile_rows, cols // 32, 17), dtype=np.uint8)
    blocks[:, :, 0] = rng.integers(0x7b, 0x80, size=(tile_rows, cols // 32), dtype=np.uint8)
    tile_np = blocks.reshape(tile_rows, row_bytes)
  else:
    tile_np = rng.integers(0x20, 0x38, size=(tile_rows, row_bytes), dtype=np.uint8)
  tile_u8 = Tensor(tile_np, device=dev).realize()
  np_dt = {dtypes.uint8: np.uint8, dtypes.uint16: np.uint16, dtypes.uint32: np.uint32}[w_dt]
  tile_w = Tensor(np.ascontiguousarray(tile_np).view(np_dt), device=dev).realize()
  reps, row_words = rows // tile_rows, row_bytes // w_dt.itemsize
  packed = tile_w.flatten() if reps == 1 else tile_w.reshape(1, tile_rows, row_words).expand(reps, tile_rows, row_words).reshape(-1)
  return packed.contiguous().realize(), tile_u8

def _assert_close(got:np.ndarray, ref:np.ndarray, rtol:float, atol:float) -> None:
  scale_atol = max(atol, rtol * float(np.max(np.abs(ref))) if ref.size else atol)
  np.testing.assert_allclose(got, ref, rtol=rtol, atol=scale_atol)

def _make_cache_flusher(flush_mb:int) -> Callable[[], None]:
  if flush_mb <= 0: return lambda: None
  src = Tensor.ones(flush_mb * 1024 * 1024 // 4, dtype=dtypes.float32).contiguous().realize()
  sink = src.sum()
  Device[Device.DEFAULT].synchronize()
  linked = link_linear(compile_linear(sink.schedule_linear(), beam=0, profile=False, cache=True))
  ctx, keep = ExecContext({}, update_stats=False, wait=True, cache=True), (src, sink, linked)
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
  elif not math.isfinite(first := float(out.flatten()[:1].float().item())): raise RuntimeError(f"non-finite benchmark output: {first}")
  for _ in range(max(0, warmup - 1)): _run_once()
  times_us = []
  for _ in range(iters):
    flush_fn()
    times_us.append(_run_once() * 1e6)
  return statistics.median(times_us), min(times_us)

def _krun(name:str, shp:tuple[int, ...], dev:str|tuple[str, ...], *args:Tensor|UOp, axis:int|None=None, **kw:int|float) -> Tensor:
  k = unwrap(slopkernels.get(name)).specialize(**kw)
  o = UOp.alloc(shp, dtypes.float32, device=dev, axis=axis)
  return Tensor(o.after(k.call(o, *(a.uop if isinstance(a, Tensor) else a for a in args))))

def _q8(x:Tensor, tokens:int, in_f:int, dev:str|tuple[str, ...], axis:int|None=None) -> tuple[UOp, UOp, UOp]:
  k = unwrap(slopkernels.get(f"amd_q8_quantize_{x.dtype.name}")).specialize(tokens=tokens, in_groups=(g := in_f // 32))
  f = dtypes.float32
  q, d, s = (UOp.alloc((tokens, g, *e), dt, device=dev, axis=axis) for e, dt in (((8,), dtypes.uint32), ((), f), ((2,), f)))
  c = k.call(q, d, s, x.reshape(tokens, in_f).uop if x.uop.axis is None else x.uop)
  return q.after(c), d.after(c), s.after(c)

@functools.cache
def _iq_lut(t:int, dev:str|tuple[str, ...], wmma:bool=False) -> tuple[UOp, ...]:
  if t not in ({16, 17, 18, 19, 21, 22, 23} if wmma else {16, 17, 18, 19, 21, 22}): return ()
  from tinygrad.runtime.autogen import ggml_common as g
  if t == 23 or (wmma and t in (18, 22)):
    vals = ([g.kvalues_iq4nl[k] for j in range(16) for i in range(16) for k in (i, j)] if t == 23 else
            [((v >> (8 * i)) & 0xff) / 64.0 for v in (g.iq3xxs_grid if t == 18 else g.iq2s_grid) for i in range(4 if t == 18 else 8)])
    words: tuple[int, ...] = struct.unpack(f"<{len(vals) // 2}I", struct.pack(f"<{len(vals)}e", *vals))
  else:
    grid, nw = {16:(g.iq2xxs_grid, 2), 17:(g.iq2xs_grid, 2), 19:(g.iq1s_grid, 2), 22:(g.iq2s_grid, 2),
                18:(g.iq3xxs_grid, 1), 21:(g.iq3s_grid, 1)}[t]
    words = tuple((v >> (32 * i)) & 0xffffffff for v in grid for i in range(nw))
  return (Tensor(np.array(words, dtype=np.uint32), device=dev).realize().uop,)

def build_benchmarks(dev:str, smoke:bool=False, fast:bool=False) -> list[tuple[str, int, int, SetupFn]]:
  target, cdna = _arch_info(dev)
  N, K, FFN = (128, 256, 256) if smoke else (4096, 4096, 14336)
  M_PREFILL, M_PREFILL_LG = (16, 32) if smoke else (128, 512)
  E, TOPK, MOE_N, MOE_K = (8, 4, 128, 256) if smoke else (64, 8, 2048, 4096)
  ATTN_H, ATTN_HKV, ATTN_D, ATTN_N, ATTN_M = (4, 2, 64, 128, 32) if smoke else (32, 8, 128, 2048, 256)
  ATTN_N_LONG, CHECK_ROWS = 256 if smoke else 8192, 4
  arch_gemm, arch_gemv = "cdna" if cdna else "rdna4" if target[0] == 12 else "rdna3", "cdna" if cdna else "rdna"
  cases: list[tuple[str, int, int, SetupFn]] = []

  def _make_qlinear_setup(qname:str, m:int, n:int, x_dt:DType) -> SetupFn:
    qtype, _, _, kname = _QUANT_META[qname]
    def _setup() -> tuple[Callable[[], Tensor], VerifyFn]:
      w_packed, tile_u8 = _make_packed(n, K, qname, dev)
      x = _nonuniform((m, K), x_dt, dev, scale=0.5, freq=0.013)
      w_len = int(w_packed.numel())
      def _verify(out:Tensor) -> None:
        w4 = ggml_data_to_tensor(tile_u8[:CHECK_ROWS].flatten(), CHECK_ROWS * K, qtype).reshape(CHECK_ROWS, K)
        _assert_close(out[:, :CHECK_ROWS].float().numpy(), (x.float() @ w4.T).numpy(), rtol=4e-2, atol=5e-2)
      if m % 16 == 0 and n % 16 == 0:
        mt = 8 if m % 128 == 0 and m >= 256 else 4 if m % 64 == 0 else 2 if m % 32 == 0 else 1
        return (lambda: _krun(f"{arch_gemm}_gemm_{kname}_f16", (m, n), dev, w_packed, x.half().contiguous(),
                              *_iq_lut(qtype, dev, True), m_blks=m // (mt * 16), m_tiles=mt, out_features=n, in_features=K, raw_len=w_len)), _verify
      ch = ceildiv(K, 1024)
      def _run() -> Tensor:
        out = _krun(f"{arch_gemv}_gemv_{kname}", (m, n, ch), dev, w_packed, *_q8(x, m, K, dev), *_iq_lut(qtype, dev),
                    tokens=m, out_features=n, in_features=K, chunks=ch, raw_len=w_len, group_rows=0)
        return out.sum(-1) if ch > 1 else out.reshape(m, n)
      return _run, _verify
    return _setup

  gemv_quants = ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if smoke else (
    ("Q8_0", "Q6_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_S", "IQ1_S", "MXFP4") if fast else
    ("Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ1_S", "MXFP4"))
  for qname in gemv_quants:
    _, w_len, w_dt, _ = _packed_words(N, K, qname)
    cases.append((f"quant_gemv_{qname}_1x{N}x{K}", w_len * w_dt.itemsize + (K + N) * 4, 2 * N * K,
                  _make_qlinear_setup(qname, 1, N, dtypes.float32)))
  if not smoke:
    for qname, m, n in ((("Q4_K", 1, FFN), ("Q8_0", 1, FFN), ("Q4_K", 4, N)) if fast else
                        (("Q4_K", 1, FFN), ("Q8_0", 1, FFN), ("MXFP4", 1, FFN), ("Q4_K", 4, N))):
      _, w_len, w_dt, _ = _packed_words(n, K, qname)
      cases.append((f"quant_gemv_{qname}_{m}x{n}x{K}", w_len * w_dt.itemsize + m * (K + n) * 4, 2 * m * n * K,
                    _make_qlinear_setup(qname, m, n, dtypes.float32)))

  gemm_quants = ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4") if (smoke or fast) else (
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q2_K", "IQ4_XS", "IQ3_S", "IQ2_S", "MXFP4")
  for qname in gemm_quants:
    _, w_len, w_dt, _ = _packed_words(N, K, qname)
    cases.append((f"quant_gemm_{qname}_{M_PREFILL}x{N}x{K}",
                  w_len * w_dt.itemsize + M_PREFILL * K * 2 + M_PREFILL * N * 4, 2 * M_PREFILL * N * K,
                  _make_qlinear_setup(qname, M_PREFILL, N, dtypes.float16)))
  if not smoke:
    for qname in (("Q4_K", "IQ4_XS") if fast else ("Q8_0", "Q4_K", "IQ4_XS", "MXFP4")):
      _, w_len, w_dt, _ = _packed_words(N, K, qname)
      cases.append((f"quant_gemm_{qname}_{M_PREFILL_LG}x{N}x{K}",
                    w_len * w_dt.itemsize + M_PREFILL_LG * K * 2 + M_PREFILL_LG * N * 4, 2 * M_PREFILL_LG * N * K,
                    _make_qlinear_setup(qname, M_PREFILL_LG, N, dtypes.float16)))

  moe_quants = ("Q4_K",) if smoke else (("Q8_0", "Q4_K", "IQ4_XS") if fast else ("Q8_0", "Q4_K", "IQ4_XS", "IQ1_S", "MXFP4"))
  for qname in moe_quants:
    qtype, per_e_words, w_dt, kname = _packed_words(MOE_N, MOE_K, qname)
    def _mk_moe(qname:str=qname, qtype:int=qtype, kname:str=kname) -> tuple[Callable[[], Tensor], VerifyFn]:
      w_packed, tile_u8 = _make_packed(E * MOE_N, MOE_K, qname, dev, seed=19)
      sel = Tensor(list(range(TOPK)), dtype=dtypes.int32, device=dev).realize()
      x = _nonuniform((1, 1, 1, MOE_K), dtypes.float32, dev, scale=0.5, freq=0.011)
      ch, w_len = ceildiv(MOE_K, 1024), int(w_packed.numel())
      def _verify(out:Tensor) -> None:
        w4 = ggml_data_to_tensor(tile_u8[:CHECK_ROWS].flatten(), CHECK_ROWS * MOE_K, qtype).reshape(CHECK_ROWS, MOE_K)
        ref1, got = (x.reshape(1, MOE_K) @ w4.T).numpy()[0], out[0, 0, :, :CHECK_ROWS].float().numpy()
        _assert_close(got, np.broadcast_to(ref1, got.shape), rtol=4e-2, atol=5e-2)
      def _run() -> Tensor:
        return _krun(f"{arch_gemv}_moe_{kname}", (TOPK, MOE_N, ch), dev, w_packed, *_q8(x, 1, MOE_K, dev), sel, *_iq_lut(qtype, dev),
                     rows=TOPK, x_rows=1, out_features=MOE_N, in_features=MOE_K, chunks=ch, raw_len=w_len, x_share=TOPK,
                     ndev=1).sum(-1).reshape(1, 1, TOPK, MOE_N)
      return _run, _verify
    cases.append((f"moe_gemv_{qname}_E{E}k{TOPK}_{MOE_N}x{MOE_K}", TOPK * per_e_words * w_dt.itemsize, 2 * TOPK * MOE_N * MOE_K, _mk_moe))

  for m, n in (((1, FFN),) if fast else ((1, N), (1, FFN), (4, N))):
    def _mk_dense(m:int=m, n:int=n) -> tuple[Callable[[], Tensor], VerifyFn]:
      w = _nonuniform((n, K), dtypes.float16, dev, scale=0.25, freq=0.019)
      x = _nonuniform((m, K), dtypes.float32, dev, scale=0.5, freq=0.023)
      def _verify(out:Tensor) -> None:
        _assert_close(out[:, :CHECK_ROWS].float().numpy(), (x @ w[:CHECK_ROWS].float().T).numpy(), rtol=3e-3, atol=3e-3)
      return (lambda: _krun("amd_gemv_f16_f32", (m, n), dev, w, x, tokens=m, out_features=n, in_blks=K // 128, val_chunk=4)), _verify
    cases.append((f"dense_gemv_f16_{m}x{n}x{K}", n * K * 2 + m * (K + n) * 4, 2 * m * n * K, _mk_dense))

  if not cdna:
    def _mk_fa_prefill() -> tuple[Callable[[], Tensor], VerifyFn]:
      q = _nonuniform((1, ATTN_H, ATTN_M, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.029)
      kv = _nonuniform((2, 1, ATTN_HKV, ATTN_N, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.031, phase=0.7)
      def _verify(out:Tensor) -> None:
        mask = Tensor.full((ATTN_M, ATTN_N), float("-inf"), dtype=dtypes.float32, device=dev).triu(ATTN_N - ATTN_M + 1)
        ref = q[:, :1].float().scaled_dot_product_attention(kv[0, :, :1].float(), kv[1, :, :1].float(), attn_mask=mask, enable_gqa=True).numpy()
        _assert_close(out[:, :1].float().numpy(), ref, rtol=5e-3, atol=5e-3)
      def _run() -> Tensor:
        return _krun(f"{arch_gemm}_flash_attn", (ATTN_H, ATTN_M, ATTN_D), dev, q.reshape(ATTN_H, ATTN_M, ATTN_D), kv,
                     UOp.const(ATTN_N), UOp.const(ATTN_M), BH=ATTN_H, m_blks=ATTN_M // 32, B=1, H_KV=ATTN_HKV, d_blks=ATTN_D // 64,
                     n_blks=ATTN_N // 32, valid_kv_offset=0, q_start_from_end=1).reshape(1, ATTN_H, ATTN_M, ATTN_D)
      return _run, _verify
    cases.append((f"flash_attn_prefill_H{ATTN_H}_M{ATTN_M}_N{ATTN_N}_D{ATTN_D}",
                  (ATTN_H * ATTN_M * ATTN_D + 2 * ATTN_HKV * ATTN_N * ATTN_D) * 2 + ATTN_H * ATTN_M * ATTN_D * 4,
                  4 * ATTN_H * ATTN_M * ATTN_N * ATTN_D, _mk_fa_prefill))

  for ctx in ((ATTN_N,) if smoke else (ATTN_N, ATTN_N_LONG)):
    def _mk_fa_dec(ctx:int=ctx) -> tuple[Callable[[], Tensor], VerifyFn]:
      q = _nonuniform((1, ATTN_H, 1, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.037)
      kv = _nonuniform((2, 1, ATTN_HKV, ctx, ATTN_D), dtypes.float16, dev, scale=0.2, freq=0.041, phase=1.1)
      def _verify(out:Tensor) -> None:
        ref = q[:, :1].float().scaled_dot_product_attention(kv[0, :, :1].float(), kv[1, :, :1].float(), enable_gqa=True).numpy()
        _assert_close(out[:, :1].float().numpy(), ref, rtol=5e-3, atol=5e-3)
      def _run() -> Tensor:
        ch, grp, waves = min(48, ceildiv(ctx, 64)), ATTN_H // ATTN_HKV, 16
        while waves * grp * ((ATTN_D + 4) * 2 + 8) > 65536: waves //= 2
        kp = unwrap(slopkernels.get("amd_flash_decode_partial")).specialize(
          B=1, H_KV=ATTN_HKV, G=grp, d_blks=ATTN_D // 64, n_blks=ctx // 64, C=ch, valid_kv_offset=0, waves=waves)
        partial, stats = (UOp.alloc((1, ATTN_H, ch, d), dtypes.float32, device=dev) for d in (ATTN_D, 2))
        cp = kp.call(partial, stats, q.uop, kv.uop, UOp.const(ctx))
        return _krun("amd_flash_decode_combine", (1, ATTN_H, 1, ATTN_D), dev, partial.after(cp), stats.after(cp), UOp.const(ctx),
                     B=1, H=ATTN_H, C=ch, d_blks=ATTN_D // 64, valid_kv_offset=0)
      return _run, _verify
    cases.append((f"flash_attn_decode_H{ATTN_H}_N{ctx}_D{ATTN_D}", 2 * ATTN_HKV * ctx * ATTN_D * 2 + ATTN_H * ATTN_D * 6,
                  4 * ATTN_H * ctx * ATTN_D, _mk_fa_dec))

  gd_h, gd_t, gd_d = (2, 8, 32) if smoke else (16, 64, 128)
  def _mk_gd() -> tuple[Callable[[], Tensor], VerifyFn]:
    q = _nonuniform((1, gd_h, gd_t, gd_d), dtypes.float32, dev, scale=0.05, freq=0.043)
    k = _nonuniform((1, gd_h, gd_t, gd_d), dtypes.float32, dev, scale=0.05, freq=0.047, phase=0.5)
    v = _nonuniform((1, gd_h, gd_t, gd_d), dtypes.float32, dev, scale=0.2, freq=0.053, phase=0.9)
    beta = (_nonuniform((1, gd_h, gd_t), dtypes.float32, dev, scale=0.05, freq=0.059) + 0.1).contiguous().realize()
    alpha = (_nonuniform((1, gd_h, gd_t), dtypes.float32, dev, scale=0.05, freq=0.061) + 0.9).contiguous().realize()
    state = Tensor.zeros(1, gd_h, gd_d, gd_d, dtype=dtypes.float32, device=dev).contiguous().realize()
    def _verify(out:Tensor) -> None:
      q0, k0, v0, b0, a0 = q[0, 0].numpy(), k[0, 0].numpy(), v[0, 0].numpy(), beta[0, 0].numpy(), alpha[0, 0].numpy()
      st, ref = np.zeros((gd_d, gd_d), dtype=np.float32), np.empty((gd_t, gd_d), dtype=np.float32)
      for t in range(gd_t):
        st *= a0[t]
        st += np.outer((v0[t] - st @ k0[t]) * b0[t], k0[t])
        ref[t] = st @ q0[t]
      _assert_close(out[0, 0].numpy(), ref, rtol=2e-3, atol=2e-4)
      _assert_close(state[0, 0].numpy(), st, rtol=2e-3, atol=2e-4)
    return (lambda: _krun("amd_gated_delta", (1, gd_h, gd_t, gd_d), dev, q, k, v, beta, alpha.reshape(1, gd_h, gd_t, 1), state,
                          (q * k).sum(-1).contiguous(), UOp.const(1), batch=1, heads=gd_h, tokens=gd_t, v_tiles=gd_d // 4,
                          k_blks=gd_d // 32, alpha_k=1)), _verify
  cases.append((f"gated_delta_prefill_H{gd_h}_T{gd_t}_D{gd_d}", (4 * gd_h * gd_t * gd_d + 2 * gd_h * gd_d * gd_d) * 4,
                6 * gd_h * gd_t * gd_d * gd_d, _mk_gd))

  def _mk_hc() -> tuple[Callable[[], Tensor], VerifyFn]:
    t_h = 2 if smoke else 32
    mix = _nonuniform((t_h, 24), dtypes.float32, dev, scale=0.2, freq=0.067)
    flat = _nonuniform((t_h, K), dtypes.float32, dev, scale=0.5, freq=0.071)
    base = _nonuniform((24,), dtypes.float16, dev, scale=0.2, freq=0.073)
    scale = (_nonuniform((3,), dtypes.float16, dev, scale=0.2, freq=0.079) + 1.0).cast(dtypes.float16).contiguous().realize()
    def _verify(out:Tensor) -> None:
      m_n = mix * (flat.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
      pre = (m_n[:, :4] * scale[0].float() + base[:4].float()).sigmoid()
      post = (m_n[:, 4:8] * scale[1].float() + base[4:8].float()).sigmoid() * 2.0
      comb = (m_n[:, 8:] * scale[2].float() + base[8:].float()).reshape(t_h, 4, 4).softmax(-1) + 1e-6
      for i in range(3): comb = comb / (comb.sum(-1 if i % 2 else -2, keepdim=True) + 1e-6)
      _assert_close(out.float().numpy(), pre.cat(post, comb.reshape(t_h, 16), dim=-1).numpy(), rtol=2e-3, atol=2e-3)
    return (lambda: _krun("amd_hc_coeffs_f16_f16", (t_h, 24), dev, mix, flat.reshape(t_h, 32, -1).square().sum(-1).contiguous(),
                          base, scale, tokens=t_h, features=K, iters=2, eps=1e-6, norm_eps=1e-6)), _verify
  cases.append((f"hc_coeffs_{2 if smoke else 32}x{K}", (2 if smoke else 32) * K * 4, 0, _mk_hc))
  return cases

def main() -> None:
  ap = argparse.ArgumentParser(description="Benchmark & verify end-to-end LLM slopkernels via UIR on AMD GPUs")
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--iters", type=int, default=5)
  ap.add_argument("--filter", type=str, default="")
  ap.add_argument("--smoke", action="store_true")
  ap.add_argument("--fast", action="store_true")
  ap.add_argument("--no-verify", action="store_true")
  ap.add_argument("--flush-mb", type=int, default=256)
  ap.add_argument("--json", type=str, default="")
  args = ap.parse_args()

  dev, target = Device.DEFAULT, getattr(Device[Device.DEFAULT], "target", None)
  flush_mb = min(args.flush_mb, 2) if args.smoke else args.flush_mb
  flush_fn, t_start = _make_cache_flusher(flush_mb), time.perf_counter()
  print(f"slopkernels LLM benchmark | DEV={dev} | target={target} | warmup={args.warmup} iters={args.iters} flush={flush_mb}MB "
        f"verify={not args.no_verify} smoke={args.smoke}", flush=True)
  print(f"{'Op':<46} {'Median (us)':>12} {'Min (us)':>10} {'GB/s':>9} {'TFLOPS':>9}", flush=True)
  print("-" * 90, flush=True)
  pat, results = re.compile(args.filter) if args.filter else None, []
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
    with open(args.json, "w") as f: json.dump([r.__dict__ for r in results], f, indent=2)

if __name__ == "__main__":
  main()
