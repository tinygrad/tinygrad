#!/usr/bin/env python3
# PROFILE=1 DEV=AMD python3 extra/k3/bench_prefill.py: the chunked prefill's kernels of a gpu, timed in a jit
#   gemm: the grouped expert gemms (gate+up, down) at T tokens of a Zipf-ish routing (16 of 896 experts a token, ZIPF=s), with a check
#   attn: the mla prefill attention (16 heads, T tokens at start_pos)
import os, time
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes, Device
from tinygrad.helpers import getenv
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import expert_gemm, expert_combine, QUANT_SIZES, IQ3_XXS
from extra.k3.bench_q4k import kernel_us

E, K, LAT, UNITS = 896, 16, 3584, 768
ROW_UP, ROW_DOWN = LAT // 256 * QUANT_SIZES[IQ3_XXS], UNITS // 256 * QUANT_SIZES[IQ3_XXS]
PART = 2 * UNITS * ROW_UP + LAT * ROW_DOWN  # 2,109,440: gate+up rows, then down
DOWN_OFF = 2 * UNITS * ROW_UP // 2
PEAK = getenv("PEAK", 191)  # dense f16 wmma TFLOPS (R9700: 64 CUs x 2 SIMDs x 512 flops a clock x 2.92 GHz)

def route(T:int, s:float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
  """rows, tslot (of P max), ntiles, P, the pairs, dest (pair t*K+k's row): each token's 16 experts by Gumbel top-k of a Zipf popularity,
  sorted, a run padded to 16"""
  logp = (-s * np.log(np.arange(1, E + 1)))[rng.permutation(E)]
  top = np.argpartition(-(logp + rng.gumbel(size=(T, E))), K, axis=1)[:, :K]
  tok, ex = np.repeat(np.arange(T), K), top.ravel()
  order = np.argsort(ex, kind="stable")
  tok, ex = tok[order], ex[order]
  dest = np.zeros(T * K, np.int32)
  pmax = T * K + E * 15
  rows, tslot = np.zeros(pmax, np.int32), np.zeros(pmax // 16, np.int32)
  p = 0
  for e in np.unique(ex):
    t = tok[ex == e]
    n = -(-len(t) // 16) * 16
    rows[p:p + len(t)], rows[p + len(t):p + n] = t, t[0]
    dest[order[ex == e]] = np.arange(p, p + len(t))
    tslot[p // 16:(p + n) // 16] = e
    p += n
  return rows, tslot, np.array([p // 16], np.int32), p, T * K, dest

def table(a:np.ndarray) -> Tensor:  # HOST=1: in uncached host memory (read over pcie, as the engine's host_tensor), else vram
  if not getenv("HOST"): return Tensor(a).realize()
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  buf = Buffer(Device.DEFAULT, a.nbytes, options=BufferSpec(host=True, uncached=True, cpu_access=True)).ensure_allocated()
  np.frombuffer(buf.get_storage().host.mv.cast('B'), dtype=a.dtype, count=a.size)[:] = a.ravel()
  return Tensor(UOp.from_buffer(buf, dtypes.int32)).reshape(a.shape)

def bench_gemm():
  rng = np.random.default_rng(0)
  # 8 distinct expert parts (sane block scales), the pool's 896 slots of them
  blob = rng.integers(0, 256, 8 * PART, dtype=np.uint8)
  blob.reshape(-1, QUANT_SIZES[IQ3_XXS])[:, :2] = np.frombuffer(np.full(8 * PART // QUANT_SIZES[IQ3_XXS], 0.01, np.float16).tobytes(),
                                                                np.uint8).reshape(-1, 2)
  pool = Tensor(blob).bitcast(dtypes.uint16).reshape(8, PART // 2).repeat(E // 8, 1).reshape(-1).contiguous().realize()
  for T in eval(os.environ.get("TS", "(1024, 2048)")):
    rows, tslot, ntiles, P, pairs, dest = route(T, getenv("ZIPF", 0.8), rng)
    x = Tensor(rng.standard_normal((T, LAT)).astype(np.float16)).realize()
    h = Tensor(rng.standard_normal((len(rows), UNITS)).astype(np.float16)).realize()
    t_rows, t_tslot, t_ntiles, t_dest = table(rows), table(tslot), table(ntiles), table(dest)
    wts = Tensor(rng.uniform(0, 1, T * K).astype(np.float32)).realize()
    gu_fn = TinyJit(lambda x, r, s, n: expert_gemm(pool, x, s, n, 2 * UNITS, LAT, PART // 2, rows=r).realize())
    ydt = dtypes.half if getenv("HALF") else dtypes.float  # (HALF=1: the down gemm's output half)
    dn_fn = TinyJit(lambda h, s, n: expert_gemm(pool, h, s, n, LAT, UNITS, PART // 2, DOWN_OFF, dtype=ydt).realize())
    res = {}
    for name, fn, args, o, i in (("gate_up", gu_fn, (x, t_rows, t_tslot, t_ntiles), 2 * UNITS, LAT),
                                 ("down", dn_fn, (h, t_tslot, t_ntiles), LAT, UNITS)):
      us = sum(kernel_us(lambda: fn(*args)).values())
      res[name] = us
      print(f"T={T} {name}: P {P} ({pairs} pairs, {P // 16} tiles) {us / 1e3:.3f} ms, {2 * pairs * o * i / us / 1e6:.1f} TFLOPS of the pairs, "
            f"{2 * P * o * i / us / 1e6:.1f} of the padded ({2 * P * o * i / us / 1e6 / PEAK * 100:.0f}% of {PEAK})", flush=True)
    y = Tensor.empty(len(rows), LAT, dtype=ydt).realize()
    cb_fn = TinyJit(lambda y, d, w: expert_combine(y, d, w, T, K).realize())
    res["combine"] = us = sum(kernel_us(lambda: cb_fn(y, t_dest, wts)).values())
    print(f"T={T} combine: {us / 1e3:.3f} ms, {T * K * LAT * ydt.itemsize / us / 1e3:.0f} GB/s of y's rows", flush=True)
    print(f"T={T} gate_up+down+combine: {sum(res.values()) / 1e3:.3f} ms a layer, {sum(res.values()) / T:.2f} us a token", flush=True)
    if getenv("CHECK", 1):  # some rows of each against the dequantized weights
      gu, dn = gu_fn(x, t_rows, t_tslot, t_ntiles).numpy(), dn_fn(h, t_tslot, t_ntiles).numpy()
      xs, hs = x.numpy().astype(np.float32), h.numpy().astype(np.float32)
      for slot in sorted({int(tslot[0]), int(tslot[ntiles[0] // 2]), int(tslot[ntiles[0] - 1])}):
        raw = Tensor(blob[(slot % 8) * PART:(slot % 8 + 1) * PART])
        wgu = ggml_data_to_tensor(raw[:DOWN_OFF * 2], 2 * UNITS * LAT, IQ3_XXS).reshape(2 * UNITS, LAT).numpy()
        wdn = ggml_data_to_tensor(raw[DOWN_OFF * 2:], LAT * UNITS, IQ3_XXS).reshape(LAT, UNITS).numpy()
        ps = np.nonzero(np.repeat(tslot[:ntiles[0]], 16) == slot)[0]
        for p in ps[:: max(1, len(ps) // 8)]:
          for got, ref in ((gu[p], wgu @ xs[rows[p]]), (dn[p].astype(np.float32), wdn @ hs[p])):
            np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2 * np.abs(ref).max())
      print(f"T={T}: checked", flush=True)

def bench_attn():
  from tinygrad.llm.kernels.amd import mla_prefill_attention
  H, D, DV, N = 16, 576, 512, getenv("N", 131072)
  cache = Tensor.randn(1, 1, N, D, dtype=dtypes.half).realize()
  for T, S in eval(os.environ.get("TSP", "((1024, 0), (2048, 0), (4096, 0), (1024, 32768), (4096, 126976))")):
    q = Tensor.randn(1, H, T, D, dtype=dtypes.half).realize()
    sp = UOp.variable("start_pos", 0, N - T)
    fn = TinyJit(lambda q, sp: mla_prefill_attention(q, cache, sp, DV, D ** -0.5).realize())
    us = sum(kernel_us(lambda: fn(q, sp.bind(S))).values())
    flops = 2 * H * (D + DV) * sum(S + t + 1 for t in range(T))
    print(f"attn T={T} start_pos={S}: {us / 1e3:.3f} ms, {flops / us / 1e6:.1f} TFLOPS ({flops / us / 1e6 / PEAK * 100:.0f}% of {PEAK})", flush=True)

if __name__ == "__main__":
  st = time.perf_counter()
  if getenv("GEMM", 1): bench_gemm()
  if getenv("ATTN", 1): bench_attn()
  print(f"{time.perf_counter() - st:.1f} s, {Device.DEFAULT}")
