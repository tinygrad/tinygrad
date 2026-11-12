#!/usr/bin/env python3
"""C9 SEQ_FP8 building blocks (576-dim rows: 512 latent dims e4m3 + 4 scales, 64 rope dims half; see fp8_rows), no model load.
Run on ts5 through gpuq (two cards): DEV=AMD PYTHONPATH=. python3 extra/glm_fp8_c9_check.py
- prefill (mla_sparse_lds): fp8 rows read raw == the same kernel on their half values (fp8_dequant), bit for bit
- decode (mla_sparse split / wmma bodies): fp8 rows close to their half values (other rounding), and vs the exact half rows (the quantization)
- seq_store + seq_gather on 2 devices: the gathered fp8 words == fp8_rows of the gathered half rows (the chunk's own rows too)
"""
import os
import numpy as np
from tinygrad import Tensor, dtypes, UOp
from tinygrad.llm.kernels.amd_glm_dsa import amd_flash_attention_mla_sparse, fp8_rows, fp8_dequant, seq_store, seq_gather, seq_rows, \
  FP8_ROPE_ROW, q4_rows, q4_dequant, Q4_ROPE_ROW

rng = np.random.default_rng(53)
D, V = 576, 512
DEV = os.environ.get("CHECK_DEV", "AMD")
DRY = DEV == "NULL"  # (a dry run on teeny: the graphs and kernels built, no values checked)

def rows(n:int) -> np.ndarray:
  # (latent dims ~ a normed vector with a few outliers, rope dims larger: the shapes of C9's)
  x = rng.standard_normal((n, D), dtype=np.float32)
  x[:, :V] *= 0.3
  x[:, :V] += (rng.random((n, V)) < 0.01) * rng.standard_normal((n, V), dtype=np.float32) * 4
  x[:, V:] *= 3
  return x.astype(np.float16)

def compare(name:str, ref:Tensor, got:Tensor, exact:bool=False, tol:float=0.02) -> None:
  a, b = ref.float().numpy(), got.float().numpy()
  if DRY: return print(f"{name}: built", flush=True)
  assert np.isfinite(b).all(), f"{name}: non-finite output"
  if exact:
    assert np.array_equal(a, b), f"{name}: not bit-equal, max|diff| {np.abs(a-b).max()}"
    print(f"{name}: EQUAL", flush=True)
    return
  rel = float(np.sqrt(np.mean((a-b)**2)) / max(float(np.sqrt(np.mean(a*a))), 1e-12))
  print(f"{name}: RMSE/RMS {rel:.5f} max|diff| {float(np.abs(a-b).max()):.5f}", flush=True)
  assert rel < tol, f"{name}: {rel=} over {tol}"

def lists(T:int, N:int, n_max:int) -> Tensor:
  idx = np.zeros((T, n_max+1), dtype=np.int32)
  for t in range(T):
    c = int(rng.integers(n_max//2, n_max+1)) if t else n_max
    idx[t, :c] = rng.choice(N, c, replace=False)
    idx[t, n_max] = c
  return Tensor(idx, device=DEV).realize()

def main(fmt:str) -> None:
  enc, dec_rows, W = (fp8_rows, fp8_dequant, FP8_ROPE_ROW) if fmt == "fp8" else (q4_rows, q4_dequant, Q4_ROPE_ROW)
  qt = 0.1 if fmt == "fp8" else 0.3
  print(f"=== {fmt} rows ({W} words)", flush=True)
  N, n_max, scale = 4096, 2048, 192**-0.5
  k = Tensor(rows(N), device=DEV).realize()
  k8 = enc(k).contiguous().realize()
  assert k8.shape == (N, W) and k8.dtype == dtypes.uint32
  kd = dec_rows(k8).contiguous().realize()
  # (the format: the rope dims as they are, the latent within e4m3's error of their scale)
  assert DRY or np.array_equal(kd.numpy()[:, V:], k.numpy()[:, V:]), "rope dims"
  compare("dequant vs rows", k[:, :V], kd[:, :V], tol=qt/2)
  # prefill: a chunk of 64 tokens, 16 heads (hl 16 on one device), the values' 512 dims
  T = 64
  q = Tensor((rng.standard_normal((1, 16, T, D), dtype=np.float32)*0.5).astype(np.float16), device=DEV).realize()
  li = lists(T, N, n_max)
  def pf(c:Tensor) -> Tensor: return amd_flash_attention_mla_sparse(q, c.reshape(1, 1, 1, N, -1), li, scale, N, vdims=V).realize()
  ref = pf(kd)
  compare("prefill lds fp8 raw vs dequant half", ref, pf(k8), exact=True)
  compare("prefill lds fp8 vs exact half (quantization)", pf(k), ref, tol=qt)
  # decode: 2 tokens (the split kernel), 32 tokens without vdims (the wmma kernel)
  for Tq in (2, 32):
    qd = q[:, :, :Tq].contiguous().realize()
    ld = lists(Tq, N, n_max)
    def dec(c:Tensor) -> Tensor: return amd_flash_attention_mla_sparse(qd, c.reshape(1, 1, 1, N, -1), ld, scale, N)[..., :V].realize()
    hd = dec(kd)
    compare(f"decode T={Tq} fp8 vs dequant half", hd, dec(k8), tol=0.01)
    compare(f"decode T={Tq} fp8 vs exact half (quantization)", dec(k), hd, tol=qt)
    if Tq == 2: assert DRY or np.abs(amd_flash_attention_mla_sparse(qd, k8.reshape(1, 1, 1, N, -1), ld, scale, N)[..., V:].numpy()).max() == 0
  # seq_store + seq_gather, 2 devices: 3000 positions stored, a chunk of 200 after them
  devs = (f"{DEV}:0", f"{DEV}:1")
  n, P, C, M = 2, 3000, 200, 4096
  full = Tensor(rows(P + C), device=DEV).realize()
  c16 = Tensor.zeros(n*seq_rows(M, n), D, dtype=dtypes.half, device=devs, axis=0).contiguous().realize()
  c8 = Tensor.zeros(n*seq_rows(M, n), W, dtype=dtypes.uint32, device=devs, axis=0).contiguous().realize()
  past = full[:P].pad(((0, 3072 - P), None)).contiguous().shard(devs).realize()
  sv = UOp.variable("start_pos", 0, M)
  c16 = seq_store(c16, past, sv.bind(0), P).realize()
  c8 = seq_store(c8, enc(past).contiguous(), sv.bind(0), P).realize()
  chunk = full[P:].pad(((0, 256 - C), None)).contiguous().shard(devs).realize()
  sv2 = UOp.variable("start_pos", 0, M)
  g16 = seq_gather(c16, chunk, sv2.bind(P), C, M).realize()
  g8 = seq_gather(c8, chunk, sv2.bind(P), C, M).realize()
  # (the first device's copy: half its rows read from the other one)
  w16, w8 = g16.numpy()[:P+C], g8.numpy()[:P+C]
  want = enc(Tensor(w16, device=DEV)).numpy()
  assert DRY or np.array_equal(w8, want), f"gather: {int((w8 != want).any(-1).sum())} rows differ"
  assert DRY or np.array_equal(w16, full.numpy()), "gather half"
  print("seq_gather fp8 == fp8_rows(gather half): EQUAL", flush=True)
  print(f"C9 {fmt} CHECK PASS", flush=True)

if __name__ == "__main__":
  for f in os.environ.get("CHECK_FMT", "fp8,q4").split(","): main(f)
