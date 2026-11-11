#!/usr/bin/env python3
"""SEQ_KV4 building blocks: q4_rows' layout (numpy decode, any device) and the MLA sparse attention on 4-bit rows (an AMD GPU): against the
half attention of the decoded rows (layout: ~exact) and of the original rows (quantization error, fp8's for scale).
Run: DEV=CPU PYTHONPATH=. python3 extra/glm_kv4_check.py (layout only) | DEV=AMD ... (both)
"""
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.llm.kernels.amd_glm import amd_flash_attention_mla_sparse, fp8_rows, q4_rows, Q4_ROW

rng = np.random.default_rng(29)

def q4_decode(w:np.ndarray) -> np.ndarray:
  """(R, 72) uint32 words -> (R, 512) float32"""
  R = w.shape[0]
  vals = ((w[:, :64, None] >> (4*np.arange(8, dtype=np.uint32))) & 15).astype(np.int32).reshape(R, 512) - 8
  mult = ((w[:, 64:68, None] >> (8*np.arange(4, dtype=np.uint32))) & 255).astype(np.float32).reshape(R, 16)
  sc = w[:, 68:72].copy().view(np.float32)
  return vals.reshape(R, 16, 32) * mult[:, :, None] * np.repeat(sc, 4, axis=1)[:, :, None]

def stats(name:str, a:np.ndarray, b:np.ndarray, lim:float) -> None:
  rel = float(np.sqrt(np.mean((a-b)**2)) / max(float(np.sqrt(np.mean(a*a))), 1e-12))
  print(f"{name}: RMSE/RMS {rel:.5f} (limit {lim})", flush=True)
  assert np.isfinite(b).all() and rel < lim, name

def main() -> None:
  N = 256
  x = (rng.standard_normal((N, 512), dtype=np.float32) * 0.1 * np.exp(rng.standard_normal((N, 1)))).astype(np.float16)
  x[3, :] = 0  # (a zero row)
  w = q4_rows(Tensor(x)).numpy()
  assert w.shape == (N, Q4_ROW), w.shape
  dec = q4_decode(w).reshape(N, 512)
  stats("q4_rows decode vs original", x.astype(np.float32), dec, 0.12)
  m = ((w[:, 64:68, None] >> (8*np.arange(4, dtype=np.uint32))) & 255)
  assert m.min() >= 1 and m.max() <= 63, (m.min(), m.max())
  if "AMD" not in Device.DEFAULT:
    print("KV4 LAYOUT PASS (no AMD device: attention skipped)")
    return
  mq = Tensor((rng.standard_normal((1, 16, 1, 512), dtype=np.float32)*0.1).astype(np.float16)).realize()
  index = np.arange(N+1, dtype=np.int32).reshape(1, N+1)
  index[0, :N], index[0, N] = rng.permutation(N), N
  ti = Tensor(index).realize()
  def att(c:Tensor) -> np.ndarray: return amd_flash_attention_mla_sparse(mq, c, ti, 512**-0.5, N).numpy().astype(np.float32)
  ref, refd = att(Tensor(x).realize()), att(Tensor(dec.astype(np.float16)).realize())
  got4, got8 = att(Tensor(w).realize()), att(fp8_rows(Tensor(x)).contiguous().realize())
  stats("MLA 4-bit vs half of the decoded rows (layout)", refd, got4, 0.01)
  stats("MLA fp8 vs half (for scale)", ref, got8, 0.1)
  stats("MLA 4-bit vs half (quantization)", ref, got4, 0.25)
  print("KV4 CHECK PASS", flush=True)

if __name__ == "__main__": main()
