#!/usr/bin/env python3
"""
one GPU: the sampler's top_k mask (kernels top_k_mask) against numpy's exact k-th largest, on logits like a model's (B rows of V), and its time
in a jit next to the nucleus it precedes (the cost a sampled step pays for its rows' top_k).
env: MOD (amd_qwen), B (54), V (151936), KS (20,1,40,0), REPS (20)
run: DEV=AMD PYTHONPATH=. python3 extra/topk_check.py
"""
import importlib, math, os, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv

K = importlib.import_module(f"tinygrad.llm.kernels.{os.environ.get('MOD', 'amd_qwen')}")
B, V, REPS = getenv("B", 54), getenv("V", 151936), getenv("REPS", 20)
ks = [float(k) for k in str(getenv("KS", "20,1,40,0")).split(",")]
rng = np.random.default_rng(0)
# (logits of a model: a few large, a long tail; ties among the tail as a quantized head gives; rows of every k in KS)
z = (rng.standard_normal((B, V)) * 2).astype(np.float32)
z[:, :32] += 8
z[::3, 100:400] = np.round(z[::3, 100:400], 1)
kk = np.array([[ks[b % len(ks)]] for b in range(B)], np.float32)
zt, kt, pt = Tensor(z).realize(), Tensor(kk).realize(), Tensor(np.full((B, 1), 0.95, np.float32)).realize()
m = K.top_k_mask(zt, V, kt).numpy()
bad = 0
for b in range(B):
  k = int(kk[b, 0])
  exp = np.ones(V, bool) if k < 1 or k >= V else z[b] >= np.sort(z[b])[::-1][k-1]
  bad += int(not np.array_equal(m[b], exp))
print(f"top_k_mask {K.__name__} B={B} V={V} ks={ks}: rows differing from numpy's exact top-k: {bad} (of {B})", flush=True)
nuc = TinyJit(lambda z, p: K.nucleus(z, V, p).realize())
both = TinyJit(lambda z, k, p: (lambda keep: (keep & K.nucleus(keep.where(z, -math.inf), V, p)).realize())(K.top_k_mask(z, V, k)))
for name, f, args in (("nucleus", nuc, (zt, pt)), ("top_k+nucleus", both, (zt, kt, pt))):
  for _ in range(3): f(*args)
  Device[Device.DEFAULT].synchronize()
  st = time.perf_counter()
  for _ in range(REPS): f(*args)
  Device[Device.DEFAULT].synchronize()
  print(f"  {name:14s} {(time.perf_counter()-st)/REPS*1e3:7.3f} ms a call (B={B})", flush=True)
