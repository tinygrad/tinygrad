#!/usr/bin/env python3
"""
one GPU: the MoE linears of GLM-5.3's batched decoding (amd_glm_dsa.moe_linear as the model calls it, expert parallel: this device holds
E experts of 256, every row selects 8) on random IQ3_S (gate+up fused, x shared by the slots) / IQ4_XS (down, x a slot's) weights.
Every shape and row count timed for each arm (ARMS: ;-separated NAME:K=V,K=V, the first the reference; a K that is a module global of
amd_glm_dsa (MOE_BUNDLE, MOE_BUNDLE_MIN, MOE_WMMA, ...) is set there too) and its output compared bit for bit with the first arm's.
Bytes: the unique local experts' weights read once (the roofline at BW GB/s) and the selections' (a read per selection).
env: TOKENS (16,32), E (19), REPS (20), ARMS, BW (640), SHAPES (4096x6144xiq3_s,6144x2048xiq4_xs), SEED (0)
run: DEV=AMD PYTHONPATH=. python3 extra/glm53_moe_bench.py
"""
import os, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import getenv
import tinygrad.llm.kernels.amd_glm_dsa as K

TYPES = {"iq3_s": K.IQ3_S, "iq4_xs": K.IQ4_XS, "q8_0": K.Q8_0}

def main():
  dev, REPS, BW, E, SLOTS, NEXP = Device.DEFAULT, getenv("REPS", 20), getenv("BW", 640), getenv("E", 19), 8, 256
  rng = np.random.default_rng(getenv("SEED", 0))
  arms = [a.split(":", 1) for a in os.environ.get("ARMS", "row:MOE_BUNDLE_MIN=100000").split(";") if a]
  toks = [int(t) for t in str(getenv("TOKENS", "16,32")).split(",")]
  tot = {name: 0.0 for name, _ in arms}
  first = Tensor([0], dtype=dtypes.int32, device=dev).realize()
  for shape in os.environ.get("SHAPES", "4096x6144xiq3_s,6144x2048xiq4_xs").split(","):
    fo, fi, tn = shape.split("x", 2)
    fout, fin, typ = int(fo), int(fi), TYPES[tn]
    ebytes = fout * fin * K.QUANT_SIZES[typ] // 256
    raw = rng.integers(0, 2**32, E * ebytes // 4, dtype=np.uint32)
    if typ in (K.IQ4_XS, K.Q8_0, K.IQ3_S):
      # (sane scales: the block's f16 d at its start, small, either sign)
      b = raw.view(np.uint8).reshape(-1, K.QUANT_SIZES[typ] if typ != K.Q8_0 else 34)
      d = np.exp(rng.uniform(np.log(1e-4), np.log(0.05), (b.shape[0], 1))) * rng.choice([-1, 1], (b.shape[0], 1))
      b[:, 0:2] = d.astype(np.float16).view(np.uint8)
    # (copies past the 64 MB infinity cache)
    ws = [Tensor(raw, device=dev).realize() for _ in range(max(1, -(-getenv("WSET_MB", 256)*10**6 // raw.nbytes)))]
    down = tn != "iq3_s"
    for T in toks:
      sel_np = np.stack([rng.choice(NEXP, SLOTS, replace=False) for _ in range(T)]).astype(np.int32).reshape(1, T, SLOTS)
      loc = sel_np[(sel_np >= 0) & (sel_np < E)]
      uniq = len(np.unique(loc))
      sel = Tensor(sel_np, device=dev).realize()
      xn = rng.standard_normal((1, T, SLOTS if down else 1, fin), dtype=np.float32)
      x = Tensor(xn, device=dev).realize()
      ref = None
      line = (f"{tn:7s} {fout}x{fin} T={T:3d} local sel {len(loc):3d} uniq {uniq:2d}: roofline uniq {uniq*ebytes/BW/1e3:6.1f} us, "
              f"per sel {len(loc)*ebytes/BW/1e3:6.1f} us:")
      for name, envs in arms:
        old, oldk = {}, {}
        for kv in [e for e in envs.split(",") if e]:
          k, v = kv.split("=", 1)
          old[k], os.environ[k] = os.environ.get(k), v
          if hasattr(K, k): oldk[k], _ = getattr(K, k), setattr(K, k, type(getattr(K, k))(int(v)))
        getenv.cache_clear()
        K._bundle_cache.clear()
        try:
          f = TinyJit(lambda s, v: [K.moe_linear(w, typ, E, v, s, first).realize() for w in ws][0])
          for _ in range(3): o = f(sel, x)
          Device[dev].synchronize()
          st = time.perf_counter()
          for _ in range(REPS): o = f(sel, x)
          Device[dev].synchronize()
          dt = (time.perf_counter() - st) / REPS / len(ws)
          out = o.float().numpy().reshape(T, SLOTS, fout)
        finally:
          for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
          for k, v in oldk.items(): setattr(K, k, v)
          getenv.cache_clear()
          K._bundle_cache.clear()
        if ref is None: ref, eq = out, "ref"
        elif np.array_equal(out.view(np.uint32), ref.view(np.uint32)): eq = "EQUAL"
        else: eq = f"DIFF {int((out != ref).sum())} max {np.abs(out-ref).max():.3g} (|ref| {np.abs(ref).max():.3g})"
        tot[name] += dt
        line += f"  {name} {dt*1e6:7.1f} us {eq}"
      print(line, flush=True)
  print("TOTAL " + "  ".join(f"{n} {t*1e6:.0f} us" for n, t in tot.items()), flush=True)

if __name__ == "__main__": main()
