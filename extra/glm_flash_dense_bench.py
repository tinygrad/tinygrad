#!/usr/bin/env python3
"""
one GPU: amd_glm's Q8_0 (TYPE=q6_k: Q6_K) linears of batched decoding's rows (Q8_ROWS_I8 set, as the model traces them) on random weights, the kernel
as q8_linear picks it. Every shape timed for each arm (ARMS: ;-separated NAME:K=V,K=V, the first the reference) and its output compared bit for
bit with the first arm's. Roofline: the weights' bytes at BW GB/s.
env: SHAPES (in x out, comma separated), ROWS (84), REPS (50), BW (640), WSET_MB (256)
run: DEV=AMD PYTHONPATH=. python3 extra/glm_flash_dense_bench.py
"""
import os, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv
import tinygrad.llm.kernels.amd_glm as K

TYPE = {"q8_0": K.Q8_0, "q6_k": K.Q6_K}[os.environ.get("TYPE", "q8_0")]

def main():
  dev, REPS, BW, T = Device.DEFAULT, getenv("REPS", 50), getenv("BW", 640), getenv("ROWS", 84)
  rng = np.random.default_rng(0)
  arms = [a.split(":", 1) for a in os.environ.get("ARMS", "base:").split(";") if a]
  tot = {name: 0.0 for name, _ in arms}
  K.Q8_ROWS_I8[0] = bool(getenv("I8ROWS", 1))  # (GLM_SEQ: off, its rows padded to 128 by the jit: ROWS=128 I8ROWS=0)
  for shape in os.environ.get("SHAPES", "4096x1536,1536x4096,4096x3072,3072x4096,4096x2048,2048x4096,4096x25824").split(","):
    fin, fout = (int(v) for v in shape.split("x"))
    if TYPE == K.Q6_K:  # ql[128] qh[64] scales[16] (int8) d (f16)
      n = fout * fin // 256
      b = rng.integers(0, 256, (n, 210), dtype=np.uint8)
      b[:, 192:208] = rng.integers(-32, 32, (n, 16), dtype=np.int8).view(np.uint8)
      b[:, 208:210] = np.full((n, 1), 0.002, np.float16).view(np.uint8)
    else:
      n = fout * fin // 32
      b = rng.integers(0, 256, (n, 34), dtype=np.uint8)
      b[:, 0:2] = np.full((n, 1), 0.01, np.float16).view(np.uint8)
    wbytes = b.nbytes
    layers = []
    for _ in range(max(1, -(-getenv("WSET_MB", 256)*10**6 // wbytes))):
      with K.Linear.empty_init(): layer = K.Linear(fin, fout, bias=False)
      layer.ggml_type, layer.shard_axis = TYPE, None
      layer.weight = Tensor(b.reshape(-1).view(np.uint16), device=dev).realize()
      layers.append(layer)
    x = Tensor(rng.standard_normal((T, fin), dtype=np.float32), device=dev).realize()
    ref, line = None, f"{fin}x{fout} rows {T}: roofline {wbytes/BW/1e3:6.1f} us:"
    for name, envs in arms:
      old:dict[str, str|None] = {}
      for kv in [e for e in envs.split(",") if e]:
        k, v = kv.split("=", 1)
        old[k], os.environ[k] = os.environ.get(k), v
      getattr(getenv, 'cache_clear')()
      for fn in vars(K).values():
        if callable(getattr(fn, 'cache_clear', None)) and getattr(fn, '__module__', '') == K.__name__: fn.cache_clear()
      try:
        f = TinyJit(lambda v: [l(v).float().realize() for l in layers][0])
        for _ in range(3): o = f(x)
        Device[dev].synchronize()
        st = time.perf_counter()
        for _ in range(REPS): o = f(x)
        Device[dev].synchronize()
        dt = (time.perf_counter() - st) / REPS / len(layers)
        out = o.numpy()
      finally:
        for ek, ev in old.items():
          if ev is None: os.environ.pop(ek, None)
          else: os.environ[ek] = ev
        getattr(getenv, 'cache_clear')()
      if ref is None: ref, eq = out, "ref"
      elif np.array_equal(out.view(np.uint32), ref.view(np.uint32)): eq = "EQUAL"
      else: eq = f"DIFF {int((out != ref).sum())} rel {np.abs(out-ref).max()/max(np.abs(ref).max(), 1e-30):.2g}"
      tot[name] += dt
      line += f"  {name} {dt*1e6:7.1f} us {eq}"
    print(line, flush=True)
  print("TOTAL " + "  ".join(f"{n} {t*1e6:.0f} us" for n, t in tot.items()), flush=True)

if __name__ == "__main__": main()
