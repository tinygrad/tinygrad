#!/usr/bin/env python3
"""
one GPU: the dense linears of GLM-5.3's batched decoding (a few rows, f16 activations: amd_glm_dsa.q8_linear as the model calls it) on
random Q8_0 / Q6_K weights. Every shape timed for each arm (ARMS: ;-separated env sets, the first the reference) and its output compared
bit for bit with the first arm's. Bytes: the weights read once (the roofline at BW GB/s).
the time a call: one jit of the layer's copies (WSET_MB of weights, past the infinity cache), launch included.
env: SHAPES (out x in [x type], comma separated), TOKENS (16,32), REPS (50), ARMS ("base:" = no env), BW (640)
run: DEV=AMD PYTHONPATH=. python3 extra/glm53_rows_bench.py
"""
import os, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv
import tinygrad.llm.kernels.amd_glm_dsa as K

SHAPES = os.environ.get("SHAPES", "2048x6144,576x6144,128x6144,160x6144,896x6144,1280x2048,4096x2048,6144x160,6144x896,6144x1280,"
                                  "6144x12288,11072x6144xq6")
rng = np.random.default_rng(0)

def blocks(rows:int, cols:int, typ:int) -> np.ndarray:
  n = rows * cols // 256
  if typ == K.Q6_K:  # ql[128] qh[64] scales[16] (int8) d (f16)
    b = rng.integers(0, 256, (n, 210), dtype=np.uint8)
    b[:, 192:208] = rng.integers(-32, 32, (n, 16), dtype=np.int8).view(np.uint8)
    b[:, 208:210] = np.full((n, 1), 0.002, np.float16).view(np.uint8)
  else:  # 8 x (d (f16), qs[32] int8)
    b = rng.integers(0, 256, (n, 8, 34), dtype=np.uint8)
    # (scales log-uniform 1e-7..0.05, either sign: products in half's subnormal range too)
    b[:, :, 0:2] = (np.exp(rng.uniform(np.log(1e-7), np.log(0.05), (n, 8, 1))) * rng.choice([-1, 1], (n, 8, 1))).astype(np.float16).view(np.uint8)
    b = b.reshape(n, 272)
  assert b.shape[1] == K.QUANT_SIZES[typ]
  return b

def main():
  dev, REPS, BW = Device.DEFAULT, getenv("REPS", 50), getenv("BW", 640)
  arms = [a.split(":", 1) for a in os.environ.get("ARMS", "base:").split(";") if a]
  toks = [int(t) for t in str(getenv("TOKENS", "16,32")).split(",")]
  tot = {name: 0.0 for name, _ in arms}
  for shape in SHAPES.split(","):
    p = shape.split("x")
    fout, fin, typ = int(p[0]), int(p[1]), K.Q6_K if len(p) > 2 and p[2] == "q6" else K.Q8_0
    raw = blocks(fout, fin, typ)
    nbytes = raw.size
    # (copies of the layer, a working set past the 64 MB infinity cache: the weights come from memory as in the model's step)
    layers = []
    for _ in range(min(64, max(1, -(-getenv("WSET_MB", 256)*10**6 // nbytes)))):
      with K.Linear.empty_init(): layer = K.Linear(fin, fout, bias=False)
      layer.ggml_type, layer.shard_axis = typ, None
      layer.weight = Tensor(raw.reshape(-1).view(np.uint16), device=dev).realize()
      layers.append(layer)
    for T in toks:
      x = Tensor(rng.standard_normal((T, fin), dtype=np.float32), device=dev).realize()
      ref, line = None, f"{fout:6d}x{fin:<6d}{'q6' if typ == K.Q6_K else 'q8'} T={T:3d} {nbytes/1e6:6.2f} MB (roofline {nbytes/BW/1e3:6.1f} us):"
      for name, envs in arms:
        old = {}
        for kv in [e for e in envs.split(",") if e]:
          k, v = kv.split("=", 1)
          old[k], os.environ[k] = os.environ.get(k), v
        getenv.cache_clear()
        try:
          f = TinyJit(lambda v: [K.q8_linear(lay, v).realize() for lay in layers][0])
          for _ in range(3): o = f(x)
          Device[dev].synchronize()
          st = time.perf_counter()
          for _ in range(REPS): o = f(x)
          Device[dev].synchronize()
          dt = (time.perf_counter() - st) / REPS / len(layers)
          out = o.numpy()
        finally:
          for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
          getenv.cache_clear()
        if ref is None: ref, eq = out, "ref"
        elif np.array_equal(out.view(np.uint32), ref.view(np.uint32)): eq = "EQUAL"
        else: eq = f"DIFF max {np.abs(out-ref).max():.3g} (|ref| {np.abs(ref).max():.3g})"
        tot[name] += dt
        line += f"  {name} {dt*1e6:7.1f} us {nbytes/dt/1e9:5.0f} GB/s {eq}"
      print(line, flush=True)
  print("TOTAL " + "  ".join(f"{n} {t*1e6:.0f} us" for n, t in tot.items()), flush=True)

if __name__ == "__main__": main()
