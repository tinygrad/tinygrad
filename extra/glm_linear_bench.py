#!/usr/bin/env python3
"""
one GPU: the time of amd_glm's q8_linear of random Q6_K weights (or Q8_0, TYPE=q8_0) for prefill-sized inputs, the kernel as the model picks
it (env flags pick the path, e.g. Q6_PREFILL=1). OUT=<file>.npz saves the outputs, REF=<file>.npz compares with a saved run (max abs diff).
env: SHAPES (in x out, comma separated: 4096x8192,8192x4096,4096x2048,2048x4096), TOKENS (4096), REPS (20)
run: DEV=PCI:0+AMD (or AMD) PYTHONPATH=. python3 extra/glm_linear_bench.py
"""
import os, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd_glm import Linear, Q6_K, Q8_0, QUANT_SIZES

TYPE = {"q6_k": Q6_K, "q8_0": Q8_0}[os.environ.get("TYPE", "q6_k")]
rng = np.random.default_rng(0)

def blocks(rows:int, cols:int) -> np.ndarray:
  n = rows * cols // 256
  if TYPE == Q6_K:  # ql[128] qh[64] scales[16] (int8) d (f16)
    b = rng.integers(0, 256, (n, 210), dtype=np.uint8)
    b[:, 192:208] = rng.integers(-32, 32, (n, 16), dtype=np.int8).view(np.uint8)
    b[:, 208:210] = np.full((n, 1), 0.002, np.float16).view(np.uint8)
  else:  # 8 x (d (f16), qs[32] int8)
    b = rng.integers(0, 256, (n, 8, 34), dtype=np.uint8)
    b[:, :, 0:2] = np.full((n, 8, 1), 0.01, np.float16).view(np.uint8)
    b = b.reshape(n, 272)
  assert b.shape[1] == QUANT_SIZES[TYPE]
  return b

T, REPS = getenv("TOKENS", 4096), getenv("REPS", 20)
dev = Device.DEFAULT
outs, ref = {}, (dict(np.load(r)) if (r:=os.environ.get("REF")) else None)
for shape in os.environ.get("SHAPES", "4096x8192,8192x4096,4096x2048,2048x4096").split(","):
  fin, fout = (int(v) for v in shape.split("x"))
  with Linear.empty_init(): layer = Linear(fin, fout, bias=False)
  layer.ggml_type, layer.shard_axis = TYPE, None
  layer.weight = Tensor(blocks(fout, fin).reshape(-1).view(np.uint16), device=dev).realize()
  x = Tensor(rng.standard_normal((T, fin), dtype=np.float32).astype(np.float16), device=dev).realize()
  f = TinyJit(lambda v: layer(v).float().realize())
  for _ in range(3): o = f(x)
  Device[dev].synchronize()
  st = time.perf_counter()
  for _ in range(REPS): o = f(x)
  Device[dev].synchronize()
  dt = (time.perf_counter() - st) / REPS
  outs[shape] = o.numpy()
  extra = f"  max|diff| vs REF {np.abs(outs[shape] - ref[shape]).max():.4g} (|ref| max {np.abs(ref[shape]).max():.4g})" if ref else ""
  print(f"{shape} x {T} tokens: {dt*1e3:.3f} ms  {2*T*fin*fout/dt/1e12:.1f} TFLOP/s{extra}", flush=True)
if (o:=os.environ.get("OUT")): np.savez(o, **outs)
