#!/usr/bin/env python3
"""
q8 gemm parameter sweep on one GPU: out (M, N) = x (M, K) f16 @ W (N, K)^T Q8_0, L weights of their own (not in the caches), the k split's
partial sums summed after (as q8_linear). checks the result against numpy.
env: SHAPES="32x320x10240,96x1024x2560" (MxNxK), CFGS="tt,bn,ks,gs;..." (gs 0: the kernel's choice), L=40, HC=1 (the up projection's mix:
N is the output dim, the weight 4N rows)
"""
import time, functools
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd_qwen import _q8_gemm_kernel, _wmma_rdna4, Q8_0

L = getenv("L", 40)
rng = np.random.default_rng(0)
dev = Device.DEFAULT
rdna4 = _wmma_rdna4(dev)

TILED = getenv("TILED", "")
def q8(N:int, K:int) -> tuple[Tensor, np.ndarray]:
  scales = rng.uniform(.001, .01, size=(N, K//32, 1)).astype(np.float16)
  quant = rng.integers(-127, 128, size=(N, K//32, 32), dtype=np.int8)
  blocks = np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1)  # (N, K//32, 34)
  # TILED=NB,GS: tile-major (wtile): (N//NB, G//GS, GS, NB, 34)
  if TILED:
    nb, gs_ = map(int, str(TILED).split(","))
    blocks = blocks.reshape(N//nb, nb, K//32//gs_, gs_, 34).transpose(0, 2, 3, 1, 4)
  raw = blocks.copy().view(np.uint16).ravel()
  return Tensor(raw, device=dev).realize(), (quant.astype(np.float32)*scales.astype(np.float32)).astype(np.float16).reshape(N, K).astype(np.float32)

def sync(): Device[dev].synchronize()
import kprof
LAST:list[str] = []
def bench(fn, *args) -> float:
  # GPU time a call (the kernels' sum, PROFILE=1), else wall time
  for _ in range(3): fn(*args)
  kprof.reset(dev)
  st = time.perf_counter()
  for _ in range(10): fn(*args)
  sync()
  if (ts:=kprof.times(dev)):
    if getenv("KVERBOSE"): print({n: (c, round(t/c, 2)) for n, (c, t) in ts.items()})
    LAST[:] = [f"{n.split('_')[0] if n.startswith('q8') else n}={t/c:.1f}" for n, (c, t) in ts.items() if c >= 10*L]
    return sum(t for _, t in ts.values()) / 10 / L
  return (time.perf_counter() - st) / 10 / L * 1e6

for shape in getenv("SHAPES", "32x320x10240").split(","):
  M, N, K = map(int, shape.split("x"))
  hc = getenv("HC", 0)
  ws = [q8(4*N if hc else N, K) for _ in range(L)]
  x = Tensor(rng.normal(size=(M, K)).astype(np.float16), device=dev).realize()
  xn = Tensor(rng.normal(size=(M, 4*N)).astype(np.float32), device=dev).realize() if hc else None
  for cfg in getenv("CFGS", "32,64,32,0").split(";"):
    if getenv("NOK", 1) and False: pass
    try:
      rows = 0
      if False: pass
      else:
        tt, bn, ks, rows, pfd = (list(map(int, cfg.split(","))) + [0, 0])[:5]
        fxn = functools.partial(_q8_gemm_kernel, token_tile=tt, ggml_type=Q8_0, block_n=bn, k_splits=ks, rdna4=rdna4,
                                rows=rows, wtile=bool(TILED), pfd=pfd, out_features=N, in_features=K)
      def one(w, x, xn):
        out = Tensor.empty(ks*M, N, dtype=dtypes.float32, device=dev)
        r = Tensor.custom_kernel(out, w, x, *([xn] if hc else []), fxn=fxn)[0]
        return r.reshape(ks, M, N).sum(0) if ks > 1 else r
      @TinyJit
      def run(x, xn): return Tensor.realize(*[one(w, x, xn) for w, _ in ws])
      t = bench(run, x, xn)
      parts = " ".join(LAST)
      got = run(x, xn).numpy()
      ref = x.numpy().astype(np.float32) @ ws[0][1].T
      if hc:
        g = 1 / (1 + np.exp(-ref))
        ref = (g * xn.numpy()).reshape(M, 4, N).mean(1)
      nr = locals().get("rows") or M  # (rows: the real ones compared)
      err = float(np.abs(got[:nr] - ref[:nr]).max() / (np.abs(ref[:nr]).max() + 1e-9))
      gbs = (ws[0][0].nbytes()) / (t * 1e-6) / 1e9
      print(f"{Device[dev].arch} {M}x{N}x{K}{' hc' if hc else ''} {cfg:18s}: {t:6.1f} us ({gbs:5.0f} GB/s)  [{parts}]  err {err:.1e}", flush=True)
    except Exception as e: print(f"{M}x{N}x{K} {cfg}: {type(e).__name__} {str(e)[:120]}", flush=True)

if getenv("LIN", 0):
  # q8_linear's own choice (decode kernel etc) for the shapes, as the model calls it (one device)
  from tinygrad.llm.kernels.amd_qwen import Linear, q8_linear
  for shape in getenv("SHAPES", "32x320x10240").split(","):
    M, N, K = map(int, shape.split("x"))
    lins = []
    for w, _ in [q8(N, K) for _ in range(L)]:
      with Linear.empty_init(): l = Linear(K, N, bias=False)
      l.ggml_type, l.weight = Q8_0, w
      lins.append(l)
    x = Tensor(rng.normal(size=(M, K)).astype(np.float16), device=dev).realize()
    @TinyJit
    def runl(x): return Tensor.realize(*[q8_linear(l, x) for l in lins])
    t = bench(runl, x)
    print(f"{Device[dev].arch} {M}x{N}x{K} q8_linear: {t:6.1f} us ({lins[0].weight.nbytes()/(t*1e-6)/1e9:5.0f} GB/s)", flush=True)
