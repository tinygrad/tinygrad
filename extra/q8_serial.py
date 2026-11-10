#!/usr/bin/env python3
"""
q8_gemm configs timed SERIALLY (every call after the one before: as in a model step, no overlap of independent kernels, which a jit of L
independent calls has and which flatters small grids), L distinct cold weights, the k-split sum included. us a call (wall of the chain / L).
env: SHAPES=MxNxK,.. CFGS="tt,bn,ks[,pfd];.." L=100
"""
import functools, time
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd_qwen import _q8_gemm_kernel, _wmma_rdna4, Q8_0

dev, rng, L = Device.DEFAULT, np.random.default_rng(0), getenv("L", 100)
rdna4 = _wmma_rdna4(dev)
DEQ:dict[int, np.ndarray] = {}
def deq(w:Tensor) -> np.ndarray: return DEQ[id(w)]
def q8(N:int, K:int) -> Tensor:
  scales = rng.uniform(.001, .01, size=(N, K//32, 1)).astype(np.float16)
  quant = rng.integers(-127, 128, size=(N, K//32, 32), dtype=np.int8)
  t = Tensor(np.concatenate((scales.view(np.uint8), quant.view(np.uint8)), axis=-1).copy().view(np.uint16).ravel(), device=dev).realize()
  DEQ.clear()
  DEQ[id(t)] = (quant.astype(np.float32) * scales.astype(np.float32)).reshape(N, K)
  return t
for shape in getenv("SHAPES", "96x1024x2560").split(","):
  M, N, K = map(int, shape.split("x"))
  ws = [q8(N, K) for _ in range(L)]
  x = Tensor(rng.normal(size=(M, K)).astype(np.float16), device=dev).realize()
  for cfg in getenv("CFGS", "96,64,4").split(";"):
    tt, bn, ks, pfd = (list(map(int, cfg.split(","))) + [0])[:4]
    try:
      fxn = functools.partial(_q8_gemm_kernel, token_tile=tt, ggml_type=Q8_0, block_n=bn, k_splits=ks, rdna4=rdna4, pfd=pfd,
                              out_features=N, in_features=K)
      def chain(x:Tensor) -> Tensor:
        dep = None
        for w in ws:
          xi = Tensor(x.uop.after(dep.uop)) if dep is not None else x
          r = Tensor.custom_kernel(Tensor.empty(ks*M, N, dtype=dtypes.float32, device=dev), w, xi, fxn=fxn)[0]
          dep = (r.reshape(ks, M, N).sum(0) if ks > 1 else r).contiguous()
        return dep.realize()
      run = TinyJit(chain)
      for _ in range(3): run(x)
      Device[dev].synchronize()
      ts = []
      for _ in range(5):
        t0 = time.perf_counter()
        run(x)
        Device[dev].synchronize()
        ts.append(time.perf_counter() - t0)
      t = min(ts) / L * 1e6
      gbs, tfs = N*K*34/32/(t*1e-6)/1e9, 2*M*N*K/(t*1e-6)/1e12
      if getenv("CHECK", 1):
        ref = (x.numpy().astype(np.float32) @ deq(ws[-1]).T)
        got = run(x).numpy()
        cerr = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9))
      else: cerr = 0.0
      print(f"{Device[dev].arch} {shape} {cfg:12s}: {t:7.2f} us serial ({gbs:5.0f} GB/s, {tfs:5.1f} TFLOP/s) err {cerr:.1e}", flush=True)
    except Exception as e: print(f"{shape} {cfg}: {type(e).__name__} {str(e)[:100]}", flush=True)

if getenv("LIN", 0):
  # q8_linear's own choice for the shapes (decode GEMV for <= DECODE_MULTI rows of a long input), serial
  from tinygrad.llm.kernels.amd_qwen import Linear, q8_linear
  for shape in getenv("SHAPES", "96x1024x2560").split(","):
    M, N, K = map(int, shape.split("x"))
    lins = []
    for _ in range(L):
      with Linear.empty_init(): lin = Linear(K, N, bias=False)
      lin.ggml_type, lin.weight = Q8_0, q8(N, K)
      lins.append(lin)
    x = Tensor(rng.normal(size=(M, K)).astype(np.float16), device=dev).realize()
    def lchain(x:Tensor) -> Tensor:
      dep = None
      for lin in lins:
        xi = Tensor(x.uop.after(dep.uop)) if dep is not None else x
        dep = q8_linear(lin, xi).contiguous()
      return dep.realize()
    lrun = TinyJit(lchain)
    for _ in range(3): lrun(x)
    Device[dev].synchronize()
    lts = []
    for _ in range(5):
      t0 = time.perf_counter()
      lrun(x)
      Device[dev].synchronize()
      lts.append(time.perf_counter() - t0)
    lt = min(lts) / L * 1e6
    print(f"{Device[dev].arch} {shape} q8_linear: {lt:7.2f} us serial ({N*K*34/32/(lt*1e-6)/1e9:5.0f} GB/s)", flush=True)
