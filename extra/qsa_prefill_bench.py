#!/usr/bin/env python3
"""
the prefill rows' sparse attention (qsa_decode_attention on compact pool lists) on ONE device: B rows (consecutive positions at the end of a
CTX-token context), H heads, every row its TOP picked pools of kpool 4 (a random NEAR-share of them common to a block of 16 neighbouring rows,
the rest random) + its tail. timed serially (a chain), PARTS/WAVES configs. env: B=576 H=12 CTX=100000 TOP=171 L=12 CFGS="1,1;1,4;2,4"
"""
import time
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv
from tinygrad.llm.kernels import amd_qwen as aq
from tinygrad.llm.kernels.amd_qwen import qsa_decode_attention, kv_q4_pack

B, H, CTX, TOP, L, KP, HD = getenv("B", 576), getenv("H", 12), getenv("CTX", 100000), getenv("TOP", 171), getenv("L", 12), 4, 256
NEAR = getenv("NEAR", 0.8)
rng = np.random.default_rng(0)
dev = Device.DEFAULT
N = CTX + B + 64
caches = [kv_q4_pack(Tensor(rng.normal(size=(N, 2*HD)).astype(np.float16))).realize() for _ in range(L)]
q = Tensor(rng.normal(size=(B, H, HD)).astype(np.float16)).realize()
pos = np.arange(CTX, CTX + B)
idx = np.zeros((B, TOP + 2), dtype=np.int32)
npools = CTX // KP
for blk in range(0, B, 16):
  common = rng.choice(npools, size=int(TOP * NEAR), replace=False)
  for r in range(blk, min(B, blk + 16)):
    own = rng.choice(npools, size=TOP - len(common), replace=False)
    picks = np.unique(np.concatenate([common, own]))[:TOP]
    picks = np.concatenate([picks, rng.choice(npools, size=TOP - len(picks))]) if len(picks) < TOP else picks
    idx[r, :TOP], idx[r, TOP], idx[r, TOP + 1] = np.sort(picks), TOP, pos[r]
index = Tensor(idx).realize()
bases = Tensor(np.zeros(B, dtype=np.int32)).realize()
outs = {}
for cfg in getenv("CFGS", "0,0,0;0,0,1").split(";"):
  parts, waves, aq.QSA_PREFILL = map(int, cfg.split(","))
  def chain(q:Tensor) -> Tensor:
    dep = None
    for c in caches:
      qi = Tensor(q.uop.after(dep.uop)) if dep is not None else q
      dep = qsa_decode_attention(qi, c.reshape(1, N, -1), index, bases, HD**-0.5, parts=parts, waves=waves, kpool=KP, bits=4).contiguous()
    return dep.realize()
  run = TinyJit(chain)
  for _ in range(3): run(q)
  Device[dev].synchronize()
  ts = []
  for _ in range(5):
    t0 = time.perf_counter()
    run(q)
    Device[dev].synchronize()
    ts.append(time.perf_counter() - t0)
  outs[cfg] = qsa_decode_attention(q, caches[0].reshape(1, N, -1), index, bases, HD**-0.5, parts=parts, waves=waves, kpool=KP, bits=4).numpy()
  ref = next(iter(outs.values()))
  if getenv("REF", 1) and "exact" not in outs:
    # the exact attention of a few rows (float64) on the cache's values (n - 8) * scale
    cw = caches[0].numpy().reshape(N, -1)
    words = cw[:, :2*HD//8]
    vals = np.zeros((N, 2*HD), dtype=np.float64)
    for j in range(8):
      vals[:, j::8] = ((words >> ((j % 4) * 8 + (j // 4) * 4)) & 15).astype(np.float64) - 8
    sc = cw[:, 2*HD//8:].view(np.float16).astype(np.float64).reshape(N, -1)
    vals = vals * np.repeat(sc, 32, axis=1)
    rows = getenv("REFROWS", 32)
    ex = np.zeros((rows, H, HD))
    qn = q.numpy().astype(np.float64)
    for r in range(rows):
      npk, qp = idx[r, TOP], idx[r, TOP + 1]
      tail0 = (qp + 1) // KP * KP
      keys = np.concatenate([(idx[r, :npk, None] * KP + np.arange(KP)).ravel(), np.arange(tail0, qp + 1)])
      kv = vals[keys]
      s_ = qn[r] @ kv[:, :HD].T * HD**-0.5
      p_ = np.exp(s_ - s_.max(-1, keepdims=True))
      ex[r] = (p_ / p_.sum(-1, keepdims=True)) @ kv[:, HD:]
    outs["exact"] = ex
  if "exact" in outs:
    e_ = outs["exact"]
    got = outs[cfg].reshape(B, H, HD)[:e_.shape[0]]
    err_ = np.abs(got - e_).max()
    print(f"  {cfg} vs exact (first {e_.shape[0]} rows): max abs err {err_:.3e}, rel to max {err_ / np.abs(e_).max():.3e}")
  err = float(np.abs(outs[cfg] - ref).max() / (np.abs(ref).max() + 1e-9))
  print(f"{Device[dev].arch} qsa B={B} H={H} top={TOP*KP} keys near={NEAR} parts={parts} waves={waves} prefill={aq.QSA_PREFILL}: "
        f"{min(ts)/L*1e6:8.1f} us  max err vs first {err:.2e}", flush=True)
