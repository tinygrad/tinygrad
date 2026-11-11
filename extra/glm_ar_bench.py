#!/usr/bin/env python3
"""
the batched decoding allreduce of amd_glm alone (no model): partial_sum of (devices, ROWS, DIM) floats, the time a call (GPU, jitted, REPS calls a
run) and the bytes a device sends. What the model's step profile calls "exchanges" is this plus every wait for the slowest device's work before it:
the difference is the step's imbalance.
env: ROWS (108), DIM (4096), REPS (50); the model's flags (FLAG_AR_MAXN, P2P_AR_PACK, ...) as the harness sets them
run: DEV=PCI:0-5+AMD VFIO=1 PYTHONPATH=. python3 extra/glm_ar_bench.py
"""
import time
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv
import tinygrad.llm.kernels.amd_glm as K

def main():
  devs = tuple(Device.DEFAULT if i == 0 else f"{Device.DEFAULT}:{i}" for i in range(getenv("NDEV", 6)))
  ROWS, DIM, REPS, C = getenv("ROWS", 108), getenv("DIM", 4096), getenv("REPS", 20), getenv("CALLS", 20)
  for rows in [ROWS] + [int(r) for r in str(getenv("MORE", "")).split(",") if r]:
    xs = [Tensor.randn(len(devs), rows, DIM).shard(devs, axis=0).realize() for _ in range(C)]
    ref = xs[-1].to("CPU").sum(0).numpy() if getenv("CHECK", 1) else None
    @TinyJit
    def f(*xs:Tensor) -> list[Tensor]: return [K.partial_sum(x).contiguous().realize() for x in xs]
    for _ in range(3): out = f(*xs)[-1]
    if ref is not None:
      got = out.to("CPU").numpy() if not isinstance(out.device, tuple) else Tensor(out.uop.mselect(0)).to("CPU").numpy()
      err = abs(got - ref).max() / (abs(ref).max() + 1e-9)
    else: err = float("nan")
    for d in devs: Device[d].synchronize()
    st = time.perf_counter()
    for _ in range(REPS): f(*xs)
    for d in devs: Device[d].synchronize()
    us = (time.perf_counter() - st) / (REPS * C) * 1e6
    print(f"rows={rows} dim={DIM}: {us:.1f} us a call ({REPS} runs of {C} in a jit), max rel err {err:.2e}", flush=True)

if __name__ == "__main__": main()
