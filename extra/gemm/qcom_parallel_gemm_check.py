#!/usr/bin/env python3
"""Two-stream, arbitrary-input FP16 GEMM benchmark for Adreno 630.

The output rows are split between two independent KGSL contexts.  Timing is
the wall-clock union of both dispatches; correctness is checked after joining
the two row partitions into one dense C matrix.
"""
from __future__ import annotations

import multiprocessing as mp
import os, time
from typing import Any

import numpy as np


def worker(rank: int, a: np.ndarray, b: np.ndarray, barrier: Any, start_at: Any, conn: Any, iters: int) -> None:
  # Import/open the device after fork so every worker owns an independent KGSL
  # context and command queue.
  from tinygrad import Device, dtypes
  from tinygrad.device import Buffer
  from extra.gemm import qcom_8x4_gemm as q8
  from extra.gemm import qcom_intensity_gemm as q4
  from extra.gemm.ir3asm import get_envelope, inject

  cpus = sorted(os.sched_getaffinity(0))
  os.sched_setaffinity(0, {cpus[rank % len(cpus)]})

  rows, k, n = a.shape[0], a.shape[1], b.shape[1]
  threads, stride = 128, n
  q8.M, q8.N, q8.K, q8.K4 = rows, stride, k, k//4
  dev = Device["QCOM"]
  env, io, sz, ro = get_envelope(dev, q8.make_donor_src8(4, threads))
  shader, hregs, fregs, _ = q8.build_8x8_split_a_unroll_shader(
    dev, threads, k_unroll=2, b_coord_delay=-1, fast_coords=True,
    stream_col1=True, add256_store_mode="tight", a_coord_delay=-1,
    relaxed_sync=True, sync_mask=14, separate_coords=True)
  assert len(shader) <= sz
  lib = inject(env, io, sz, ro, shader, fregs=fregs, hregs=hregs)

  ab = Buffer("QCOM", a.size, dtypes.half).allocate()
  bb = Buffer("QCOM", b.size, dtypes.half).allocate()
  cb = Buffer("QCOM", rows*stride, dtypes.half).allocate()
  ab.copyin(memoryview(np.ascontiguousarray(a)).cast("B"))
  bb.copyin(memoryview(np.ascontiguousarray(b)).cast("B"))
  cb.copyin(memoryview(np.zeros(rows*stride, dtype=np.float16)).cast("B"))
  specs = [((0, dtypes.half, (rows, k//4, 4)),), ((0, dtypes.half, (k, n//4, 4)),),
           ((0, dtypes.half, None),)]
  prg = dev.runtime("gemm_h", lib, buf_dtypes=specs)
  launch = dict(global_size=(n//256, rows//32, 1), local_size=(threads, 1, 1), wait=True)

  for _ in range(5): prg(ab._buf, bb._buf, cb._buf, **launch)
  samples = []
  for _ in range(iters):
    barrier.wait()
    while time.perf_counter() < start_at.value: pass
    start = time.perf_counter()
    event_time = prg(ab._buf, bb._buf, cb._buf, **launch)
    end = time.perf_counter()
    samples.append((start, end, event_time))

  got = np.empty((rows, stride), dtype=np.float16)
  cb.copyout(memoryview(got).cast("B"))
  conn.send((rank, samples, got))
  conn.close()


def main() -> None:
  m, n, k = (int(os.getenv(x, "1024")) for x in ("M", "N", "K"))
  seed, iters = int(os.getenv("SEED", "0")), int(os.getenv("ITERS", "10"))
  if m % 64 or n % 256 or k % 8:
    raise ValueError("parallel tiled kernel requires M%64 == 0, N%256 == 0, K%8 == 0")

  # Independent streams make B generation invariant to the chosen M split.
  a = (np.random.default_rng(seed).standard_normal((m, k))*0.05).astype(np.float16)
  b = (np.random.default_rng(seed+1).standard_normal((k, n))*0.05).astype(np.float16)
  halves = (a[:m//2], a[m//2:])
  barrier = mp.Barrier(3)
  start_at = mp.Value("d", 0.0, lock=False)
  pipes = [mp.Pipe(duplex=False) for _ in range(2)]
  procs = [mp.Process(target=worker, args=(rank, halves[rank], b, barrier, start_at, pipes[rank][1], iters)) for rank in range(2)]
  for proc in procs: proc.start()
  for _, send_conn in pipes: send_conn.close()
  for _ in range(iters):
    start_at.value = time.perf_counter() + 0.01
    barrier.wait()
  results = [pipes[rank][0].recv() for rank in range(2)]
  for proc in procs:
    proc.join()
    if proc.exitcode: raise RuntimeError(f"worker exited with status {proc.exitcode}")
  results.sort(key=lambda x: x[0])

  overlap_times = [max(results[0][1][i][1], results[1][1][i][1]) -
                   min(results[0][1][i][0], results[1][1][i][0]) for i in range(iters)]
  best = min(overlap_times)
  got = np.concatenate((results[0][2], results[1][2]), axis=0).astype(np.float32)
  expected = a.astype(np.float32) @ b.astype(np.float32)
  delta = np.abs(expected-got)
  correct = np.allclose(expected, got, rtol=2e-2, atol=2e-2)
  bad = ~np.isfinite(got) | (delta > 0.02)
  event_ms = [[sample[2]*1e3 for sample in result[1]] for result in results]
  print(f"shape={m}x{n}x{k} streams=2 accumulate=fp16 elapsed_ms={best*1e3:.3f} "
        f"gflops={2*m*n*k/best/1e9:.1f} max_abs={delta.max():.9g} mean_abs={delta.mean():.9g} "
        f"allclose={correct} bad_count={bad.sum()}")
  print(f"worker_event_ms={[round(min(x), 3) for x in event_ms]}")
  if not correct: raise SystemExit(1)


if __name__ == "__main__": main()
