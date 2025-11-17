#!/usr/bin/env python3
"""Test Phase 3 runtime - THE CRITICAL PHASE"""
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

B, Cin, Cout, H, W = 1, 16, 16, 32, 32

def bench_runtime(x, w, pad, ctx_dict, label, warmup=3, runs=10):
    with Context(**ctx_dict):
        t0 = time.time()
        out = x.conv2d(w, padding=pad)
        out.realize()
        compile_time = time.time() - t0

        # Warmup
        for _ in range(warmup):
            _ = out.numpy()

        # Timed runs
        times = []
        for _ in range(runs):
            t1 = time.time()
            _ = out.numpy()
            times.append(time.time() - t1)

        avg_runtime = np.mean(times) * 1000  # ms
        std_runtime = np.std(times) * 1000   # ms

    print(f"{label:<30} Compile: {compile_time*1000:>7.1f}ms | Runtime: {avg_runtime:>7.3f}ms ± {std_runtime:.3f}ms")
    return compile_time * 1000, avg_runtime

print("="*80)
print("PHASE 3: XHAT * GHAT + REDUCE (MHAT) - EXPECTED BOTTLENECK!")
print("="*80)

np.random.seed(42)
x = Tensor.randn(B, Cin, H, W, dtype=dtypes.float32).realize()
w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()

c_base, r_base = bench_runtime(x, w, 1, {'WINO': 0, 'WINO_OLD': 0}, "BASELINE (no wino)")
c_old, r_old = bench_runtime(x, w, 1, {'WINO': 0, 'WINO_OLD': 1}, "OLD (complete)")
c_new, r_new = bench_runtime(x, w, 1, {'WINO': 1, 'WINO_OLD': 0}, "PHASE 3 NEW (+ MHAT)")

print("\n" + "="*80)
print("PHASE 3 ANALYSIS - THE CRITICAL PHASE")
print("="*80)
print(f"Baseline runtime: {r_base:.3f}ms")
print(f"OLD full runtime: {r_old:.3f}ms ({r_old/r_base:.2f}× vs baseline)")
print(f"NEW Phase 3 runtime: {r_new:.3f}ms ({r_new/r_base:.2f}× vs baseline)")
print(f"\nNEW vs OLD: {r_new:.3f}ms vs {r_old:.3f}ms")
if r_new > r_old:
    print(f"❌ NEW Phase 3 is {(r_new-r_old)/r_old*100:.1f}% SLOWER than complete OLD ({r_new/r_old:.2f}×) ⚠️ BOTTLENECK!")
else:
    print(f"✓ NEW Phase 3 is {(r_old-r_new)/r_new*100:.1f}% FASTER than complete OLD ({r_old/r_new:.2f}×)")

print("\nPrevious findings: Phase 3 was 40× slower (16.3ms vs 0.4ms)")
print("Let's see if we can reproduce this...")
