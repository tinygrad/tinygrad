#!/usr/bin/env python3
"""
Runtime-focused investigation of winograd performance.
Measure actual execution time, not just compile time.
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

# Test shape - same as before
B, Cin, Cout, H, W = 1, 16, 16, 32, 32

def bench_runtime(x, w, pad, ctx_dict, label, warmup=3, runs=10):
    """Benchmark with warmup and multiple runs for accurate runtime measurement"""
    with Context(**ctx_dict):
        # Compile
        t0 = time.time()
        out = x.conv2d(w, padding=pad)
        out.realize()
        compile_time = time.time() - t0

        # Warmup runs
        for _ in range(warmup):
            result = out.numpy()

        # Timed runs
        times = []
        for _ in range(runs):
            t1 = time.time()
            result = out.numpy()
            times.append(time.time() - t1)

        avg_runtime = np.mean(times)
        std_runtime = np.std(times)

    print(f"{label:<30} Compile: {compile_time*1000:>7.1f}ms | "
          f"Runtime: {avg_runtime*1000:>7.3f}ms ± {std_runtime*1000:.3f}ms")
    return compile_time, avg_runtime, result

print("="*80)
print(f"Winograd RUNTIME Investigation - Shape: B={B} Cin={Cin} Cout={Cout} H={H}x{W}")
print("="*80)

# Create test data
np.random.seed(42)
x = Tensor.randn(B, Cin, H, W, dtype=dtypes.float32).realize()
w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()

print("\n" + "-"*80)
print("PHASE-BY-PHASE RUNTIME ANALYSIS")
print("-"*80)

print("\nBASELINE: No winograd")
c_base, r_base, base_result = bench_runtime(x, w, 1, {'WINO': 0, 'WINO_OLD': 0}, "BASELINE")

print("\n" + "="*80)
print("IMPORTANT: Modify rangeify.py winowrite() between each phase as instructed")
print("="*80)

input("\nPHASE 1: XHAT only - Press Enter when ready...")
print("\nPhase 1 should return:")
print("  return XHAT.index(*other_loops_x, *other_loops_w, *[ox//4 for ox in o_axes], *[ox%4 for ox in o_axes])")
c_new, r_new, new_result = bench_runtime(x, w, 1, {'WINO': 1, 'WINO_OLD': 0}, "PHASE 1 NEW (XHAT only)")
c_old, r_old, old_result = bench_runtime(x, w, 1, {'WINO': 0, 'WINO_OLD': 1}, "PHASE 1 OLD (XHAT only)")
print(f"Runtime comparison: NEW {r_new/r_old:.2f}× vs OLD")
if r_new > r_old:
    print(f"❌ NEW is {(r_new-r_old)/r_old*100:.1f}% SLOWER runtime")
else:
    print(f"✓ NEW is {(r_old-r_new)/r_new*100:.1f}% FASTER runtime")

input("\nPHASE 2: XHAT + GHAT - Press Enter when ready...")
print("\nPhase 2 should return XHAT with same indexing")
c_new, r_new, new_result = bench_runtime(x, w, 1, {'WINO': 1, 'WINO_OLD': 0}, "PHASE 2 NEW (XHAT+GHAT)")
c_old, r_old, old_result = bench_runtime(x, w, 1, {'WINO': 0, 'WINO_OLD': 1}, "PHASE 2 OLD (XHAT+GHAT)")
print(f"Runtime comparison: NEW {r_new/r_old:.2f}× vs OLD")
if r_new > r_old:
    print(f"❌ NEW is {(r_new-r_old)/r_old*100:.1f}% SLOWER runtime")
else:
    print(f"✓ NEW is {(r_old-r_new)/r_new*100:.1f}% FASTER runtime")

input("\nPHASE 3: XHAT * GHAT + REDUCE (MHAT) - Press Enter when ready...")
print("\nPhase 3 should return:")
print("  return MHAT.index(*other_loop_ranges_xhat, *other_loop_ranges_ghat, *tile_ranges1, *inner6_1)")
c_new, r_new, new_result = bench_runtime(x, w, 1, {'WINO': 1, 'WINO_OLD': 0}, "PHASE 3 NEW (+ MHAT)")
c_old, r_old, old_result = bench_runtime(x, w, 1, {'WINO': 0, 'WINO_OLD': 1}, "PHASE 3 OLD (+ MHAT)")
print(f"Runtime comparison: NEW {r_new/r_old:.2f}× vs OLD")
if r_new > r_old:
    print(f"❌ NEW is {(r_new-r_old)/r_old*100:.1f}% SLOWER runtime ⚠️ BOTTLENECK?")
else:
    print(f"✓ NEW is {(r_old-r_new)/r_new*100:.1f}% FASTER runtime")

input("\nPHASE 4: Full algorithm (+ output transform) - Press Enter when ready...")
print("\nPhase 4 is the complete implementation")
c_new, r_new, new_result = bench_runtime(x, w, 1, {'WINO': 1, 'WINO_OLD': 0}, "PHASE 4 NEW (Full)")
c_old, r_old, old_result = bench_runtime(x, w, 1, {'WINO': 0, 'WINO_OLD': 1}, "PHASE 4 OLD (Full)")
print(f"Runtime comparison: NEW {r_new/r_old:.2f}× vs OLD")
if r_new > r_old:
    print(f"❌ NEW is {(r_new-r_old)/r_old*100:.1f}% SLOWER runtime")
else:
    print(f"✓ NEW is {(r_old-r_new)/r_new*100:.1f}% FASTER runtime")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print("\nKey question: Which phase shows the biggest runtime degradation?")
print("Previous findings showed Phase 3 had 40× slower runtime, but Phase 4 recovered!")
