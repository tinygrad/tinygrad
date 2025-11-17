#!/usr/bin/env python3
"""
Proper apples-to-apples comparison at each phase.
Both NEW and OLD modified to do THE SAME WORK at each phase.
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

B, Cin, Cout, H, W = 1, 16, 16, 32, 32

def proper_benchmark(x_np, w_np, ctx_dict, label, warmup=3, runs=10):
    with Context(**ctx_dict):
        x = Tensor(x_np, dtype=dtypes.float32).realize()
        w = Tensor(w_np, dtype=dtypes.float32).realize()

        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        compile_time = time.time() - t0

        result = out.numpy()

        runtime_samples = []
        for _ in range(warmup + runs):
            x = Tensor(x_np, dtype=dtypes.float32).realize()
            w = Tensor(w_np, dtype=dtypes.float32).realize()
            t1 = time.time()
            out = x.conv2d(w, padding=1)
            out.realize()
            runtime_samples.append(time.time() - t1)

        runtime_samples = runtime_samples[warmup:]
        avg_runtime = np.mean(runtime_samples) * 1000

    print(f"{label:<20} | Compile: {compile_time*1000:>7.1f}ms | Runtime: {avg_runtime:>7.1f}ms")
    return compile_time * 1000, avg_runtime, result

np.random.seed(42)
x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

print("="*80)
print("PROPER APPLES-TO-APPLES PHASE COMPARISON")
print("="*80)
print("\nBaseline (no winograd):")
c_base, r_base, base_result = proper_benchmark(x_np, w_np, {'WINO': 0, 'WINO_OLD': 0}, "BASELINE")

phases = [
    ("Phase 1", "XHAT only"),
    ("Phase 2", "XHAT + GHAT"),
    ("Phase 3", "+ Multiply/Reduce/MHAT"),
    ("Phase 4", "+ Output transform (COMPLETE)"),
]

for i, (phase_name, description) in enumerate(phases, 1):
    print(f"\n{'='*80}")
    print(f"{phase_name}: {description}")
    print(f"{'='*80}")

    if i < 4:
        print(f"\n⚠️  You need to modify BOTH rangeify.py AND tensor.py to implement {phase_name}")
        print(f"    Both should do THE SAME WORK (e.g., both stop after XHAT for Phase 1)")
        input(f"\nPress Enter when both implementations are at {phase_name}...")
    else:
        print(f"\n✓ Phase 4 - both should be complete implementations")
        input(f"\nPress Enter when ready to test complete implementations...")

    c_old, r_old, old_result = proper_benchmark(x_np, w_np, {'WINO': 0, 'WINO_OLD': 1}, f"OLD {phase_name}")
    c_new, r_new, new_result = proper_benchmark(x_np, w_np, {'WINO': 1, 'WINO_OLD': 0}, f"NEW {phase_name}")

    # Analysis
    print(f"\n📊 {phase_name} Analysis:")
    print(f"   Runtime: NEW {r_new:.1f}ms vs OLD {r_old:.1f}ms")
    diff_ms = r_new - r_old
    diff_pct = (r_new/r_old - 1) * 100

    if abs(diff_pct) < 2:
        print(f"   ≈ Same ({diff_pct:+.1f}%)")
    elif diff_pct > 0:
        print(f"   ❌ NEW is {diff_pct:.1f}% SLOWER ({diff_ms:+.1f}ms)")
    else:
        print(f"   ✅ NEW is {-diff_pct:.1f}% FASTER ({diff_ms:.1f}ms)")

    # Compile time
    print(f"   Compile: NEW {c_new:.1f}ms vs OLD {c_old:.1f}ms ({(c_new/c_old-1)*100:+.1f}%)")

    # Track jump from previous phase
    if i > 1:
        print(f"\n📈 Jump from Phase {i-1} to Phase {i}:")
        if i == 2:
            # Compare to Phase 1 (need to store previous values)
            pass
        else:
            # We'd need to store previous phase results to show jumps
            pass

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print("\nNow we can see WHERE the performance diverges between NEW and OLD")
print("by comparing the same amount of work at each phase.")
