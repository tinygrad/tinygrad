#!/usr/bin/env python3
"""Test single phase runtime with proper measurement"""
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time
import sys

B, Cin, Cout, H, W = 1, 16, 16, 32, 32

def proper_benchmark(x_np, w_np, ctx_dict, label, warmup=3, runs=10):
    with Context(**ctx_dict):
        # Compile
        x = Tensor(x_np, dtype=dtypes.float32).realize()
        w = Tensor(w_np, dtype=dtypes.float32).realize()

        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        compile_time = time.time() - t0

        result = out.numpy()

        # Runtime
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

    print(f"{label:<25} | Compile: {compile_time*1000:>7.1f}ms | Runtime: {avg_runtime:>7.1f}ms")
    return compile_time * 1000, avg_runtime, result

# Create test data
np.random.seed(42)
x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

phase = sys.argv[1] if len(sys.argv) > 1 else "unknown"
print(f"\n{'='*80}")
print(f"PHASE {phase} RUNTIME TEST")
print(f"{'='*80}\n")

# Baseline
c_base, r_base, base_result = proper_benchmark(x_np, w_np, {'WINO': 0, 'WINO_OLD': 0}, "BASELINE (no wino)")

# OLD complete
c_old, r_old, old_result = proper_benchmark(x_np, w_np, {'WINO': 0, 'WINO_OLD': 1}, "OLD (complete)")

# NEW (current phase)
c_new, r_new, new_result = proper_benchmark(x_np, w_np, {'WINO': 1, 'WINO_OLD': 0}, f"NEW (Phase {phase})")

# Analysis
print(f"\n{'='*80}")
print(f"PHASE {phase} ANALYSIS")
print(f"{'='*80}")
print(f"Baseline runtime:  {r_base:.1f}ms")
print(f"OLD runtime:       {r_old:.1f}ms ({r_old/r_base:.1f}× vs baseline)")
print(f"NEW runtime:       {r_new:.1f}ms ({r_new/r_base:.1f}× vs baseline)")
print(f"\nNEW vs OLD: {r_new:.1f}ms vs {r_old:.1f}ms")

diff_ms = r_new - r_old
diff_pct = (r_new/r_old - 1) * 100

if abs(diff_pct) < 2:
    print(f"≈ Same runtime ({diff_pct:+.1f}%)")
elif diff_pct > 0:
    print(f"❌ NEW is {diff_pct:.1f}% SLOWER ({diff_ms:+.1f}ms)")
else:
    print(f"✅ NEW is {-diff_pct:.1f}% FASTER ({diff_ms:+.1f}ms)")

# Accuracy
err = np.abs(new_result - base_result).max()
print(f"\nAccuracy: max error = {err:.2e}")
