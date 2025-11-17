#!/usr/bin/env python3
"""
Systematic investigation of where NEW winograd loses performance vs OLD.
Each phase adds more complexity to isolate the bottleneck.
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import WINO, WINO_OLD, Context
import numpy as np
import time

# Test shape - medium size where OLD is clearly faster
B, Cin, Cout, H, W = 1, 16, 16, 32, 32

def bench(x, w, pad, ctx_dict, label):
    """Run benchmark and return metrics"""
    with Context(**ctx_dict):
        t0 = time.time()
        out = x.conv2d(w, padding=pad)
        out.realize()
        compile_time = time.time() - t0

        t1 = time.time()
        result = out.numpy()
        run_time = time.time() - t1

    print(f"{label:<30} Compile: {compile_time*1000:>7.1f}ms | Run: {run_time*1000:>7.1f}ms")
    return compile_time, run_time, result

print("="*80)
print(f"Winograd Performance Bisection - Shape: B={B} Cin={Cin} Cout={Cout} H={H}x{W}")
print("="*80)

# Create test data
np.random.seed(42)
x = Tensor.randn(B, Cin, H, W, dtype=dtypes.float32).realize()
w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()

print("\n" + "-"*80)
print("BASELINE: Standard convolution (no winograd)")
print("-"*80)
c_base, r_base, base_result = bench(x, w, 1, {'WINO': 0, 'WINO_OLD': 0}, "BASE (no wino)")

print("\n" + "-"*80)
print("CURRENT IMPLEMENTATIONS")
print("-"*80)
c_old, r_old, old_result = bench(x, w, 1, {'WINO': 0, 'WINO_OLD': 1}, "OLD (tensor.py)")
c_new, r_new, new_result = bench(x, w, 1, {'WINO': 1, 'WINO_OLD': 0}, "NEW (unified kron)")

print("\n" + "-"*80)
print("PERFORMANCE COMPARISON")
print("-"*80)
print(f"{'Metric':<30} {'BASE':<15} {'OLD':<15} {'NEW':<15} {'OLD vs NEW':<20}")
print("-"*80)
print(f"{'Compile Time':<30} {c_base*1000:>7.1f}ms{'':<7} {c_old*1000:>7.1f}ms{'':<7} "
      f"{c_new*1000:>7.1f}ms{'':<7} {('NEW faster' if c_new < c_old else 'OLD faster'):>20}")
print(f"{'Runtime':<30} {r_base*1000:>7.1f}ms{'':<7} {r_old*1000:>7.1f}ms{'':<7} "
      f"{r_new*1000:>7.1f}ms{'':<7} {('NEW faster' if r_new < r_old else 'OLD faster'):>20}")
print(f"{'Compile Overhead':<30} {'-':<15} {c_old/c_base:>7.1f}×{'':<7} "
      f"{c_new/c_base:>7.1f}×{'':<7} {'':<20}")
print(f"{'Runtime Speedup':<30} {'-':<15} {r_base/r_old:>7.2f}×{'':<7} "
      f"{r_base/r_new:>7.2f}×{'':<7} {'':<20}")

# Check correctness
d_old = np.abs(old_result - base_result)
d_new = np.abs(new_result - base_result)
print(f"{'Numerical Error':<30} {'-':<15} {d_old.max():<15.2e} {d_new.max():<15.2e} {'':<20}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if c_new > c_old:
    pct = ((c_new - c_old) / c_old) * 100
    print(f"❌ NEW compile time is {pct:.1f}% SLOWER than OLD ({c_new*1000:.1f}ms vs {c_old*1000:.1f}ms)")
else:
    pct = ((c_old - c_new) / c_new) * 100
    print(f"✓ NEW compile time is {pct:.1f}% FASTER than OLD ({c_new*1000:.1f}ms vs {c_old*1000:.1f}ms)")

if r_new > r_old:
    pct = ((r_new - r_old) / r_old) * 100
    print(f"❌ NEW runtime is {pct:.1f}% SLOWER than OLD ({r_new*1000:.1f}ms vs {r_old*1000:.1f}ms)")
else:
    pct = ((r_old - r_new) / r_new) * 100
    print(f"✓ NEW runtime is {pct:.1f}% FASTER than OLD ({r_new*1000:.1f}ms vs {r_old*1000:.1f}ms)")

print("\nNext steps:")
print("1. If NEW is slower: Modify schedule/rangeify.py to simplify winowrite incrementally")
print("2. Start by returning XHAT early (skip GHAT, MHAT, output)")
print("3. Progressively add back components to find where performance degrades")
print("4. Check DEBUG=2 output to see kernel fusion differences")
print("="*80)
