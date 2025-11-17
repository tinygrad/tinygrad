#!/usr/bin/env python3
"""
Simple runtime test with proper measurement
"""

from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import Context
import numpy as np
import time

B, Cin, Cout, H, W = 1, 16, 16, 32, 32
np.random.seed(42)
x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

print("="*80)
print("SIMPLE RUNTIME TEST")
print(f"Shape: B={B}, Cin={Cin}, Cout={Cout}, H={H}×{W}")
print("="*80)

# OLD winograd
print("\nOLD winograd (WINO_OLD=1):")
with Context(WINO=0, WINO_OLD=1):
    x = Tensor(x_np, dtype=dtypes.float32).realize()
    w = Tensor(w_np, dtype=dtypes.float32).realize()

    # Compile
    print("  Compiling...", end='', flush=True)
    t0 = time.time()
    out = x.conv2d(w, padding=1)
    out.realize()
    Device[out.device].synchronize()
    compile_time = time.time() - t0
    print(f" {compile_time*1000:.1f}ms")

    result_old = out.numpy()

    # Runtime (single run after compilation)
    x = Tensor(x_np, dtype=dtypes.float32).realize()
    w = Tensor(w_np, dtype=dtypes.float32).realize()
    Device[x.device].synchronize()

    t1 = time.time()
    out = x.conv2d(w, padding=1)
    out.realize()
    Device[out.device].synchronize()
    runtime_old = time.time() - t1
    print(f"  Runtime: {runtime_old*1000:.3f}ms")

# NEW winograd
print("\nNEW winograd (WINO=1):")
with Context(WINO=1, WINO_OLD=0):
    x = Tensor(x_np, dtype=dtypes.float32).realize()
    w = Tensor(w_np, dtype=dtypes.float32).realize()

    # Compile
    print("  Compiling...", end='', flush=True)
    t0 = time.time()
    out = x.conv2d(w, padding=1)
    out.realize()
    Device[out.device].synchronize()
    compile_time = time.time() - t0
    print(f" {compile_time*1000:.1f}ms")

    result_new = out.numpy()

    # Runtime (single run after compilation)
    x = Tensor(x_np, dtype=dtypes.float32).realize()
    w = Tensor(w_np, dtype=dtypes.float32).realize()
    Device[x.device].synchronize()

    t1 = time.time()
    out = x.conv2d(w, padding=1)
    out.realize()
    Device[out.device].synchronize()
    runtime_new = time.time() - t1
    print(f"  Runtime: {runtime_new*1000:.3f}ms")

# BASELINE
print("\nBASELINE (no winograd):")
with Context(WINO=0, WINO_OLD=0):
    x = Tensor(x_np, dtype=dtypes.float32).realize()
    w = Tensor(w_np, dtype=dtypes.float32).realize()

    # Compile
    print("  Compiling...", end='', flush=True)
    t0 = time.time()
    out = x.conv2d(w, padding=1)
    out.realize()
    Device[out.device].synchronize()
    compile_time = time.time() - t0
    print(f" {compile_time*1000:.1f}ms")

    result_baseline = out.numpy()

    # Runtime (single run after compilation)
    x = Tensor(x_np, dtype=dtypes.float32).realize()
    w = Tensor(w_np, dtype=dtypes.float32).realize()
    Device[x.device].synchronize()

    t1 = time.time()
    out = x.conv2d(w, padding=1)
    out.realize()
    Device[out.device].synchronize()
    runtime_baseline = time.time() - t1
    print(f"  Runtime: {runtime_baseline*1000:.3f}ms")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Baseline:  {runtime_baseline*1000:.3f}ms")
print(f"OLD wino:  {runtime_old*1000:.3f}ms ({runtime_old/runtime_baseline:.2f}× vs baseline)")
print(f"NEW wino:  {runtime_new*1000:.3f}ms ({runtime_new/runtime_baseline:.2f}× vs baseline)")
print(f"\nNEW vs OLD: {(runtime_new/runtime_old-1)*100:+.1f}%")

if abs(runtime_new/runtime_old - 1) < 0.05:
    print("Result: ≈ SAME performance")
elif runtime_new < runtime_old:
    print(f"Result: ✅ NEW is {(1-runtime_new/runtime_old)*100:.1f}% FASTER")
else:
    print(f"Result: ❌ NEW is {(runtime_new/runtime_old-1)*100:.1f}% SLOWER")

# Accuracy
old_err = np.abs(result_old - result_baseline).max()
new_err = np.abs(result_new - result_baseline).max()
print(f"\nAccuracy: OLD err={old_err:.2e}, NEW err={new_err:.2e}")
print("="*80)
