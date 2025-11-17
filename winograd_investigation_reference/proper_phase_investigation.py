#!/usr/bin/env python3
"""
Proper phase-by-phase runtime investigation.
Measures actual kernel execution time (realize), not data transfer.

Instructions:
1. Modify rangeify.py::winowrite between each phase as indicated
2. Run this script for each phase
3. Observe WHERE runtime degrades compared to OLD
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

B, Cin, Cout, H, W = 1, 16, 16, 32, 32

def proper_benchmark(x_np, w_np, ctx_dict, label, warmup=3, runs=10):
    """Measure actual kernel execution time, not data transfer"""
    with Context(**ctx_dict):
        # COMPILE PHASE
        x = Tensor(x_np, dtype=dtypes.float32).realize()
        w = Tensor(w_np, dtype=dtypes.float32).realize()

        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()  # Compile + first run
        compile_time = time.time() - t0

        # Get result for accuracy
        result = out.numpy()

        # RUNTIME PHASE - measure realize() time
        runtime_samples = []
        for _ in range(warmup + runs):
            x = Tensor(x_np, dtype=dtypes.float32).realize()
            w = Tensor(w_np, dtype=dtypes.float32).realize()

            t1 = time.time()
            out = x.conv2d(w, padding=1)
            out.realize()  # This is the actual kernel execution
            runtime_samples.append(time.time() - t1)

        runtime_samples = runtime_samples[warmup:]
        avg_runtime = np.mean(runtime_samples) * 1000
        std_runtime = np.std(runtime_samples) * 1000
        min_runtime = np.min(runtime_samples) * 1000

    print(f"{label:<25} | Compile: {compile_time*1000:>7.1f}ms | "
          f"Runtime: {avg_runtime:>7.3f}ms ± {std_runtime:>5.3f}ms")

    return compile_time * 1000, avg_runtime, result

print("="*80)
print("PROPER PHASE-BY-PHASE RUNTIME INVESTIGATION")
print(f"Shape: B={B}, Cin={Cin}, Cout={Cout}, H={H}×{W}")
print("="*80)

# Create test data once
np.random.seed(42)
x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

# Get baseline
print("\nBASELINE:")
c_base, r_base, base_result = proper_benchmark(x_np, w_np, {'WINO': 0, 'WINO_OLD': 0}, "BASELINE (no wino)")

# Get OLD complete (for reference)
print("\nOLD COMPLETE (reference):")
c_old, r_old, old_result = proper_benchmark(x_np, w_np, {'WINO': 0, 'WINO_OLD': 1}, "OLD (complete)")

print("\n" + "="*80)
print("PHASE 1: XHAT ONLY")
print("="*80)
print("\nModify winowrite to return after creating XHAT:")
print("  XHAT = kron(X_tiled, winograd_Bt, ...)")
print("  return XHAT.index(*other_loops_x, *other_loops_w, *[ox//4 for ox in o_axes], *[ox%4 for ox in o_axes])")
input("\nPress Enter when ready to test Phase 1...")

c_new, r_new, new_result = proper_benchmark(x_np, w_np, {'WINO': 1, 'WINO_OLD': 0}, "NEW Phase 1 (XHAT)")
print(f"\n📊 Phase 1 Analysis:")
print(f"   Runtime: {r_new:.1f}ms (NEW) vs {r_old:.1f}ms (OLD complete)")
print(f"   Difference: {r_new - r_old:+.1f}ms ({(r_new/r_old - 1)*100:+.1f}%)")

print("\n" + "="*80)
print("PHASE 2: XHAT + GHAT")
print("="*80)
print("\nModify winowrite to create both transforms:")
print("  XHAT = kron(X_tiled, winograd_Bt, ...)")
print("  w_sub = w_like.substitute(...)")
print("  GHAT = kron(w_sub, winograd_G, ...)")
print("  return XHAT.index(...)")
input("\nPress Enter when ready to test Phase 2...")

c_new, r_new, new_result = proper_benchmark(x_np, w_np, {'WINO': 1, 'WINO_OLD': 0}, "NEW Phase 2 (XHAT+GHAT)")
print(f"\n📊 Phase 2 Analysis:")
print(f"   Runtime: {r_new:.1f}ms (NEW) vs {r_old:.1f}ms (OLD complete)")
print(f"   Difference: {r_new - r_old:+.1f}ms ({(r_new/r_old - 1)*100:+.1f}%)")

print("\n" + "="*80)
print("PHASE 3: XHAT * GHAT + REDUCE (MHAT) - CRITICAL PHASE")
print("="*80)
print("\nModify winowrite to add multiply/reduce:")
print("  mhat_redu = (XHAT.index(...) * GHAT.index(...)).reduce(*other_reduces, arg=Ops.ADD)")
print("  MHAT = mhat_redu.bufferize(...)")
print("  return MHAT.index(*other_loops_x, *other_loops_w, *[ox//6 for ox in o_axes], *[ox%6 for ox in o_axes])")
input("\nPress Enter when ready to test Phase 3...")

c_new, r_new, new_result = proper_benchmark(x_np, w_np, {'WINO': 1, 'WINO_OLD': 0}, "NEW Phase 3 (+MHAT)")
print(f"\n📊 Phase 3 Analysis:")
print(f"   Runtime: {r_new:.1f}ms (NEW) vs {r_old:.1f}ms (OLD complete)")
print(f"   Difference: {r_new - r_old:+.1f}ms ({(r_new/r_old - 1)*100:+.1f}%)")
if r_new > r_old * 1.1:
    print(f"   ⚠️  BOTTLENECK DETECTED! Phase 3 is significantly slower!")

print("\n" + "="*80)
print("PHASE 4: FULL ALGORITHM (+ output transform)")
print("="*80)
print("\nRestore complete implementation:")
print("  cp tinygrad/schedule/rangeify.py.backup tinygrad/schedule/rangeify.py")
input("\nPress Enter when ready to test Phase 4...")

c_new, r_new, new_result = proper_benchmark(x_np, w_np, {'WINO': 1, 'WINO_OLD': 0}, "NEW Phase 4 (complete)")

# Accuracy check
new_err = np.abs(new_result - base_result).max()
old_err = np.abs(old_result - base_result).max()
print(f"\n📊 Phase 4 Analysis:")
print(f"   Runtime: {r_new:.1f}ms (NEW) vs {r_old:.1f}ms (OLD)")
print(f"   Difference: {r_new - r_old:+.1f}ms ({(r_new/r_old - 1)*100:+.1f}%)")
print(f"   Accuracy: NEW error={new_err:.2e}, OLD error={old_err:.2e}")

if abs(r_new - r_old) < r_old * 0.05:
    print(f"\n✅ NEW and OLD have similar runtime (within 5%)")
else:
    if r_new > r_old:
        print(f"\n❌ NEW is {(r_new/r_old - 1)*100:.1f}% SLOWER than OLD")
    else:
        print(f"\n✅ NEW is {(r_old/r_new - 1)*100:.1f}% FASTER than OLD")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print("\nKey question: Which phase showed the biggest runtime jump?")
print("That's where the bottleneck is.")
