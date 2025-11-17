#!/usr/bin/env python3
"""
Proper runtime benchmark with accuracy verification.
Measures actual kernel execution time, not just data transfer.
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

def benchmark(B, Cin, Cout, H, W, warmup=2, runs=10):
    print(f"\n{'='*80}")
    print(f"Shape: B={B}, Cin={Cin}, Cout={Cout}, H={H}×{W}")
    print(f"{'='*80}")

    # Create test data once
    np.random.seed(42)
    x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
    w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

    results = {}

    for name, ctx_dict in [
        ("BASELINE", {'WINO': 0, 'WINO_OLD': 0}),
        ("OLD", {'WINO': 0, 'WINO_OLD': 1}),
        ("NEW", {'WINO': 1, 'WINO_OLD': 0}),
    ]:
        with Context(**ctx_dict):
            # COMPILE PHASE: First run to compile kernels
            x = Tensor(x_np, dtype=dtypes.float32).realize()
            w = Tensor(w_np, dtype=dtypes.float32).realize()

            t0 = time.time()
            out = x.conv2d(w, padding=1)
            out.realize()  # This compiles and runs first time
            compile_time = time.time() - t0

            # Get result for accuracy check
            result = out.numpy()

            # RUNTIME PHASE: Measure pure execution time
            # Create fresh tensors to avoid any caching
            runtime_samples = []
            for _ in range(warmup + runs):
                x = Tensor(x_np, dtype=dtypes.float32).realize()
                w = Tensor(w_np, dtype=dtypes.float32).realize()

                t1 = time.time()
                out = x.conv2d(w, padding=1)
                out.realize()  # This executes the already-compiled kernel
                elapsed = time.time() - t1

                runtime_samples.append(elapsed)

            # Drop warmup samples
            runtime_samples = runtime_samples[warmup:]
            avg_runtime = np.mean(runtime_samples) * 1000  # ms
            std_runtime = np.std(runtime_samples) * 1000   # ms
            min_runtime = np.min(runtime_samples) * 1000   # ms

            results[name] = {
                'compile': compile_time * 1000,  # ms
                'runtime_avg': avg_runtime,
                'runtime_std': std_runtime,
                'runtime_min': min_runtime,
                'result': result
            }

            print(f"{name:<12} | Compile: {compile_time*1000:>7.1f}ms | "
                  f"Runtime: {avg_runtime:>7.3f}ms ± {std_runtime:>5.3f}ms (min: {min_runtime:>6.3f}ms)")

    # ACCURACY VERIFICATION
    print(f"\n{'-'*80}")
    print("ACCURACY VERIFICATION")
    print(f"{'-'*80}")

    baseline_result = results['BASELINE']['result']
    old_result = results['OLD']['result']
    new_result = results['NEW']['result']

    old_err = np.abs(old_result - baseline_result).max()
    new_err = np.abs(new_result - baseline_result).max()
    old_vs_new_err = np.abs(old_result - new_result).max()

    print(f"OLD vs BASELINE max error: {old_err:.2e}")
    print(f"NEW vs BASELINE max error: {new_err:.2e}")
    print(f"OLD vs NEW max error:      {old_vs_new_err:.2e}")

    if old_err < 1e-2 and new_err < 1e-2:
        print("✅ Both implementations are numerically accurate!")
    else:
        print("⚠️  Accuracy issues detected!")

    # PERFORMANCE COMPARISON
    print(f"\n{'-'*80}")
    print("RUNTIME PERFORMANCE COMPARISON")
    print(f"{'-'*80}")

    baseline_rt = results['BASELINE']['runtime_avg']
    old_rt = results['OLD']['runtime_avg']
    new_rt = results['NEW']['runtime_avg']

    print(f"BASELINE: {baseline_rt:.3f}ms (reference)")
    print(f"OLD:      {old_rt:.3f}ms ({old_rt/baseline_rt:>5.2f}× vs baseline)")
    print(f"NEW:      {new_rt:.3f}ms ({new_rt/baseline_rt:>5.2f}× vs baseline)")
    print()
    print(f"🎯 NEW vs OLD runtime: {new_rt:.3f}ms vs {old_rt:.3f}ms")

    if new_rt < old_rt:
        speedup = old_rt / new_rt
        print(f"✅ NEW is {(old_rt-new_rt)/old_rt*100:.1f}% FASTER ({speedup:.2f}× speedup)")
    else:
        slowdown = new_rt / old_rt
        print(f"❌ NEW is {(new_rt-old_rt)/old_rt*100:.1f}% SLOWER ({slowdown:.2f}× slower)")

    # COMPILE TIME COMPARISON
    print(f"\n{'-'*80}")
    print("COMPILE TIME COMPARISON")
    print(f"{'-'*80}")

    old_ct = results['OLD']['compile']
    new_ct = results['NEW']['compile']

    print(f"OLD compile: {old_ct:.1f}ms")
    print(f"NEW compile: {new_ct:.1f}ms")

    if new_ct < old_ct:
        print(f"NEW compiles {(old_ct-new_ct)/old_ct*100:.1f}% faster")
    else:
        print(f"NEW compiles {(new_ct-old_ct)/old_ct*100:.1f}% slower")

    return results

# Test both small and large shapes
print("="*80)
print("PROPER RUNTIME BENCHMARK WITH ACCURACY VERIFICATION")
print("="*80)

# Small shape
small_results = benchmark(B=1, Cin=16, Cout=16, H=32, W=32, warmup=3, runs=15)

# Large shape
large_results = benchmark(B=1, Cin=64, Cout=64, H=64, W=64, warmup=2, runs=10)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nSmall shape (32×32, Cin=16):")
new_rt = small_results['NEW']['runtime_avg']
old_rt = small_results['OLD']['runtime_avg']
if new_rt < old_rt:
    print(f"  ✅ NEW is {old_rt/new_rt:.2f}× faster runtime")
else:
    print(f"  ❌ NEW is {new_rt/old_rt:.2f}× slower runtime")

print("\nLarge shape (64×64, Cin=64):")
new_rt = large_results['NEW']['runtime_avg']
old_rt = large_results['OLD']['runtime_avg']
if new_rt < old_rt:
    print(f"  ✅ NEW is {old_rt/new_rt:.2f}× faster runtime")
else:
    print(f"  ❌ NEW is {new_rt/old_rt:.2f}× slower runtime")

print("\n" + "="*80)
