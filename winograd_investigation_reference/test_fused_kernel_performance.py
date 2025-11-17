#!/usr/bin/env python3
"""
Compare performance: 4-buffer winograd vs 3-buffer winograd (no MHAT bufferize)
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

def benchmark(B, Cin, Cout, H, W, warmup=2, runs=8):
    """Benchmark a single shape"""
    np.random.seed(42)
    x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
    w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

    results = {}

    # Baseline
    with Context(WINO=0, WINO_OLD=0):
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
        results['baseline'] = {
            'compile': compile_time * 1000,
            'runtime': np.mean(runtime_samples) * 1000,
            'result': result
        }

    # NEW winograd (3 buffers - no MHAT bufferize)
    with Context(WINO=1, WINO_OLD=0):
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
        results['new_fused'] = {
            'compile': compile_time * 1000,
            'runtime': np.mean(runtime_samples) * 1000,
            'result': result
        }

    return results

# Test shapes
test_shapes = [
    (1, 8, 8, 32, 32),      # Small
    (1, 16, 16, 32, 32),    # Medium
    (1, 32, 32, 64, 64),    # Large
]

print("="*100)
print("PERFORMANCE COMPARISON: FUSED WINOGRAD (3 buffers) vs BASELINE")
print("="*100)

all_results = []

for i, (B, Cin, Cout, H, W) in enumerate(test_shapes, 1):
    print(f"\n[{i}/{len(test_shapes)}] Shape: B={B}, Cin={Cin}, Cout={Cout}, H={H}×{W}")
    print("-"*100)

    results = benchmark(B, Cin, Cout, H, W)
    all_results.append((B, Cin, Cout, H, W, results))

    base_rt = results['baseline']['runtime']
    new_rt = results['new_fused']['runtime']

    base_compile = results['baseline']['compile']
    new_compile = results['new_fused']['compile']

    print(f"  Compile time:")
    print(f"    Baseline:    {base_compile:>7.1f}ms")
    print(f"    NEW (fused): {new_compile:>7.1f}ms ({new_compile/base_compile:>5.2f}×)")

    print(f"  Runtime:")
    print(f"    Baseline:    {base_rt:>7.3f}ms")
    print(f"    NEW (fused): {new_rt:>7.3f}ms ({new_rt/base_rt:>5.2f}×)")

    diff_pct = (new_rt/base_rt - 1) * 100
    if abs(diff_pct) < 5:
        print(f"  Result: ≈ SAME runtime ({diff_pct:+.1f}%)")
    elif diff_pct > 0:
        print(f"  Result: ❌ NEW {diff_pct:.1f}% SLOWER")
    else:
        print(f"  Result: ✅ NEW {-diff_pct:.1f}% FASTER")

    # Accuracy
    err = np.abs(results['new_fused']['result'] - results['baseline']['result']).max()
    print(f"  Accuracy: max error = {err:.2e}")

# Summary
print("\n" + "="*100)
print("SUMMARY")
print("="*100)

print(f"\n{'Shape':<25} {'Baseline':<15} {'NEW Fused':<15} {'Speedup':<15}")
print("-"*100)

for B, Cin, Cout, H, W, results in all_results:
    shape_str = f"B={B} C={Cin}→{Cout} {H}×{W}"
    base_rt = results['baseline']['runtime']
    new_rt = results['new_fused']['runtime']
    speedup = base_rt / new_rt

    print(f"{shape_str:<25} {base_rt:>7.3f}ms        {new_rt:>7.3f}ms        {speedup:>6.2f}×")

# Overall stats
speedups = [results['baseline']['runtime'] / results['new_fused']['runtime']
            for _, _, _, _, _, results in all_results]
avg_speedup = np.mean(speedups)

print("\n" + "="*100)
print(f"Average speedup vs baseline: {avg_speedup:.2f}×")

if avg_speedup > 1.05:
    print(f"✅ NEW fused winograd is {(avg_speedup-1)*100:.1f}% FASTER on average!")
elif avg_speedup < 0.95:
    print(f"❌ NEW fused winograd is {(1-avg_speedup)*100:.1f}% SLOWER on average")
else:
    print(f"≈ NEW fused winograd has SIMILAR performance to baseline")

print("\nKey achievement: Reduced from 4 bufferizes to 3 bufferizes (2 kernels instead of 4)")
print("="*100)
