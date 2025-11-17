#!/usr/bin/env python3
"""
Compare NEW winograd (2 buffers: MHAT+output) vs OLD winograd (4 buffers)
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

def benchmark_shape(B, Cin, Cout, H, W, warmup=3, runs=10):
    """Benchmark a single shape configuration"""
    np.random.seed(42)
    x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
    w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

    results = {}

    # BASELINE (no winograd)
    print(f"  Running BASELINE (no winograd)...", end='', flush=True)
    with Context(WINO=0, WINO_OLD=0):
        x = Tensor(x_np, dtype=dtypes.float32).realize()
        w = Tensor(w_np, dtype=dtypes.float32).realize()

        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        compile_time = time.time() - t0
        result_baseline = out.numpy()

        runtime_samples = []
        for _ in range(warmup + runs):
            x = Tensor(x_np, dtype=dtypes.float32).realize()
            w = Tensor(w_np, dtype=dtypes.float32).realize()
            t1 = time.time()
            out = x.conv2d(w, padding=1)
            out.realize()
            runtime_samples.append(time.time() - t1)

        runtime_samples = runtime_samples[warmup:]
        results['BASELINE'] = {
            'compile': compile_time * 1000,
            'runtime': np.mean(runtime_samples) * 1000,
            'result': result_baseline
        }
    print(f" {results['BASELINE']['runtime']:.1f}ms")

    # OLD winograd (4 buffers)
    print(f"  Running OLD winograd (4 buffers)...", end='', flush=True)
    with Context(WINO=0, WINO_OLD=1):
        x = Tensor(x_np, dtype=dtypes.float32).realize()
        w = Tensor(w_np, dtype=dtypes.float32).realize()

        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        compile_time = time.time() - t0
        result_old = out.numpy()

        runtime_samples = []
        for _ in range(warmup + runs):
            x = Tensor(x_np, dtype=dtypes.float32).realize()
            w = Tensor(w_np, dtype=dtypes.float32).realize()
            t1 = time.time()
            out = x.conv2d(w, padding=1)
            out.realize()
            runtime_samples.append(time.time() - t1)

        runtime_samples = runtime_samples[warmup:]
        results['OLD'] = {
            'compile': compile_time * 1000,
            'runtime': np.mean(runtime_samples) * 1000,
            'result': result_old
        }
    print(f" {results['OLD']['runtime']:.1f}ms")

    # NEW winograd (2 buffers: MHAT+output)
    print(f"  Running NEW winograd (2 buffers)...", end='', flush=True)
    with Context(WINO=1, WINO_OLD=0):
        x = Tensor(x_np, dtype=dtypes.float32).realize()
        w = Tensor(w_np, dtype=dtypes.float32).realize()

        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        compile_time = time.time() - t0
        result_new = out.numpy()

        runtime_samples = []
        for _ in range(warmup + runs):
            x = Tensor(x_np, dtype=dtypes.float32).realize()
            w = Tensor(w_np, dtype=dtypes.float32).realize()
            t1 = time.time()
            out = x.conv2d(w, padding=1)
            out.realize()
            runtime_samples.append(time.time() - t1)

        runtime_samples = runtime_samples[warmup:]
        results['NEW'] = {
            'compile': compile_time * 1000,
            'runtime': np.mean(runtime_samples) * 1000,
            'result': result_new
        }
    print(f" {results['NEW']['runtime']:.1f}ms")

    return results

# Test shapes
test_shapes = [
    (1, 8, 8, 32, 32),      # Small
    (1, 16, 16, 32, 32),    # Medium channels
    (1, 32, 32, 64, 64),    # Large
    (1, 64, 64, 64, 64),    # Very large
]

print("="*100)
print("PERFORMANCE COMPARISON: NEW (2 buffers) vs OLD (4 buffers) WINOGRAD")
print("="*100)

all_results = []

for i, (B, Cin, Cout, H, W) in enumerate(test_shapes, 1):
    print(f"\n[{i}/{len(test_shapes)}] Shape: B={B}, Cin={Cin}, Cout={Cout}, H={H}×{W}")
    print("-"*100)

    results = benchmark_shape(B, Cin, Cout, H, W)
    all_results.append((B, Cin, Cout, H, W, results))

    baseline_rt = results['BASELINE']['runtime']
    old_rt = results['OLD']['runtime']
    new_rt = results['NEW']['runtime']

    baseline_compile = results['BASELINE']['compile']
    old_compile = results['OLD']['compile']
    new_compile = results['NEW']['compile']

    print(f"\n  Compile time:")
    print(f"    BASELINE: {baseline_compile:>7.1f}ms")
    print(f"    OLD:      {old_compile:>7.1f}ms ({old_compile/baseline_compile:>5.2f}×)")
    print(f"    NEW:      {new_compile:>7.1f}ms ({new_compile/baseline_compile:>5.2f}×)")

    print(f"  Runtime:")
    print(f"    BASELINE: {baseline_rt:>7.1f}ms")
    print(f"    OLD:      {old_rt:>7.1f}ms ({old_rt/baseline_rt:>5.2f}×)")
    print(f"    NEW:      {new_rt:>7.1f}ms ({new_rt/baseline_rt:>5.2f}×)")

    diff_pct = (new_rt/old_rt - 1) * 100
    if abs(diff_pct) < 5:
        print(f"  NEW vs OLD: ≈ SAME ({diff_pct:+.1f}%)")
    elif diff_pct > 0:
        print(f"  NEW vs OLD: ❌ NEW {diff_pct:.1f}% SLOWER")
    else:
        print(f"  NEW vs OLD: ✅ NEW {-diff_pct:.1f}% FASTER")

    # Accuracy checks
    old_err = np.abs(results['OLD']['result'] - results['BASELINE']['result']).max()
    new_err = np.abs(results['NEW']['result'] - results['BASELINE']['result']).max()
    print(f"  Accuracy: OLD err={old_err:.2e}, NEW err={new_err:.2e}")

# Summary table
print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
print(f"{'Shape':<25} {'Baseline':<12} {'OLD':<12} {'NEW':<12} {'NEW vs OLD':<15}")
print("-"*100)

for B, Cin, Cout, H, W, results in all_results:
    shape_str = f"B={B} C={Cin}→{Cout} {H}×{W}"
    baseline_rt = results['BASELINE']['runtime']
    old_rt = results['OLD']['runtime']
    new_rt = results['NEW']['runtime']
    diff_pct = (new_rt/old_rt - 1) * 100

    print(f"{shape_str:<25} {baseline_rt:>7.1f}ms    {old_rt:>7.1f}ms    {new_rt:>7.1f}ms    {diff_pct:>+6.1f}%")

# Statistics
print("\n" + "="*100)
print("OVERALL STATISTICS")
print("="*100)

new_wins = 0
old_wins = 0
ties = 0

for B, Cin, Cout, H, W, results in all_results:
    old_rt = results['OLD']['runtime']
    new_rt = results['NEW']['runtime']
    diff_pct = (new_rt/old_rt - 1) * 100

    if abs(diff_pct) < 5:
        ties += 1
    elif diff_pct > 0:
        old_wins += 1
    else:
        new_wins += 1

print(f"NEW wins (>5% faster): {new_wins}/{len(test_shapes)}")
print(f"OLD wins (>5% faster): {old_wins}/{len(test_shapes)}")
print(f"Ties (within 5%): {ties}/{len(test_shapes)}")

# Overall average
all_diffs = []
for B, Cin, Cout, H, W, results in all_results:
    old_rt = results['OLD']['runtime']
    new_rt = results['NEW']['runtime']
    all_diffs.append((new_rt/old_rt - 1) * 100)

avg_diff = np.mean(all_diffs)
print(f"\nAverage difference (NEW vs OLD): {avg_diff:+.1f}%")

if avg_diff > 5:
    print(f"❌ On average, NEW is {avg_diff:.1f}% SLOWER than OLD")
elif avg_diff < -5:
    print(f"✅ On average, NEW is {-avg_diff:.1f}% FASTER than OLD")
else:
    print(f"≈ On average, NEW and OLD have SIMILAR performance ({avg_diff:+.1f}%)")

print("\n" + "="*100)
print("BUFFER COUNT COMPARISON")
print("="*100)
print("OLD approach: 4 buffers (XHAT, GHAT, MHAT, output)")
print("NEW approach: 2 buffers (MHAT, output)")
print("Reduction: 50% fewer buffers and kernel launches!")
print("="*100)
