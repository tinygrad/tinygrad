#!/usr/bin/env python3
"""
Comprehensive runtime comparison of OLD vs NEW winograd across different shapes.
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

    for name, ctx_dict in [
        ("BASELINE", {'WINO': 0, 'WINO_OLD': 0}),
        ("OLD", {'WINO': 0, 'WINO_OLD': 1}),
        ("NEW", {'WINO': 1, 'WINO_OLD': 0}),
    ]:
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
            std_runtime = np.std(runtime_samples) * 1000
            min_runtime = np.min(runtime_samples) * 1000

            results[name] = {
                'compile': compile_time * 1000,
                'runtime': avg_runtime,
                'runtime_std': std_runtime,
                'runtime_min': min_runtime,
                'result': result
            }

    return results

# Test shapes
test_shapes = [
    # Small shapes
    (1, 16, 16, 32, 32),   # Original test shape
    (1, 8, 8, 32, 32),     # Fewer channels
    (1, 32, 32, 32, 32),   # More channels

    # Medium shapes
    (1, 16, 16, 64, 64),   # Larger spatial
    (1, 32, 32, 64, 64),
    (1, 64, 64, 64, 64),   # Large channels + spatial

    # Asymmetric shapes
    (1, 16, 32, 32, 32),   # Different cin/cout
    (1, 64, 16, 32, 32),

    # Batch size variations
    (2, 16, 16, 32, 32),   # Batch=2
    (4, 16, 16, 32, 32),   # Batch=4
]

print("="*100)
print("COMPREHENSIVE RUNTIME COMPARISON: OLD vs NEW WINOGRAD")
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

    # Print compact summary
    print(f"  BASELINE: {baseline_rt:>7.1f}ms")
    print(f"  OLD:      {old_rt:>7.1f}ms ({old_rt/baseline_rt:>5.1f}× vs baseline)")
    print(f"  NEW:      {new_rt:>7.1f}ms ({new_rt/baseline_rt:>5.1f}× vs baseline)")

    diff_ms = new_rt - old_rt
    diff_pct = (new_rt/old_rt - 1) * 100

    if abs(diff_pct) < 2:
        print(f"  Result:   ≈ SAME ({diff_pct:+.1f}%)")
    elif diff_pct > 0:
        print(f"  Result:   ❌ NEW {diff_pct:.1f}% SLOWER ({diff_ms:+.1f}ms)")
    else:
        print(f"  Result:   ✅ NEW {-diff_pct:.1f}% FASTER ({diff_ms:.1f}ms)")

    # Accuracy check
    old_err = np.abs(results['OLD']['result'] - results['BASELINE']['result']).max()
    new_err = np.abs(results['NEW']['result'] - results['BASELINE']['result']).max()
    print(f"  Accuracy: OLD err={old_err:.2e}, NEW err={new_err:.2e}")

# Summary table
print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
print(f"{'Shape':<25} {'Baseline':<12} {'OLD':<12} {'NEW':<12} {'Diff':<15} {'Winner':<10}")
print("-"*100)

for B, Cin, Cout, H, W, results in all_results:
    shape_str = f"B={B} C={Cin}→{Cout} {H}×{W}"
    baseline_rt = results['BASELINE']['runtime']
    old_rt = results['OLD']['runtime']
    new_rt = results['NEW']['runtime']
    diff_pct = (new_rt/old_rt - 1) * 100

    if abs(diff_pct) < 2:
        winner = "≈ SAME"
    elif diff_pct > 0:
        winner = f"OLD ({-diff_pct:.1f}%)"
    else:
        winner = f"NEW ({-diff_pct:.1f}%)"

    print(f"{shape_str:<25} {baseline_rt:>7.1f}ms    {old_rt:>7.1f}ms    {new_rt:>7.1f}ms    "
          f"{diff_pct:>+6.1f}%        {winner:<10}")

# Statistics
print("\n" + "="*100)
print("STATISTICS")
print("="*100)

new_wins = 0
old_wins = 0
ties = 0
total_new_slower = 0
total_old_slower = 0

for B, Cin, Cout, H, W, results in all_results:
    old_rt = results['OLD']['runtime']
    new_rt = results['NEW']['runtime']
    diff_pct = (new_rt/old_rt - 1) * 100

    if abs(diff_pct) < 2:
        ties += 1
    elif diff_pct > 0:
        old_wins += 1
        total_new_slower += diff_pct
    else:
        new_wins += 1
        total_old_slower += -diff_pct

print(f"NEW wins: {new_wins}/{len(test_shapes)}")
print(f"OLD wins: {old_wins}/{len(test_shapes)}")
print(f"Ties (within 2%): {ties}/{len(test_shapes)}")

if old_wins > 0:
    print(f"\nAverage slowdown when NEW loses: {total_new_slower/old_wins:.1f}%")
if new_wins > 0:
    print(f"Average speedup when NEW wins: {total_old_slower/new_wins:.1f}%")

# Overall average
all_diffs = []
for B, Cin, Cout, H, W, results in all_results:
    old_rt = results['OLD']['runtime']
    new_rt = results['NEW']['runtime']
    all_diffs.append((new_rt/old_rt - 1) * 100)

avg_diff = np.mean(all_diffs)
print(f"\nOverall average difference: {avg_diff:+.1f}%")

if avg_diff > 2:
    print(f"❌ On average, NEW is {avg_diff:.1f}% SLOWER than OLD")
elif avg_diff < -2:
    print(f"✅ On average, NEW is {-avg_diff:.1f}% FASTER than OLD")
else:
    print(f"≈ On average, NEW and OLD have SIMILAR performance ({avg_diff:+.1f}%)")

print("\n" + "="*100)
