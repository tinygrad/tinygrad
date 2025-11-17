#!/usr/bin/env python3
"""
Benchmark HYBRID (r_16_6_6 grid) vs OLD (r_6_6_16 grid) winograd performance
"""
import time
import os
import numpy as np
from tinygrad import Tensor, dtypes, Device

def benchmark_winograd(name, wino_old, input_sizes, num_warmup=5, num_runs=20):
    """Benchmark winograd with given configuration"""
    os.environ['RANGEIFY'] = '1'
    os.environ['WINO_OLD'] = '1' if wino_old else '0'
    os.environ['WINO'] = '0' if wino_old else '1'

    results = {}

    for size in input_sizes:
        print(f"\n{name} - Testing {size}×{size} input...")

        # Create tensors
        x = Tensor.randn(1, 16, size, size, dtype=dtypes.float32).realize()
        w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()

        # Warmup
        for _ in range(num_warmup):
            out = x.conv2d(w, padding=1).realize()

        # Benchmark
        times = []
        for i in range(num_runs):
            # Synchronize before timing
            Device[Device.DEFAULT].synchronize()
            start = time.perf_counter()
            out = x.conv2d(w, padding=1).realize()
            Device[Device.DEFAULT].synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{num_runs} runs complete")

        results[size] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }

        print(f"  Min: {results[size]['min']:.3f}ms")
        print(f"  Mean: {results[size]['mean']:.3f}ms ± {results[size]['std']:.3f}ms")
        print(f"  Median: {results[size]['median']:.3f}ms")

    return results

def main():
    print("="*70)
    print("WINOGRAD PERFORMANCE BENCHMARK: HYBRID vs OLD")
    print("="*70)
    print(f"Device: {Device.DEFAULT}")
    print(f"Warmup runs: 5")
    print(f"Benchmark runs: 20")

    # Test multiple input sizes
    input_sizes = [24, 32, 48, 64, 96]

    # Benchmark OLD
    print("\n" + "="*70)
    print("BENCHMARKING OLD (r_6_6_16 grid, coalesced writes)")
    print("="*70)
    old_results = benchmark_winograd("OLD", True, input_sizes)

    # Benchmark HYBRID
    print("\n" + "="*70)
    print("BENCHMARKING HYBRID (r_16_6_6 grid, scattered writes)")
    print("="*70)
    hybrid_results = benchmark_winograd("HYBRID", False, input_sizes)

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Size':<10} {'OLD (ms)':<15} {'HYBRID (ms)':<15} {'Slowdown':<15} {'Status':<10}")
    print("-"*70)

    for size in input_sizes:
        old_time = old_results[size]['min']
        hybrid_time = hybrid_results[size]['min']
        slowdown = (hybrid_time / old_time - 1) * 100

        if slowdown < 10:
            status = "✅ GOOD"
        elif slowdown < 20:
            status = "⚠️ OK"
        else:
            status = "❌ BAD"

        print(f"{size}×{size:<5} {old_time:>6.3f}ms       {hybrid_time:>6.3f}ms       {slowdown:>+6.1f}%         {status}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    avg_slowdown = np.mean([(hybrid_results[s]['min'] / old_results[s]['min'] - 1) * 100
                             for s in input_sizes])

    print(f"\nAverage slowdown: {avg_slowdown:+.1f}%")

    if avg_slowdown < 10:
        print("\n✅ EXCELLENT! HYBRID performance is within 10% of OLD.")
        print("   Recommendation: Ship HYBRID as-is.")
    elif avg_slowdown < 20:
        print("\n⚠️ ACCEPTABLE. HYBRID is 10-20% slower than OLD.")
        print("   Recommendation: Ship HYBRID, but consider optimization later.")
    elif avg_slowdown < 50:
        print("\n❌ SIGNIFICANT SLOWDOWN. HYBRID is 20-50% slower than OLD.")
        print("   Recommendation: Investigate optimizations or major refactor.")
    else:
        print("\n❌ CRITICAL! HYBRID is >50% slower than OLD.")
        print("   Recommendation: Major refactor needed to match OLD's structure.")

    print("\n" + "="*70)

    # Detailed stats table
    print("\nDETAILED STATISTICS")
    print("="*70)
    for size in input_sizes:
        print(f"\n{size}×{size} Input:")
        print(f"  OLD:")
        print(f"    Min:    {old_results[size]['min']:.3f}ms")
        print(f"    Mean:   {old_results[size]['mean']:.3f}ms ± {old_results[size]['std']:.3f}ms")
        print(f"    Median: {old_results[size]['median']:.3f}ms")
        print(f"  HYBRID:")
        print(f"    Min:    {hybrid_results[size]['min']:.3f}ms")
        print(f"    Mean:   {hybrid_results[size]['mean']:.3f}ms ± {hybrid_results[size]['std']:.3f}ms")
        print(f"    Median: {hybrid_results[size]['median']:.3f}ms")

if __name__ == "__main__":
    main()
