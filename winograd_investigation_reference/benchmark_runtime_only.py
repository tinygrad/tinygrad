#!/usr/bin/env python3
"""
Benchmark RUNTIME ONLY (excluding compilation) for HYBRID vs OLD
"""
import time
import os
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import GlobalCounters

def benchmark_runtime_only(name, wino_old, input_sizes, num_warmup=10, num_runs=50):
    """Benchmark winograd runtime excluding compilation time"""
    os.environ['RANGEIFY'] = '1'
    os.environ['WINO_OLD'] = '1' if wino_old else '0'
    os.environ['WINO'] = '0' if wino_old else '1'

    results = {}

    for size in input_sizes:
        print(f"\n{name} - Testing {size}×{size} input...")

        # Create tensors
        x = Tensor.randn(1, 16, size, size, dtype=dtypes.float32).realize()
        w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()

        # EXTENSIVE warmup to ensure all kernels are compiled
        print(f"  Warming up (compiling kernels)...")
        for i in range(num_warmup):
            out = x.conv2d(w, padding=1).realize()
            if i == 0:
                print(f"    First run complete (compilation done)")

        print(f"  Benchmarking runtime (kernels pre-compiled)...")

        # Reset GlobalCounters
        GlobalCounters.reset()

        # Benchmark with pre-compiled kernels
        times = []
        kernel_times = []

        for i in range(num_runs):
            # Clear counters
            GlobalCounters.reset()

            # Synchronize before timing
            Device[Device.DEFAULT].synchronize()

            start = time.perf_counter()
            out = x.conv2d(w, padding=1).realize()
            Device[Device.DEFAULT].synchronize()

            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

            # Get kernel time from GlobalCounters
            if hasattr(GlobalCounters, 'time_sum_s'):
                kernel_time = GlobalCounters.time_sum_s * 1000  # Convert to ms
                kernel_times.append(kernel_time)

            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{num_runs} runs complete")

        # Use kernel times if available, otherwise use wall time
        if kernel_times and len(kernel_times) == num_runs:
            result_times = kernel_times
            time_type = "Kernel time"
        else:
            result_times = times
            time_type = "Wall time"

        results[size] = {
            'mean': np.mean(result_times),
            'std': np.std(result_times),
            'min': np.min(result_times),
            'max': np.max(result_times),
            'median': np.median(result_times),
            'time_type': time_type
        }

        print(f"  {time_type}:")
        print(f"    Min:    {results[size]['min']:.3f}ms")
        print(f"    Mean:   {results[size]['mean']:.3f}ms ± {results[size]['std']:.3f}ms")
        print(f"    Median: {results[size]['median']:.3f}ms")

    return results

def main():
    print("="*70)
    print("RUNTIME-ONLY BENCHMARK: HYBRID vs OLD")
    print("="*70)
    print(f"Device: {Device.DEFAULT}")
    print(f"Warmup runs: 10 (to compile all kernels)")
    print(f"Benchmark runs: 50 (kernels pre-compiled)")
    print(f"Measuring: Kernel execution time only, not compilation")

    # Test multiple input sizes
    input_sizes = [24, 32, 48, 64, 96, 128]

    # Benchmark OLD
    print("\n" + "="*70)
    print("BENCHMARKING OLD (r_6_6_16 grid, coalesced writes)")
    print("="*70)
    old_results = benchmark_runtime_only("OLD", True, input_sizes)

    # Benchmark HYBRID
    print("\n" + "="*70)
    print("BENCHMARKING HYBRID (r_16_6_6 grid, scattered writes)")
    print("="*70)
    hybrid_results = benchmark_runtime_only("HYBRID", False, input_sizes)

    # Compare results
    print("\n" + "="*70)
    print("RUNTIME COMPARISON (Compilation Excluded)")
    print("="*70)
    print(f"\n{'Size':<10} {'OLD (ms)':<15} {'HYBRID (ms)':<15} {'Slowdown':<15} {'Status':<10}")
    print("-"*70)

    for size in input_sizes:
        old_time = old_results[size]['min']
        hybrid_time = hybrid_results[size]['min']
        slowdown = (hybrid_time / old_time - 1) * 100

        if slowdown < 5:
            status = "✅ GREAT"
        elif slowdown < 10:
            status = "✅ GOOD"
        elif slowdown < 20:
            status = "⚠️ OK"
        else:
            status = "❌ BAD"

        print(f"{size}×{size:<5} {old_time:>6.3f}ms       {hybrid_time:>6.3f}ms       {slowdown:>+6.1f}%         {status}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (RUNTIME ONLY)")
    print("="*70)

    avg_slowdown = np.mean([(hybrid_results[s]['min'] / old_results[s]['min'] - 1) * 100
                             for s in input_sizes])

    print(f"\nAverage runtime slowdown: {avg_slowdown:+.1f}%")
    print(f"Time measurement: {old_results[input_sizes[0]]['time_type']}")

    if avg_slowdown < 5:
        print("\n✅ OUTSTANDING! HYBRID runtime is within 5% of OLD.")
        print("   Recommendation: Ship HYBRID immediately.")
    elif avg_slowdown < 10:
        print("\n✅ EXCELLENT! HYBRID runtime is within 10% of OLD.")
        print("   Recommendation: Ship HYBRID as-is.")
    elif avg_slowdown < 20:
        print("\n⚠️ ACCEPTABLE. HYBRID runtime is 10-20% slower than OLD.")
        print("   Recommendation: Ship HYBRID, but note for future optimization.")
    elif avg_slowdown < 50:
        print("\n❌ SIGNIFICANT SLOWDOWN. HYBRID runtime is 20-50% slower than OLD.")
        print("   Recommendation: Investigate optimizations or refactor.")
    else:
        print("\n❌ CRITICAL! HYBRID runtime is >50% slower than OLD.")
        print("   Recommendation: Major refactor needed.")

    print("\n" + "="*70)

    # Comparison vs previous benchmark
    print("\nNOTE: Compare these RUNTIME numbers to previous TOTAL TIME benchmark.")
    print("      If RUNTIME is similar to TOTAL TIME for OLD, compilation overhead is small.")
    print("      If HYBRID RUNTIME is much better than HYBRID TOTAL TIME, NEW has higher")
    print("      compilation overhead but similar execution performance.")

    # Detailed stats
    print("\n" + "="*70)
    print("DETAILED RUNTIME STATISTICS")
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
        diff = hybrid_results[size]['min'] - old_results[size]['min']
        print(f"  Difference: {diff:+.3f}ms ({(diff/old_results[size]['min'])*100:+.1f}%)")

if __name__ == "__main__":
    main()
