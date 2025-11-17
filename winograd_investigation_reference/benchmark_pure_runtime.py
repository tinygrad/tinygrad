#!/usr/bin/env python3
"""
Pure runtime benchmark - extensive warmup to eliminate compilation overhead
"""
import time
import os
import numpy as np
from tinygrad import Tensor, dtypes, Device

def benchmark_pure_runtime(name, wino_old, input_sizes, num_warmup=30, num_runs=100):
    """Benchmark with extensive warmup to ensure compilation is done"""
    os.environ['RANGEIFY'] = '1'
    os.environ['WINO_OLD'] = '1' if wino_old else '0'
    os.environ['WINO'] = '0' if wino_old else '1'

    results = {}

    for size in input_sizes:
        print(f"\n{name} - Testing {size}×{size} input...")

        # Create tensors
        x = Tensor.randn(1, 16, size, size, dtype=dtypes.float32).realize()
        w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()

        # EXTENSIVE warmup - measure first run (with compilation)
        first_run_time = None
        print(f"  Warmup phase (compiling + stabilizing)...")
        for i in range(num_warmup):
            Device[Device.DEFAULT].synchronize()
            start = time.perf_counter()
            out = x.conv2d(w, padding=1).realize()
            Device[Device.DEFAULT].synchronize()
            elapsed = (time.perf_counter() - start) * 1000

            if i == 0:
                first_run_time = elapsed
                print(f"    First run: {elapsed:.3f}ms (includes compilation)")
            elif i == num_warmup - 1:
                print(f"    Last warmup run: {elapsed:.3f}ms (pure runtime)")

        # Benchmark pure runtime
        print(f"  Benchmarking pure runtime ({num_runs} runs)...")
        times = []

        for i in range(num_runs):
            Device[Device.DEFAULT].synchronize()
            start = time.perf_counter()
            out = x.conv2d(w, padding=1).realize()
            Device[Device.DEFAULT].synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{num_runs} runs")

        results[size] = {
            'first_run': first_run_time,
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'compilation_overhead': first_run_time - np.min(times) if first_run_time else 0
        }

        print(f"  Results:")
        print(f"    First run (with compilation): {results[size]['first_run']:.3f}ms")
        print(f"    Min runtime: {results[size]['min']:.3f}ms")
        print(f"    Mean runtime: {results[size]['mean']:.3f}ms ± {results[size]['std']:.3f}ms")
        print(f"    Compilation overhead: ~{results[size]['compilation_overhead']:.3f}ms")

    return results

def main():
    print("="*70)
    print("PURE RUNTIME BENCHMARK: HYBRID vs OLD")
    print("="*70)
    print(f"Device: {Device.DEFAULT}")
    print(f"Warmup runs: 30 (extensive to fully compile and stabilize)")
    print(f"Benchmark runs: 100 (pure runtime, no compilation)")

    input_sizes = [24, 32, 48, 64, 96, 128]

    # Benchmark OLD
    print("\n" + "="*70)
    print("BENCHMARKING OLD (r_6_6_16 grid, coalesced writes)")
    print("="*70)
    old_results = benchmark_pure_runtime("OLD", True, input_sizes)

    # Benchmark HYBRID
    print("\n" + "="*70)
    print("BENCHMARKING HYBRID (r_16_6_6 grid, scattered writes)")
    print("="*70)
    hybrid_results = benchmark_pure_runtime("HYBRID", False, input_sizes)

    # Compare results
    print("\n" + "="*70)
    print("PURE RUNTIME COMPARISON (Post-Compilation)")
    print("="*70)
    print(f"\n{'Size':<10} {'OLD (ms)':<12} {'HYBRID (ms)':<12} {'Diff':<12} {'Status':<10}")
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

        print(f"{size}×{size:<5} {old_time:>6.3f}ms    {hybrid_time:>6.3f}ms    {slowdown:>+6.1f}%    {status}")

    # Compilation overhead analysis
    print("\n" + "="*70)
    print("COMPILATION OVERHEAD ANALYSIS")
    print("="*70)
    print(f"\n{'Size':<10} {'OLD Compile':<15} {'HYBRID Compile':<15} {'Difference':<15}")
    print("-"*70)

    for size in input_sizes:
        old_compile = old_results[size]['compilation_overhead']
        hybrid_compile = hybrid_results[size]['compilation_overhead']
        compile_diff = hybrid_compile - old_compile

        print(f"{size}×{size:<5} {old_compile:>6.3f}ms        {hybrid_compile:>6.3f}ms        {compile_diff:>+6.3f}ms")

    avg_old_compile = np.mean([old_results[s]['compilation_overhead'] for s in input_sizes])
    avg_hybrid_compile = np.mean([hybrid_results[s]['compilation_overhead'] for s in input_sizes])

    print(f"\nAverage compilation overhead:")
    print(f"  OLD:    {avg_old_compile:.3f}ms")
    print(f"  HYBRID: {avg_hybrid_compile:.3f}ms")
    print(f"  HYBRID has {avg_hybrid_compile - avg_old_compile:+.3f}ms more compilation time")

    # Runtime summary
    print("\n" + "="*70)
    print("PURE RUNTIME SUMMARY")
    print("="*70)

    avg_slowdown = np.mean([(hybrid_results[s]['min'] / old_results[s]['min'] - 1) * 100
                             for s in input_sizes])

    print(f"\nAverage pure runtime slowdown: {avg_slowdown:+.1f}%")

    if avg_slowdown < 5:
        print("\n✅ OUTSTANDING! HYBRID pure runtime within 5% of OLD.")
        print("   The scattered writes have negligible impact!")
    elif avg_slowdown < 10:
        print("\n✅ EXCELLENT! HYBRID pure runtime within 10% of OLD.")
        print("   Scattered writes have minimal impact. Ship it!")
    elif avg_slowdown < 20:
        print("\n⚠️ ACCEPTABLE. HYBRID pure runtime 10-20% slower.")
        print("   Scattered writes have moderate impact.")
    else:
        print("\n❌ SIGNIFICANT. HYBRID pure runtime >20% slower.")
        print("   Scattered writes are hurting performance.")

    # Detailed comparison
    print("\n" + "="*70)
    print("DETAILED RUNTIME COMPARISON")
    print("="*70)

    for size in input_sizes:
        print(f"\n{size}×{size} Input:")
        old_min = old_results[size]['min']
        hybrid_min = hybrid_results[size]['min']
        diff = hybrid_min - old_min
        pct = (diff / old_min) * 100

        print(f"  OLD min runtime:    {old_min:.3f}ms")
        print(f"  HYBRID min runtime: {hybrid_min:.3f}ms")
        print(f"  Difference:         {diff:+.3f}ms ({pct:+.1f}%)")

        # Show if HYBRID is better
        if diff < 0:
            print(f"  ⭐ HYBRID is FASTER by {abs(diff):.3f}ms!")

if __name__ == "__main__":
    main()
