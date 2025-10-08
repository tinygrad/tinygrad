#!/usr/bin/env python3
"""
Performance Benchmark: UPat RHS Pattern Matching

This benchmark validates that the new UPat RHS pattern matching infrastructure
does not introduce unacceptable performance overhead compared to traditional
lambda RHS patterns.

Acceptance Criteria: <20% performance regression in compiled mode (UPAT_COMPILE=1)

Usage:
  # Run in compiled mode (default)
  python3 extra/benchmarks/benchmark_upat_rhs.py

  # Run in interpreted mode
  UPAT_COMPILE=0 python3 extra/benchmarks/benchmark_upat_rhs.py

  # Run both modes
  python3 extra/benchmarks/benchmark_upat_rhs.py --both-modes
"""

import time
import os
import sys
import argparse
from typing import Callable
from dataclasses import dataclass

# Add tinygrad to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tinygrad.uop.ops import UOp, UPat, PatternMatcher
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv

@dataclass
class BenchmarkResult:
  """Results from a single benchmark run"""
  name: str
  pattern_type: str  # "lambda" or "upat"
  compile_mode: str  # "compiled" or "interpreted"
  iterations: int
  total_time_ms: float
  avg_time_ns: float

  def __str__(self):
    return f"{self.name:30s} | {self.pattern_type:6s} | {self.compile_mode:11s} | {self.avg_time_ns:8.2f} ns/iter | {self.total_time_ms:8.2f} ms total"


def benchmark_pattern(name: str, matcher: PatternMatcher, test_uop: UOp,
                     iterations: int = 100000, warmup: int = 1000) -> BenchmarkResult:
  """
  Benchmark a pattern matcher's rewrite performance.

  Args:
    name: Descriptive name for the benchmark
    matcher: PatternMatcher to test
    test_uop: UOp to rewrite (should match the pattern)
    iterations: Number of iterations to run
    warmup: Number of warmup iterations

  Returns:
    BenchmarkResult with timing data
  """
  # Determine pattern type from matcher
  pattern_type = "lambda"  # Default assumption

  # Warmup phase
  for _ in range(warmup):
    matcher.rewrite(test_uop)

  # Benchmark phase
  start = time.perf_counter()
  for _ in range(iterations):
    matcher.rewrite(test_uop)
  end = time.perf_counter()

  total_time_ms = (end - start) * 1000
  avg_time_ns = (end - start) * 1e9 / iterations
  # Check the actual environment variable at runtime
  compile_mode = "compiled" if os.environ.get("UPAT_COMPILE", "1") == "1" else "interpreted"

  return BenchmarkResult(
    name=name,
    pattern_type=pattern_type,
    compile_mode=compile_mode,
    iterations=iterations,
    total_time_ms=total_time_ms,
    avg_time_ns=avg_time_ns
  )


# ============================================================================
# SIMPLE PATTERNS - Basic identity and single-operation patterns
# ============================================================================

def benchmark_simple_identity_lambda(iterations: int = 100000):
  """Benchmark: CONST(x) → x (lambda RHS)"""
  from tinygrad.uop import Ops
  pm = PatternMatcher([(UPat(Ops.CONST, name="x"), lambda x: x)])
  test_uop = UOp.const(dtypes.int, 42)
  return benchmark_pattern("Simple: CONST(x) → x", pm, test_uop, iterations)


def benchmark_simple_add_zero_lambda(iterations: int = 100000):
  """Benchmark: x + 0 → x (lambda RHS)"""
  pm = PatternMatcher([(UPat.var("x") + 0, lambda x: x)])
  test_uop = UOp.const(dtypes.int, 42) + 0
  return benchmark_pattern("Simple: x + 0 → x", pm, test_uop, iterations)


def benchmark_simple_mul_one_lambda(iterations: int = 100000):
  """Benchmark: x * 1 → x (lambda RHS)"""
  pm = PatternMatcher([(UPat.var("x") * 1, lambda x: x)])
  test_uop = UOp.const(dtypes.int, 42) * 1
  return benchmark_pattern("Simple: x * 1 → x", pm, test_uop, iterations)


# ============================================================================
# MODERATE PATTERNS - Algebraic simplifications with multiple variables
# ============================================================================

def benchmark_moderate_div_mul_lambda(iterations: int = 100000):
  """Benchmark: (x * y) / y → x (lambda RHS)"""
  pm = PatternMatcher([
    ((UPat.var("x") * UPat.var("y")) / UPat.var("y"), lambda x, y: x)
  ])
  x = UOp.const(dtypes.int, 10)
  y = UOp.const(dtypes.int, 3)
  test_uop = (x * y) / y
  return benchmark_pattern("Moderate: (x*y)/y → x", pm, test_uop, iterations)


def benchmark_moderate_double_add_lambda(iterations: int = 100000):
  """Benchmark: x + x → x * 2 (lambda RHS)"""
  pm = PatternMatcher([
    (UPat.var("x") + UPat.var("x"), lambda x: x * 2)
  ])
  x = UOp.const(dtypes.int, 21)
  test_uop = x + x
  return benchmark_pattern("Moderate: x + x → x*2", pm, test_uop, iterations)


def benchmark_moderate_mul_zero_lambda(iterations: int = 100000):
  """Benchmark: x * 0 → 0 (lambda RHS)"""
  pm = PatternMatcher([
    (UPat.var("x") * 0, lambda x: UOp.const(x.dtype, 0))
  ])
  x = UOp.const(dtypes.int, 42)
  test_uop = x * 0
  return benchmark_pattern("Moderate: x * 0 → 0", pm, test_uop, iterations)


# ============================================================================
# COMPLEX PATTERNS - Nested operations and multiple transformations
# ============================================================================

def benchmark_complex_nested_mul_div_lambda(iterations: int = 100000):
  """Benchmark: ((x * a) * b) / (a * b) → x (lambda RHS)"""
  pm = PatternMatcher([
    (((UPat.var("x") * UPat.var("a")) * UPat.var("b")) / (UPat.var("a") * UPat.var("b")),
     lambda x, a, b: x)
  ])
  x = UOp.const(dtypes.int, 5)
  a = UOp.const(dtypes.int, 2)
  b = UOp.const(dtypes.int, 3)
  test_uop = ((x * a) * b) / (a * b)
  return benchmark_pattern("Complex: ((x*a)*b)/(a*b) → x", pm, test_uop, iterations)


def benchmark_complex_bool_algebra_lambda(iterations: int = 100000):
  """Benchmark: (x & y) | (x & z) → x & (y | z) (lambda RHS)"""
  pm = PatternMatcher([
    ((UPat.var("x") & UPat.var("y")) | (UPat.var("x") & UPat.var("z")),
     lambda x, y, z: x & (y | z))
  ])
  x = UOp.const(dtypes.bool, True)
  y = UOp.const(dtypes.bool, False)
  z = UOp.const(dtypes.bool, True)
  test_uop = (x & y) | (x & z)
  return benchmark_pattern("Complex: (x&y)|(x&z) → x&(y|z)", pm, test_uop, iterations)


def benchmark_complex_nested_add_lambda(iterations: int = 100000):
  """Benchmark: (x + a) + b → x + (a + b) (lambda RHS) - simpler associativity"""
  pm = PatternMatcher([
    ((UPat.var("x") + UPat.var("a")) + UPat.var("b"),
     lambda x, a, b: x + (a + b))
  ])
  x = UOp.const(dtypes.int, 1)
  a = UOp.const(dtypes.int, 2)
  b = UOp.const(dtypes.int, 3)
  test_uop = (x + a) + b
  return benchmark_pattern("Complex: (x+a)+b → x+(a+b)", pm, test_uop, iterations)


# ============================================================================
# NOTE: UPat RHS benchmarks would go here once Tasks 1 & 2 are complete
# ============================================================================
#
# Example of what UPat RHS benchmarks would look like:
#
# def benchmark_simple_add_zero_upat(iterations: int = 100000):
#   """Benchmark: x + 0 → x (UPat RHS)"""
#   pm = PatternMatcher([(UPat.var("x") + 0, UPat.var("x"))])
#   test_uop = UOp.const(dtypes.int, 42) + 0
#   result = benchmark_pattern("Simple: x + 0 → x (UPat)", pm, test_uop, iterations)
#   result.pattern_type = "upat"
#   return result
#
# Once UPat RHS is implemented, we can add parallel UPat versions of each
# lambda benchmark above and compare performance.
# ============================================================================


def run_all_benchmarks(iterations: int = 100000) -> list[BenchmarkResult]:
  """Run all benchmarks and return results"""
  results = []

  print(f"\n{'='*90}")
  print(f"Running benchmarks with {iterations:,} iterations each...")
  print(f"Compile mode: {'COMPILED' if os.environ.get('UPAT_COMPILE', '1') == '1' else 'INTERPRETED'}")
  print(f"{'='*90}\n")

  # Simple patterns
  print("SIMPLE PATTERNS:")
  results.append(benchmark_simple_identity_lambda(iterations))
  results.append(benchmark_simple_add_zero_lambda(iterations))
  results.append(benchmark_simple_mul_one_lambda(iterations))

  # Moderate patterns
  print("\nMODERATE PATTERNS:")
  results.append(benchmark_moderate_div_mul_lambda(iterations))
  results.append(benchmark_moderate_double_add_lambda(iterations))
  results.append(benchmark_moderate_mul_zero_lambda(iterations))

  # Complex patterns
  print("\nCOMPLEX PATTERNS:")
  results.append(benchmark_complex_nested_mul_div_lambda(iterations))
  results.append(benchmark_complex_bool_algebra_lambda(iterations))
  results.append(benchmark_complex_nested_add_lambda(iterations))

  return results


def print_summary(results: list[BenchmarkResult]):
  """Print formatted summary of benchmark results"""
  print(f"\n{'='*90}")
  print("BENCHMARK RESULTS SUMMARY")
  print(f"{'='*90}")
  print(f"{'Pattern':30s} | {'Type':6s} | {'Mode':11s} | {'Avg Time':^15s} | {'Total Time':^15s}")
  print(f"{'-'*90}")

  for result in results:
    print(result)

  print(f"{'='*90}")

  # Calculate averages by complexity level
  simple = [r for r in results if r.name.startswith("Simple:")]
  moderate = [r for r in results if r.name.startswith("Moderate:")]
  complex_results = [r for r in results if r.name.startswith("Complex:")]

  if simple:
    avg_simple = sum(r.avg_time_ns for r in simple) / len(simple)
    print(f"\nAverage Simple Pattern Time:   {avg_simple:8.2f} ns/iter")

  if moderate:
    avg_moderate = sum(r.avg_time_ns for r in moderate) / len(moderate)
    print(f"Average Moderate Pattern Time: {avg_moderate:8.2f} ns/iter")

  if complex_results:
    avg_complex = sum(r.avg_time_ns for r in complex_results) / len(complex_results)
    print(f"Average Complex Pattern Time:  {avg_complex:8.2f} ns/iter")

  print()


def compare_modes():
  """Run benchmarks in both compiled and interpreted modes and compare"""
  print("\n" + "="*90)
  print("DUAL-MODE COMPARISON: Testing both UPAT_COMPILE=1 and UPAT_COMPILE=0")
  print("="*90)

  # Save current environment
  original_compile = os.environ.get("UPAT_COMPILE", "1")

  # Run in compiled mode
  os.environ["UPAT_COMPILE"] = "1"
  print("\n" + ">"*90)
  print("> COMPILED MODE (UPAT_COMPILE=1)")
  print(">"*90)
  compiled_results = run_all_benchmarks(iterations=100000)
  print_summary(compiled_results)

  # Run in interpreted mode
  os.environ["UPAT_COMPILE"] = "0"
  print("\n" + ">"*90)
  print("> INTERPRETED MODE (UPAT_COMPILE=0)")
  print(">"*90)
  interpreted_results = run_all_benchmarks(iterations=100000)
  print_summary(interpreted_results)

  # Restore environment
  os.environ["UPAT_COMPILE"] = original_compile

  # Compare results
  print("\n" + "="*90)
  print("SPEEDUP ANALYSIS (Compiled vs Interpreted)")
  print("="*90)
  print(f"{'Pattern':30s} | {'Compiled (ns)':14s} | {'Interpreted (ns)':16s} | {'Speedup':8s}")
  print("-"*90)

  for comp, interp in zip(compiled_results, interpreted_results):
    assert comp.name == interp.name, "Result mismatch"
    speedup = interp.avg_time_ns / comp.avg_time_ns if comp.avg_time_ns > 0 else 0
    print(f"{comp.name:30s} | {comp.avg_time_ns:14.2f} | {interp.avg_time_ns:16.2f} | {speedup:8.2f}x")

  avg_speedup = sum(interp.avg_time_ns / comp.avg_time_ns
                   for comp, interp in zip(compiled_results, interpreted_results)
                   if comp.avg_time_ns > 0) / len(compiled_results)

  print("="*90)
  print(f"\nAverage Compiled Speedup: {avg_speedup:.2f}x faster than interpreted")
  print()


def main():
  parser = argparse.ArgumentParser(
    description="Benchmark UPat RHS pattern matching performance",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Run in current mode (respects UPAT_COMPILE env var)
  python3 extra/benchmarks/benchmark_upat_rhs.py

  # Run in both compiled and interpreted modes
  python3 extra/benchmarks/benchmark_upat_rhs.py --both-modes

  # Run with custom iteration count
  python3 extra/benchmarks/benchmark_upat_rhs.py --iterations 1000000

Note: UPat RHS benchmarks require Tasks 1 & 2 to be completed first.
      Currently only lambda RHS benchmarks are available.
    """
  )

  parser.add_argument(
    "--iterations", "-n",
    type=int,
    default=100000,
    help="Number of iterations per benchmark (default: 100000)"
  )

  parser.add_argument(
    "--both-modes", "-b",
    action="store_true",
    help="Run benchmarks in both compiled and interpreted modes"
  )

  args = parser.parse_args()

  if args.both_modes:
    compare_modes()
  else:
    results = run_all_benchmarks(args.iterations)
    print_summary(results)

    print("\nNOTE: This benchmark currently only tests lambda RHS patterns.")
    print("      UPat RHS benchmarks will be added once Tasks 1 & 2 are complete.")
    print("\nACCEPTANCE CRITERIA:")
    print("  - UPat RHS should be <20% slower than lambda RHS in compiled mode")
    print("  - If regression >20%, optimization strategy should be documented")
    print()


if __name__ == "__main__":
  main()
