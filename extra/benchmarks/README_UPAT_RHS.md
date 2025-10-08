# UPat RHS Performance Benchmark

## Overview

This benchmark validates that the new UPat RHS pattern matching infrastructure does not introduce unacceptable performance overhead compared to traditional lambda RHS patterns.

**Acceptance Criterion**: <20% performance regression in compiled mode

## Quick Start

```bash
# Run benchmark in compiled mode (default)
python3 extra/benchmarks/benchmark_upat_rhs.py

# Run in interpreted mode
UPAT_COMPILE=0 python3 extra/benchmarks/benchmark_upat_rhs.py

# Compare both modes
python3 extra/benchmarks/benchmark_upat_rhs.py --both-modes

# Custom iteration count
python3 extra/benchmarks/benchmark_upat_rhs.py --iterations 1000000
```

## Why This Benchmark Matters

Pattern matching is executed extensively during tinygrad's optimization passes, potentially running **millions of times per compilation**. Even small performance regressions can significantly impact user-facing compilation speed.

**Performance Impact Example:**
- Lambda RHS (compiled): ~250 ns/iter for simple patterns
- 20% regression = +50 ns/iter
- 1 million rewrites = +50 ms per compilation
- 100 compilations/day = +5 seconds/day of compilation overhead

## Benchmark Structure

### Pattern Complexity Levels

The benchmark tests three complexity levels to ensure comprehensive coverage:

#### 1. Simple Patterns
- **Identity**: `CONST(x) → x`
- **Additive identity**: `x + 0 → x`
- **Multiplicative identity**: `x * 1 → x`

**Characteristics**: Single operation, single variable

#### 2. Moderate Patterns
- **Multiplicative cancellation**: `(x * y) / y → x`
- **Addition doubling**: `x + x → x * 2`
- **Multiplicative annihilation**: `x * 0 → 0`

**Characteristics**: Multiple variables, algebraic simplification

#### 3. Complex Patterns
- **Nested cancellation**: `((x*a)*b)/(a*b) → x`
- **Boolean distributive law**: `(x&y)|(x&z) → x&(y|z)`
- **Associativity**: `(x+a)+b → x+(a+b)`

**Characteristics**: Deeply nested operations, multiple transformations

## Test Methodology

### Warmup Phase
- 1,000 iterations to warm up JIT/caches
- Results discarded

### Benchmark Phase
- 100,000 iterations (default)
- Measures time.perf_counter() before/after
- Calculates average time per iteration

### Dual-Mode Testing
- **Compiled Mode** (UPAT_COMPILE=1): Uses optimized `upat_compile` matchers
- **Interpreted Mode** (UPAT_COMPILE=0): Uses Python `upat_interpret` matchers

## Current Results (Lambda Baseline)

### Compiled Mode Performance

| Complexity | Avg Time (ns/iter) | Throughput (ops/sec) |
|-----------|-------------------|---------------------|
| Simple | 251 | 4.0M |
| Moderate | 1,416 | 710K |
| Complex | 1,954 | 513K |

### Compiled vs Interpreted Speedup

Compiled mode is **4-5x faster** than interpreted mode:
- Simple patterns: 4.6x faster
- Moderate patterns: 2.3x faster
- Complex patterns: 4.2x faster

**See `BENCHMARK_UPAT_RHS_RESULTS.md` for detailed results.**

## Adding UPat RHS Benchmarks

Once Tasks 1 & 2 are complete, add parallel UPat RHS versions:

```python
def benchmark_simple_add_zero_upat(iterations: int = 100000):
  """Benchmark: x + 0 → x (UPat RHS)"""
  pm = PatternMatcher([(UPat.var("x") + 0, UPat.var("x"))])
  test_uop = UOp.const(dtypes.int, 42) + 0
  result = benchmark_pattern("Simple: x + 0 → x (UPat)", pm, test_uop, iterations)
  result.pattern_type = "upat"
  return result

# Add to run_all_benchmarks():
results.append(benchmark_simple_add_zero_lambda(iterations))
results.append(benchmark_simple_add_zero_upat(iterations))  # Compare directly
```

## Validation Checklist

- [ ] UPat RHS benchmarks added for all 9 patterns
- [ ] All benchmarks run successfully in compiled mode
- [ ] All benchmarks run successfully in interpreted mode
- [ ] Performance regression calculated for each pattern
- [ ] All patterns meet <20% regression criterion
- [ ] If >20% regression, optimization strategy documented

## Optimization Strategies

If UPat RHS exceeds 20% regression:

### Strategy 1: Optimize `reconstruct_uop`
- Cache dtype inference
- Inline simple patterns
- Pre-compute UOp templates

### Strategy 2: Enhanced Compilation
- Generate specialized reconstruction code in `upat_compile`
- Combine match + reconstruction into single optimized function

### Strategy 3: Complexity Thresholds
- Simple UPat RHS: Direct substitution
- Complex UPat RHS: Fall back to lambda if needed

### Strategy 4: Profile-Guided Optimization
- Identify hot paths with profiling
- Optimize top 20% of patterns (80/20 rule)

**See `BENCHMARK_UPAT_RHS_RESULTS.md` for detailed optimization analysis.**

## Files

- `benchmark_upat_rhs.py` - Main benchmark script
- `BENCHMARK_UPAT_RHS_RESULTS.md` - Detailed results and analysis
- `README_UPAT_RHS.md` - This file

## Dependencies

- tinygrad (parent directory)
- Python 3.8+
- Standard library only (time, os, sys, argparse)

## Contributing

When adding new benchmark patterns:

1. Add to appropriate complexity level
2. Include both lambda and UPat RHS versions (when available)
3. Provide descriptive name in format: `Complexity: LHS → RHS`
4. Ensure pattern actually matches the test UOp
5. Document expected performance characteristics

## Questions?

See implementation documentation in:
- `/home/molaco/Documents/tinygrad/IMPL2.md` - UPat RHS implementation guide
- `/home/molaco/Documents/tinygrad/tasks.yaml` - Task breakdown
