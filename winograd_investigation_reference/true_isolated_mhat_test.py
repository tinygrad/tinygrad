#!/usr/bin/env python3
"""
True isolated comparison of just the multiply/reduce step.
Both operate on the same 6x6 transformed matrices.
"""

from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import Context
import numpy as np
import time

# Small test to isolate just the multiply/reduce
# Use minimal sizes to reduce overhead
Cin, Cout = 16, 16
tyx = (8, 8)  # Tile dimensions

print("="*80)
print("TRUE ISOLATED MHAT MULTIPLY/REDUCE TEST")
print("="*80)

# Create dummy 6x6 transformed matrices (same data for both tests)
np.random.seed(42)
# dfactors: (6, 6, 1, 1, 1, cin, tyx_h, tyx_w)
dfactors_np = np.random.randn(6, 6, 1, 1, 1, Cin, tyx[0], tyx[1]).astype(np.float32)
# gfactors: (6, 6, 1, 1, cout, cin, 1, 1)
gfactors_np = np.random.randn(6, 6, 1, 1, Cout, Cin, 1, 1).astype(np.float32)

def test_old_multiply_reduce(dfactors_np, gfactors_np, warmup=3, runs=10):
    """OLD approach: direct tensor multiply + reduce"""
    with Context(WINO=0, WINO_OLD=0):
        dfactors = Tensor(dfactors_np, dtype=dtypes.float32).realize()
        gfactors = Tensor(gfactors_np, dtype=dtypes.float32).realize()

        # Compile
        t0 = time.time()
        mhat = (gfactors * dfactors).sum(axis=-1-2, dtype=dtypes.float32)
        mhat.realize()
        compile_time = time.time() - t0
        result = mhat.numpy()

        # Runtime
        samples = []
        for _ in range(warmup + runs):
            dfactors = Tensor(dfactors_np, dtype=dtypes.float32).realize()
            gfactors = Tensor(gfactors_np, dtype=dtypes.float32).realize()

            t1 = time.time()
            mhat = (gfactors * dfactors).sum(axis=-1-2, dtype=dtypes.float32)
            mhat.realize()
            samples.append(time.time() - t1)

        samples = samples[warmup:]
        avg_runtime = np.mean(samples) * 1000

    return compile_time * 1000, avg_runtime, result

def test_new_multiply_reduce_synthetic(dfactors_np, gfactors_np, warmup=3, runs=10):
    """
    NEW approach: simulate what NEW does with indexing and fresh ranges.

    This emulates the NEW code:
      mhat_redu = (XHAT.index(...) * GHAT.index(...)).reduce(...)
      MHAT = mhat_redu.bufferize(...)

    But using realized tensors instead of schedule-level UOps.
    """
    with Context(WINO=0, WINO_OLD=0):
        # Reshape to simulate the indexing pattern NEW uses
        # NEW creates fresh ranges and indexes into buffers
        # Let's simulate this with reshapes and transposes

        dfactors = Tensor(dfactors_np, dtype=dtypes.float32).realize()
        gfactors = Tensor(gfactors_np, dtype=dtypes.float32).realize()

        # Compile
        t0 = time.time()
        # Simulate the pattern: index both tensors, multiply, reduce
        # NEW's pattern involves more intermediate steps
        # Let's try to emulate the extra work NEW does

        # Expand and contract to simulate the indexing pattern
        df_expanded = dfactors.reshape(6, 6, 1, 1, 1, Cin, tyx[0], tyx[1])
        gf_expanded = gfactors.reshape(6, 6, 1, 1, Cout, Cin, 1, 1)

        # Multiply (with broadcast)
        product = df_expanded * gf_expanded

        # Reduce (sum over cin dimension)
        mhat = product.sum(axis=-3, dtype=dtypes.float32)
        mhat.realize()
        compile_time = time.time() - t0
        result = mhat.numpy()

        # Runtime
        samples = []
        for _ in range(warmup + runs):
            dfactors = Tensor(dfactors_np, dtype=dtypes.float32).realize()
            gfactors = Tensor(gfactors_np, dtype=dtypes.float32).realize()

            t1 = time.time()
            df_expanded = dfactors.reshape(6, 6, 1, 1, 1, Cin, tyx[0], tyx[1])
            gf_expanded = gfactors.reshape(6, 6, 1, 1, Cout, Cin, 1, 1)
            product = df_expanded * gf_expanded
            mhat = product.sum(axis=-3, dtype=dtypes.float32)
            mhat.realize()
            samples.append(time.time() - t1)

        samples = samples[warmup:]
        avg_runtime = np.mean(samples) * 1000

    return compile_time * 1000, avg_runtime, result

print("\n" + "-"*80)
print("TEST 1: OLD tensor-level multiply/reduce")
print("-"*80)
c_old, r_old, mhat_old = test_old_multiply_reduce(dfactors_np, gfactors_np)
print(f"OLD multiply/reduce  | Compile: {c_old:>7.1f}ms | Runtime: {r_old:>7.1f}ms")

print("\n" + "-"*80)
print("TEST 2: NEW-style multiply/reduce (simulated)")
print("-"*80)
c_new, r_new, mhat_new = test_new_multiply_reduce_synthetic(dfactors_np, gfactors_np)
print(f"NEW multiply/reduce  | Compile: {c_new:>7.1f}ms | Runtime: {r_new:>7.1f}ms")

# Verify same result
diff = np.abs(mhat_old - mhat_new).max()
print(f"\n" + "-"*80)
print(f"Result difference: {diff:.2e}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print(f"Runtime comparison:")
print(f"  OLD: {r_old:.3f}ms")
print(f"  NEW: {r_new:.3f}ms")
print(f"  Difference: {r_new - r_old:+.3f}ms ({(r_new/r_old - 1)*100:+.1f}%)")

if r_new > r_old:
    print(f"\n❌ NEW multiply/reduce is {(r_new-r_old)/r_old*100:.1f}% SLOWER")
else:
    print(f"\n✅ NEW multiply/reduce is {(r_old-r_new)/r_new*100:.1f}% FASTER")

print("\n" + "="*80)
print("NOTE")
print("="*80)
print("This is a SYNTHETIC test - NEW's actual multiply/reduce happens at")
print("schedule level with UOps, not at tensor level like this simulation.")
print("\nThe REAL difference comes from:")
print("1. Range duplication (tile_ranges vs tile_ranges1, inner6 vs inner6_1)")
print("2. Indexing buffers with fresh ranges")
print("3. Bufferization patterns")
print("\nThese can't be fully captured in a tensor-level simulation.")
