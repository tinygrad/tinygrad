#!/usr/bin/env python3
"""
Isolated test of just the MHAT multiply/reduce step.
Both OLD and NEW operate on the same precomputed dfactors/gfactors data.
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np
import time

# Test parameters
B, Cin, Cout, H, W = 1, 16, 16, 32, 32
HWI = (6, 6)  # Winograd input tile size

# Winograd matrices
winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0],
               [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6],
              [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]

def _apply_winograd_matrix(mat, t, ndim):
    """Helper from tensor.py"""
    for i in range(ndim): t = t.transpose(i, -ndim+i)
    for i in range(ndim):
        t = Tensor(mat, dtype=t.dtype, device=t.device) @ t.transpose(-ndim, -1)
        t = t.transpose(-ndim, -1).transpose(-1, i-ndim)
    return t

def test_old_mhat_multiply(dfactors, gfactors, label="OLD"):
    """OLD tensor-level approach: multiply + reduce"""
    with Context(WINO=0, WINO_OLD=0):  # Disable winograd to measure just this operation
        # dfactors: (6, 6, bs, groups, 1, cin, tyx_h, tyx_w)
        # gfactors: (6, 6, 1, groups, rcout, cin, 1, 1)

        t0 = time.time()
        # This is what OLD does in Phase 3 (tensor.py line 2588)
        mhat = (gfactors * dfactors).sum(axis=-1-2, dtype=dtypes.float32)  # sum over cin dimension
        mhat.realize()
        compile_time = time.time() - t0

        # Runtime measurement
        runtime_samples = []
        for _ in range(13):  # 3 warmup + 10 runs
            t1 = time.time()
            mhat = (gfactors * dfactors).sum(axis=-1-2, dtype=dtypes.float32)
            mhat.realize()
            runtime_samples.append(time.time() - t1)

        runtime_samples = runtime_samples[3:]
        avg_runtime = np.mean(runtime_samples) * 1000

    print(f"{label:<20} | Compile: {compile_time*1000:>7.1f}ms | Runtime: {avg_runtime:>7.1f}ms")
    return compile_time * 1000, avg_runtime, mhat.numpy()

def test_new_mhat_multiply(x_np, w_np, label="NEW"):
    """
    NEW approach: trigger winograd rewrite with a real conv operation.
    This will go through the full NEW pipeline including the multiply/reduce step.
    """
    with Context(WINO=1, WINO_OLD=0):
        x = Tensor(x_np, dtype=dtypes.float32).realize()
        w = Tensor(w_np, dtype=dtypes.float32).realize()

        t0 = time.time()
        out = x.conv2d(w, padding=1)
        out.realize()
        compile_time = time.time() - t0

        # Runtime measurement
        runtime_samples = []
        for _ in range(13):
            x = Tensor(x_np, dtype=dtypes.float32).realize()
            w = Tensor(w_np, dtype=dtypes.float32).realize()
            t1 = time.time()
            out = x.conv2d(w, padding=1)
            out.realize()
            runtime_samples.append(time.time() - t1)

        runtime_samples = runtime_samples[3:]
        avg_runtime = np.mean(runtime_samples) * 1000

    print(f"{label:<20} | Compile: {compile_time*1000:>7.1f}ms | Runtime: {avg_runtime:>7.1f}ms")
    return compile_time * 1000, avg_runtime, out.numpy()

print("="*80)
print("ISOLATED MHAT MULTIPLY/REDUCE TEST")
print("="*80)

# Create test data
np.random.seed(42)
groups = 1
rcout = Cout // groups

# Step 1: Create inputs that will produce dfactors and gfactors
x_np = np.random.randn(B, Cin, H, W).astype(np.float32)
w_np = np.random.randn(Cout, Cin, 3, 3).astype(np.float32)

# Step 2: Manually compute dfactors and gfactors using OLD's approach
# This simulates what happens before the multiply step
print("\nPrecomputing dfactors and gfactors using OLD's method...")

with Context(WINO=0, WINO_OLD=0):
    # Compute d (pooled input)
    pads = [[1, 1 + (-(H + 2 - 2) % 4)], [1, 1 + (-(W + 2 - 2) % 4)]]
    x_tensor = Tensor(x_np, dtype=dtypes.float32)
    d = x_tensor.pad(sum(pads, []))._pool(HWI, (4, 4))
    d = d.permute(4, 5, 0, 1, 2, 3)  # Move HW to front
    tyx = d.shape[-2:]

    # Apply Bt transform
    dfactors = _apply_winograd_matrix(winograd_Bt, d, 2).reshape(*HWI, B, groups, 1, Cin, *tyx)
    dfactors_np = dfactors.realize().numpy()

    # Compute g (weights)
    w_tensor = Tensor(w_np, dtype=dtypes.float32)
    g = w_tensor.permute(2, 3, 0, 1)  # Move HW to front

    # Apply G transform
    gfactors = _apply_winograd_matrix(winograd_G, g, 2).reshape(*HWI, 1, groups, rcout, Cin, *([1]*len(tyx)))
    gfactors_np = gfactors.realize().numpy()

print(f"dfactors shape: {dfactors_np.shape}")
print(f"gfactors shape: {gfactors_np.shape}")

# Step 3: Test OLD's multiply/reduce
print("\n" + "-"*80)
print("TEST 1: OLD tensor-level multiply/reduce")
print("-"*80)
c_old, r_old, mhat_old = test_old_mhat_multiply(dfactors, gfactors, "OLD multiply/reduce")

# Step 4: Test NEW's full pipeline (includes multiply/reduce)
print("\n" + "-"*80)
print("TEST 2: NEW full pipeline (includes multiply/reduce)")
print("-"*80)
c_new, r_new, out_new = test_new_mhat_multiply(x_np, w_np, "NEW full pipeline")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print(f"\nRuntime comparison:")
print(f"  OLD multiply/reduce only: {r_old:.1f}ms")
print(f"  NEW full pipeline:         {r_new:.1f}ms")

if r_new > r_old:
    print(f"  ❌ NEW full pipeline is {(r_new-r_old)/r_old*100:.1f}% SLOWER (+{r_new-r_old:.1f}ms)")
    print(f"\n  Note: NEW includes ALL phases (XHAT+GHAT+multiply+output transform)")
    print(f"        OLD only includes the multiply/reduce step")
    print(f"        The difference ({r_new-r_old:.1f}ms) represents overhead from other phases")
else:
    print(f"  ✅ NEW is {(r_old-r_new)/r_new*100:.1f}% FASTER")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("This test shows that we can't easily isolate just the multiply/reduce step")
print("for NEW because it's integrated into the full winograd rewrite.")
print("\nTo properly compare, we'd need to modify the NEW implementation to accept")
print("pre-transformed inputs, which would require changes to the rewrite pattern.")
