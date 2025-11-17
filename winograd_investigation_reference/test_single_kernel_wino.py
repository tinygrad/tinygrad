#!/usr/bin/env python3
"""
Test if removing MHAT bufferize allows single-kernel winograd
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
import numpy as np

# Small shape to reduce debug output
B, Cin, Cout, H, W = 1, 8, 8, 32, 32

print("="*80)
print("TESTING SINGLE-KERNEL WINOGRAD (NO MHAT BUFFERIZE)")
print(f"Shape: B={B}, Cin={Cin}, Cout={Cout}, H={H}×{W}")
print("="*80)

# Create test data
np.random.seed(42)
x = Tensor.randn(B, Cin, H, W, dtype=dtypes.float32).realize()
w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()

# Run with new winograd
print("\nRunning with WINO=1 (modified - no MHAT bufferize)...")
with Context(WINO=1, WINO_OLD=0):
    try:
        out = x.conv2d(w, padding=1)
        out.realize()
        result = out.numpy()
        print("✅ SUCCESS - Kernel compiled and executed!")
        print(f"   Output shape: {result.shape}")
        print(f"   Output range: [{result.min():.4f}, {result.max():.4f}]")
    except Exception as e:
        print(f"❌ FAILED with error:")
        print(f"   {type(e).__name__}: {e}")

# Compare with baseline for correctness
print("\nRunning baseline (no winograd) for comparison...")
with Context(WINO=0, WINO_OLD=0):
    out_base = x.conv2d(w, padding=1)
    out_base.realize()
    result_base = out_base.numpy()

# Check accuracy
if 'result' in locals():
    diff = np.abs(result - result_base).max()
    print(f"\n{'='*80}")
    print(f"Accuracy check: max error = {diff:.2e}")
    if diff < 1e-2:
        print("✅ Results match baseline!")
    else:
        print("⚠️  Results differ from baseline")
