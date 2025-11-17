#!/usr/bin/env python3
"""Test if NEW produces same numerical output as OLD"""
import numpy as np
from tinygrad import Tensor, dtypes
import os

# Test with both OLD and NEW
for wino_old in [True, False]:
    os.environ['RANGEIFY'] = '1'
    os.environ['WINO_OLD'] = '1' if wino_old else '0'
    os.environ['WINO'] = '0' if wino_old else '1'

    # Fixed seed for reproducibility
    Tensor.manual_seed(42)

    x = Tensor.randn(1, 16, 64, 64, dtype=dtypes.float32).realize()
    w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()
    out = x.conv2d(w, padding=1).realize()

    result = out.numpy()

    if wino_old:
        old_result = result
        print(f"OLD result shape: {result.shape}")
        print(f"OLD result range: [{result.min():.4f}, {result.max():.4f}]")
        print(f"OLD result mean: {result.mean():.4f}, std: {result.std():.4f}")
    else:
        new_result = result
        print(f"\nNEW result shape: {result.shape}")
        print(f"NEW result range: [{result.min():.4f}, {result.max():.4f}]")
        print(f"NEW result mean: {result.mean():.4f}, std: {result.std():.4f}")

# Compare
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
diff = np.abs(old_result - new_result)
max_diff = diff.max()
mean_diff = diff.mean()
print(f"Max absolute difference: {max_diff:.2e}")
print(f"Mean absolute difference: {mean_diff:.2e}")
print(f"Relative error: {(max_diff / (np.abs(old_result).max() + 1e-10)):.2e}")

if max_diff < 1e-4:
    print("\n✅ PASS! Results are numerically identical (within floating point precision)")
else:
    print(f"\n❌ FAIL! Results differ by {max_diff:.2e}")
