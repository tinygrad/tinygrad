#!/usr/bin/env python3
"""
Debug which axes are which to understand grid organization
"""
import os
os.environ['DEBUG'] = '5'
os.environ['RANGEIFY'] = '1'
os.environ['WINO'] = '1'

from tinygrad import Tensor, dtypes

# Small test to see axis names
x = Tensor.randn(1, 16, 64, 64, dtype=dtypes.float32).realize()
w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()
out = x.conv2d(w, padding=1)
# Don't realize - just check what happens
print("Input shape:", x.shape)
print("Weight shape:", w.shape)
print("Expected output:", out.shape)

# The axes should be:
# - Batch: 1
# - Channels out: 16
# - Spatial: 64×64 → tiled to 16×16 tiles of 4×4
#   → 16×16 = 256 spatial positions

print("\nExpected grid organization for OLD (r_6_6_16):")
print("  gid.x = 16 (channels - cout)")
print("  gid.y = 6 (first spatial tile dim)")
print("  gid.z = 6 (second spatial tile dim)")
print("  Total: 16 × 6 × 6 = 576 threadgroups")

print("\nCurrent NEW (r_16_6_6) has them swapped!")
