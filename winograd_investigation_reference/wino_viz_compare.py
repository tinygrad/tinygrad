#!/usr/bin/env python3
"""
Minimal script to compare NEW (unified kron) vs OLD (tensor.py) Winograd implementations.
Use with VIZ to visually inspect the compiled kernels.

Usage:
  # Run NEW version and open VIZ
  WINO=1 WINO_OLD=0 VIZ=1 python3 wino_viz_compare.py

  # Run OLD version and open VIZ
  WINO=0 WINO_OLD=1 VIZ=1 python3 wino_viz_compare.py

  # Run baseline (no winograd) for comparison
  WINO=0 WINO_OLD=0 VIZ=1 python3 wino_viz_compare.py
"""

from tinygrad import Tensor, dtypes
from tinygrad.helpers import WINO, WINO_OLD
import numpy as np
import time

# Representative large shape where OLD is faster in both compile and runtime
# From benchmark: OLD compile 113ms, NEW compile 137ms
#                 OLD run 2.9ms (0.40× speedup), NEW run 7.1ms (0.16× speedup)
B, Cin, Cout, H, W = 1, 64, 64, 64, 64

print(f"\n{'='*80}")
print(f"Winograd Comparison - Shape: B={B} Cin={Cin} Cout={Cout} H={H}x{W}")
print(f"{'='*80}")

# Create same tensors for fair comparison
np.random.seed(42)
x = Tensor.randn(B, Cin, H, W, dtype=dtypes.float32).realize()
w = Tensor.randn(Cout, Cin, 3, 3, dtype=dtypes.float32).realize()

print(f"\nRunning with current settings:")
print(f"  WINO={WINO.value}, WINO_OLD={WINO_OLD.value}")

# Compile
t0 = time.time()
out = x.conv2d(w, padding=1)
out.realize()
compile_time = time.time() - t0

# Run
t1 = time.time()
result = out.numpy()
run_time = time.time() - t1

print(f"\nResults:")
print(f"  Compile time: {compile_time*1000:.1f} ms")
print(f"  Run time:     {run_time*1000:.1f} ms")
print(f"  Output shape: {result.shape}")

# Show which version is active
if WINO.value:
    print(f"\n✓ Using NEW winograd (unified kron) - schedule/rangeify.py")
elif WINO_OLD.value:
    print(f"\n✓ Using OLD winograd (tensor ops) - tensor.py")
else:
    print(f"\n✓ Using BASELINE (standard conv) - no winograd")

print(f"\n{'='*80}")
print("If VIZ=1 is set, the visualization should open in your browser")
print(f"{'='*80}\n")
