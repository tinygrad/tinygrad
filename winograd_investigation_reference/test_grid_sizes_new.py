#!/usr/bin/env python3
"""Test what grid NEW produces for different input sizes"""
import subprocess
import os

sizes = [24, 32, 40, 48, 64]

print("OLD vs NEW Grid Comparison")
print("="*70)

for size in sizes:
    grids = {}

    for name, wino_old in [("OLD", True), ("NEW", False)]:
        env = os.environ.copy()
        env['DEBUG'] = '4'
        env['RANGEIFY'] = '1'
        env['WINO_OLD'] = '1' if wino_old else '0'
        env['WINO'] = '0' if wino_old else '1'

        code = f"""
from tinygrad import Tensor, dtypes
x = Tensor.randn(1, 16, {size}, {size}, dtype=dtypes.float32).realize()
w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()
out = x.conv2d(w, padding=1).realize()
"""

        result = subprocess.run(['python3', '-c', code], env=env,
                              capture_output=True, text=True)

        # Extract grid from kernel signature
        for line in result.stdout.split('\n'):
            if 'kernel void r_' in line:
                grid = line.split('kernel void ')[1].split('(')[0]
                grids[name] = grid
                break

    # Extract first 3 dims (global grid)
    def get_global_grid(grid_str):
        parts = grid_str.split('_')[1:]  # Skip 'r'
        if len(parts) >= 3:
            return f"{parts[0]}×{parts[1]}×{parts[2]}"
        return grid_str

    print(f"\nInput {size}×{size}:")
    print(f"  OLD: {grids.get('OLD', 'N/A'):30s} → {get_global_grid(grids.get('OLD', ''))}")
    print(f"  NEW: {grids.get('NEW', 'N/A'):30s} → {get_global_grid(grids.get('NEW', ''))}")

    if grids.get('OLD') == grids.get('NEW'):
        print(f"  ✅ MATCH!")
    else:
        print(f"  ❌ DIFFERENT")
