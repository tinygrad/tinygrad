#!/usr/bin/env python3
"""Test what grid OLD produces for different input sizes"""
import subprocess
import os

sizes = [24, 32, 40, 48, 64]

for size in sizes:
    env = os.environ.copy()
    env['DEBUG'] = '4'
    env['RANGEIFY'] = '1'
    env['WINO_OLD'] = '1'

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
            print(f"Input {size}×{size}: grid = {grid}")
            break
