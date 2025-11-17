#!/usr/bin/env python3
import subprocess
import os

env = os.environ.copy()
env['DEBUG'] = '4'
env['RANGEIFY'] = '1'
env['WINO'] = '1'
env['WINO_OLD'] = '0'

code = """
from tinygrad import Tensor, dtypes
x = Tensor.randn(1, 16, 64, 64, dtype=dtypes.float32).realize()
w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()
out = x.conv2d(w, padding=1)
out.realize()
"""

print("Capturing NEW kernel with fixed order...")
result = subprocess.run(['python3', '-c', code], env=env,
                       capture_output=True, text=True,
                       cwd='/Users/niranjanbaskaran/git/tinygrad')

# Extract kernel
lines = result.stdout.split('\n')
kernels = []
in_kernel = False
current = []
brace_count = 0

for line in lines:
    if 'kernel void r_' in line:
        in_kernel = True
        current = [line]
        brace_count = line.count('{') - line.count('}')
    elif in_kernel:
        current.append(line)
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0:
            kernels.append('\n'.join(current))
            in_kernel = False
            current = []

kernel = max(kernels, key=len) if kernels else ""

# Check signature
sig_line = kernel.split('\n')[0] if kernel else ""
print(f"\nKernel signature:")
print(f"  {sig_line}")

# Check if data buffers are in OLD order
if 'data1_2304' in sig_line and 'data2_65536' in sig_line:
    print(f"\n✅ SUCCESS! Buffer order matches OLD:")
    print(f"     data1_2304 (GHAT - weights)")
    print(f"     data2_65536 (XHAT - inputs)")
elif 'data1_65536' in sig_line and 'data2_2304' in sig_line:
    print(f"\n❌ Still swapped:")
    print(f"     data1_65536 (XHAT - inputs) - should be GHAT")
    print(f"     data2_2304 (GHAT - weights) - should be XHAT")
else:
    print(f"\n⚠ Different buffer sizes - check manually")

# Check grid dims
if 'r_6_6_16' in sig_line:
    print(f"\n✅ Grid organization matches OLD: r_6_6_16...")
elif 'r_16_6_6' in sig_line:
    print(f"\n❌ Grid still different: r_16_6_6...")
else:
    print(f"\n⚠ Different grid organization")
