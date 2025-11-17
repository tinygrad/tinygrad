#!/usr/bin/env python3
"""Capture the current HYBRID kernel before attempting fixes"""
import subprocess
import os

env = os.environ.copy()
env['DEBUG'] = '4'
env['RANGEIFY'] = '1'
env['WINO'] = '1'

code = """
from tinygrad import Tensor, dtypes
x = Tensor.randn(1, 16, 64, 64, dtype=dtypes.float32).realize()
w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()
out = x.conv2d(w, padding=1)
out.realize()
"""

print("Capturing HYBRID kernel...")
result = subprocess.run(['python3', '-c', code], env=env,
                       capture_output=True, text=True,
                       cwd='/Users/niranjanbaskaran/git/tinygrad')

# Extract the winograd kernel (largest one)
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

# Get the largest kernel (winograd)
if kernels:
    kernel = max(kernels, key=len)

    with open('hybrid_wino_kernel.metal', 'w') as f:
        f.write(kernel)

    print(f"\n✅ Saved HYBRID kernel to hybrid_wino_kernel.metal")
    print(f"   Lines: {len(kernel.split(chr(10)))}")

    # Extract signature
    sig = kernel.split('\n')[0]
    print(f"\nSignature: {sig[:100]}...")

    # Check grid
    if 'r_16_6_6' in sig:
        print("\nGrid: r_16_6_6 (HYBRID - channels first)")
    elif 'r_6_6_16' in sig:
        print("\nGrid: r_6_6_16 (OLD-style - spatial first)")

    # Check buffer order
    if 'data1_2304' in sig and 'data2_65536' in sig:
        print("Buffers: data1_2304 (GHAT), data2_65536 (XHAT) - OLD-style order")
    elif 'data1_65536' in sig and 'data2_2304' in sig:
        print("Buffers: data1_65536 (XHAT), data2_2304 (GHAT) - NEW-style order")

    # Check write pattern
    write_lines = [l for l in kernel.split('\n') if '*(data0' in l and '=' in l]
    print(f"\nWrite pattern sample (first 5 writes):")
    for w in write_lines[:5]:
        print(f"  {w.strip()}")

    if 'float2' in '\n'.join(write_lines[:10]):
        print("\n✅ Uses coalesced float2 writes")
    else:
        print("\n⚠️  Uses scattered single float writes")
else:
    print("❌ No kernel found!")
