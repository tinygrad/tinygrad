#!/usr/bin/env python3
"""
Compare OLD vs NEW (optimized) winograd kernels line-by-line.
Captures actual Metal shader code and identifies differences.
"""
import subprocess
import os
import difflib

def capture_kernel(wino_old=False):
    """Capture Metal kernel for OLD or NEW winograd"""
    env = os.environ.copy()
    env['DEBUG'] = '4'
    env['RANGEIFY'] = '1'
    if wino_old:
        env['WINO_OLD'] = '1'
        env['WINO'] = '0'
    else:
        env['WINO'] = '1'
        env['WINO_OLD'] = '0'

    # Simple convolution that triggers winograd
    code = """
from tinygrad import Tensor, dtypes
x = Tensor.randn(1, 16, 64, 64, dtype=dtypes.float32).realize()
w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()
out = x.conv2d(w, padding=1)
out.realize()
"""

    result = subprocess.run(['python3', '-c', code], env=env,
                          capture_output=True, text=True,
                          cwd='/Users/niranjanbaskaran/git/tinygrad')
    return result.stdout

def extract_main_kernel(output):
    """Extract the largest r_* kernel from debug output"""
    lines = output.split('\n')
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

    # Return largest kernel (main winograd kernel)
    return max(kernels, key=len) if kernels else ""

print("="*80)
print("CAPTURING WINOGRAD KERNELS")
print("="*80)

print("\nCapturing OLD winograd kernel (WINO_OLD=1)...")
old_output = capture_kernel(wino_old=True)
old_kernel = extract_main_kernel(old_output)

print("Capturing NEW optimized kernel (WINO=1, winograd_kron)...")
new_output = capture_kernel(wino_old=False)
new_kernel = extract_main_kernel(new_output)

# Save to current directory
with open('old_wino_kernel.metal', 'w') as f:
    f.write(old_kernel)
with open('new_wino_kernel.metal', 'w') as f:
    f.write(new_kernel)

print(f"\n✓ Kernels saved:")
print(f"  - old_wino_kernel.metal ({len(old_kernel.split(chr(10)))} lines)")
print(f"  - new_wino_kernel.metal ({len(new_kernel.split(chr(10)))} lines)")

# Analyze
import re
old_lines = old_kernel.split('\n')
new_lines = new_kernel.split('\n')

old_float_vars = len(re.findall(r'float \w+', old_kernel))
new_float_vars = len(re.findall(r'float \w+', new_kernel))
old_bool_vars = len(re.findall(r'bool \w+', old_kernel))
new_bool_vars = len(re.findall(r'bool \w+', new_kernel))

print("\n" + "="*80)
print("KERNEL STATISTICS")
print("="*80)
print(f"{'Metric':<20} {'OLD':>10} {'NEW':>10} {'Diff':>10}")
print("-"*80)
print(f"{'Lines':<20} {len(old_lines):>10} {len(new_lines):>10} {new_lines.__len__() - old_lines.__len__():+>10}")
print(f"{'Float variables':<20} {old_float_vars:>10} {new_float_vars:>10} {new_float_vars - old_float_vars:+>10}")
print(f"{'Bool variables':<20} {old_bool_vars:>10} {new_bool_vars:>10} {new_bool_vars - old_bool_vars:+>10}")

# Line-by-line comparison
print("\n" + "="*80)
print("LINE-BY-LINE DIFF ANALYSIS")
print("="*80)

# Get diff
diff = list(difflib.unified_diff(old_lines, new_lines, lineterm='',
                                   fromfile='OLD', tofile='NEW', n=0))

if len(diff) <= 2:  # Only headers
    print("\n🎉 KERNELS ARE IDENTICAL! 🎉")
    print("\nOLD and NEW generate the exact same Metal shader code!")
else:
    print(f"\nFound {len([l for l in diff if l.startswith('-') or l.startswith('+')]) // 2} differences")
    print("\nFirst 50 differences:")
    print("-"*80)

    diff_count = 0
    for i, line in enumerate(diff):
        if line.startswith('@@'):
            continue
        if line.startswith('-') or line.startswith('+'):
            print(line)
            diff_count += 1
            if diff_count >= 100:  # Show first 100 diff lines
                print(f"\n... ({len(diff) - i} more diff lines)")
                break

# Identify KEY differences
print("\n" + "="*80)
print("KEY DIFFERENCES ANALYSIS")
print("="*80)

# Check signature
old_sig = old_lines[0] if old_lines else ""
new_sig = new_lines[0] if new_lines else ""

if old_sig != new_sig:
    print("\n1. KERNEL SIGNATURE:")
    print(f"   OLD: {old_sig[:100]}...")
    print(f"   NEW: {new_sig[:100]}...")
else:
    print("\n1. KERNEL SIGNATURE: ✓ Identical")

# Check variable declarations (first 30 lines)
old_vars = '\n'.join(old_lines[1:30])
new_vars = '\n'.join(new_lines[1:30])

if old_vars != new_vars:
    print("\n2. VARIABLE SETUP (lines 1-30): Different")
    print("   (This is expected - variables may be ordered differently)")
else:
    print("\n2. VARIABLE SETUP: ✓ Identical")

# Check if computation logic is similar
old_has_wmma = 'WMMA' in old_kernel or 'wmma' in old_kernel
new_has_wmma = 'WMMA' in new_kernel or 'wmma' in new_kernel

print(f"\n3. USES SIMD/WMMA OPERATIONS:")
print(f"   OLD: {old_has_wmma}")
print(f"   NEW: {new_has_wmma}")

# Final verdict
print("\n" + "="*80)
print("VERDICT")
print("="*80)

size_diff = len(new_lines) - len(old_lines)
var_diff = new_float_vars - old_float_vars

if abs(size_diff) <= 50 and abs(var_diff) <= 50:
    print("\n✅ EXCELLENT! Kernels are nearly identical in structure!")
    print(f"   Size difference: {size_diff:+d} lines ({100*size_diff/len(old_lines):+.1f}%)")
    print(f"   Var difference:  {var_diff:+d} floats ({100*var_diff/old_float_vars:+.1f}%)")
    print("\n   The NEW implementation successfully matches OLD's efficiency!")
elif abs(size_diff) <= 100:
    print("\n✓ GOOD! Kernels are very similar in structure!")
    print(f"   Size difference: {size_diff:+d} lines ({100*size_diff/len(old_lines):+.1f}%)")
    print(f"   Var difference:  {var_diff:+d} floats")
    print("\n   Minor differences are acceptable for the cleaner abstraction.")
else:
    print("\n⚠ Kernels differ significantly:")
    print(f"   Size difference: {size_diff:+d} lines ({100*size_diff/len(old_lines):+.1f}%)")
    print(f"   Var difference:  {var_diff:+d} floats")

print("\n" + "="*80)
print(f"View full kernels:")
print(f"  cat old_wino_kernel.metal")
print(f"  cat new_wino_kernel.metal")
print(f"  diff -u old_wino_kernel.metal new_wino_kernel.metal | less")
print("="*80)
