#!/usr/bin/env python3
"""Compare OLD vs NEW (optimized) winograd performance"""
import os
import time

# Test OLD winograd
print("="*80)
print("OLD WINOGRAD (Tensor-level _apply_winograd_matrix)")
print("="*80)
os.environ['DEBUG'] = '3'
os.environ['RANGEIFY'] = '1'
os.environ['WINO_OLD'] = '1'
os.environ['WINO'] = '0'

from tinygrad import Tensor, dtypes

x = Tensor.randn(1, 16, 64, 64, dtype=dtypes.float32).realize()
w = Tensor.randn(16, 16, 3, 3, dtype=dtypes.float32).realize()
out = x.conv2d(w, padding=1)
out.realize()

print("\n" + "="*80)
print("NEW WINOGRAD OPTIMIZED (winograd_kron with factored matrix constants)")
print("="*80)
# Restart interpreter by re-importing after env change
import sys
import importlib
for mod in list(sys.modules.keys()):
    if 'tinygrad' in mod:
        del sys.modules[mod]

os.environ['WINO'] = '1'
os.environ['WINO_OLD'] = '0'

from tinygrad import Tensor as Tensor2, dtypes as dtypes2

x2 = Tensor2.randn(1, 16, 64, 64, dtype=dtypes2.float32).realize()
w2 = Tensor2.randn(16, 16, 3, 3, dtype=dtypes2.float32).realize()
out2 = x2.conv2d(w2, padding=1)
out2.realize()

print("\n" + "="*80)
print("Look for the 'r_' kernel timings (tm ...us) in the output above!")
print("="*80)
