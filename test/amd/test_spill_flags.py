#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import unittest
from tinygrad import Device
from tinygrad.helpers import Context

KERNEL = '''
kernel void test(global float* out, int n) {
  int idx = get_global_id(0);
  float vals[16];
  for (int i = 0; i < 16; i++) vals[i] = idx + i;
  for (int i = 0; i < n; i++) out[idx] = vals[i%16] + vals[(i+1)%16];
}
'''

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDSpillFlags(unittest.TestCase):
  def test_spill(self):
    from tinygrad.runtime.support.compiler_amd import HIPCompiler
    import time
    for spill in [0, 1]:
      with Context(SPILL=spill):
        t = time.perf_counter()
        HIPCompiler("gfx1100").compile(KERNEL)
        print(f"\nSPILL={spill}: {time.perf_counter() - t:.3f}s")

if __name__ == "__main__":
  unittest.main()
