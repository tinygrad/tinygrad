#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import unittest
from tinygrad import Device
from tinygrad.helpers import Context, SPILL

KERNEL = '''
define amdgpu_kernel void @test(ptr addrspace(1) %out, i32 %n) {
entry:
  ret void
}
'''

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDSpillFlags(unittest.TestCase):
  def test_spill(self):
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
    print(f"\nSPILL value: {SPILL.value}")

if __name__ == "__main__":
  unittest.main()
