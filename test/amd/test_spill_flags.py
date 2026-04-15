#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import unittest
from tinygrad import Device
from tinygrad.helpers import Context

KERNEL = '''
define amdgpu_kernel void @test(ptr addrspace(1) %out, i32 %n) {
entry:
  %0 = alloca [32 x float], align 4, addrspace(5)
  %1 = alloca [32 x float], align 4, addrspace(5)
  %2 = alloca [32 x float], align 4, addrspace(5)
  %3 = alloca [32 x float], align 4, addrspace(5)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %val = bitcast i32 %i to float
  store float %val, ptr addrspace(5) %0
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
'''

# With DEBUG=7 you can see the LLVM IR and check for spill instructions: writelane, readlane, buffer_store

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDSpillFlags(unittest.TestCase):
  def test_spill(self):
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
    # SPILL=0 should prevent most spills
    with Context(SPILL=0, DEBUG=7):
      AMDLLVMCompiler("gfx1100").compile(KERNEL)
    # SPILL=1 allows spills
    with Context(SPILL=1, DEBUG=7):
      AMDLLVMCompiler("gfx1100").compile(KERNEL)

if __name__ == "__main__":
  unittest.main()
