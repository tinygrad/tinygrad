#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import time, unittest
from tinygrad import Device
from tinygrad.helpers import Context, Timing

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

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDSpillFlags(unittest.TestCase):
  def test_spill(self):
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler

    with Timing("SPILL=0 "):
      with Context(SPILL=0): AMDLLVMCompiler("gfx1100").compile(KERNEL)

    with Timing("SPILL=1 "):
      with Context(SPILL=1): AMDLLVMCompiler("gfx1100").compile(KERNEL)

if __name__ == "__main__":
  unittest.main()
