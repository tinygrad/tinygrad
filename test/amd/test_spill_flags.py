#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import unittest
from tinygrad import Device

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDSpillFlags(unittest.TestCase):
  def test_spill_prevention_flags(self):
    """Test that spill prevention flags don't cause compilation failures."""
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
    src = '''
define amdgpu_kernel void @many_regs(ptr addrspace(1) %out, i32 %n) {
entry:
  %0 = alloca [32 x float], align 4, addrspace(5)
  %1 = alloca [32 x float], align 4, addrspace(5)
  %2 = alloca [32 x float], align 4, addrspace(5)
  %3 = alloca [32 x float], align 4, addrspace(5)
  %4 = alloca [32 x float], align 4, addrspace(5)
  %5 = alloca [32 x float], align 4, addrspace(5)
  %6 = alloca [32 x float], align 4, addrspace(5)
  %7 = alloca [32 x float], align 4, addrspace(5)
  %8 = alloca [32 x float], align 4, addrspace(5)
  %9 = alloca [32 x float], align 4, addrspace(5)
  %10 = alloca [32 x float], align 4, addrspace(5)
  %11 = alloca [32 x float], align 4, addrspace(5)
  %12 = alloca [32 x float], align 4, addrspace(5)
  %13 = alloca [32 x float], align 4, addrspace(5)
  %14 = alloca [32 x float], align 4, addrspace(5)
  %15 = alloca [32 x float], align 4, addrspace(5)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %val = bitcast i32 %i to float
  %ptr0 = getelementptr inbounds [32 x float], ptr addrspace(5) %0, i32 0, i32 %i
  store float %val, ptr addrspace(5) %ptr0
  %i.next = add i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
    '''
    compiler = AMDLLVMCompiler("gfx1100")
    compiler.compile(src)

  def test_wwm_alloc_zero(self):
    """Test that -amdgpu-num-vgprs-for-wwm-alloc=0 works (fails if WWM needed)."""
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
    src = '''
define amdgpu_kernel void @simple(ptr addrspace(1) %out) {
entry:
  store float 1.0, ptr addrspace(1) %out
  ret void
}
    '''
    compiler = AMDLLVMCompiler("gfx1100")
    compiler.compile(src)

if __name__ == "__main__":
  unittest.main()