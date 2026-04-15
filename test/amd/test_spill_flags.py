#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import unittest
from tinygrad import Device
from tinygrad.helpers import Context

# Kernel that forces many register spills
KERNEL_MANY_REGS = '''
define amdgpu_kernel void @test(ptr addrspace(1) %out, i32 %n) {
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
  %v0 = fadd float %val, %val
  %v1 = fadd float %v0, %v0
  %v2 = fadd float %v1, %v0
  %v3 = fadd float %v2, %v0
  %v4 = fadd float %v3, %v0
  %v5 = fadd float %v4, %v0
  %v6 = fadd float %v5, %v0
  %v7 = fadd float %v6, %v0
  store float %v7, ptr addrspace(5) %0
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
    from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler, CompileError
    import time
    for spill in [0, 1]:
      try:
        with Context(SPILL=spill, DEBUG=7):
          t = time.perf_counter()
          AMDLLVMCompiler("gfx1100").compile(KERNEL_MANY_REGS)
          print(f"\nSPILL={spill}: {time.perf_counter() - t:.3f}s (compiled)")
      except (CompileError, RuntimeError) as e:
        print(f"\nSPILL={spill}: failed - {e}")

if __name__ == "__main__":
  unittest.main()
