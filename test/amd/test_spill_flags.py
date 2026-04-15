#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import time, unittest
from tinygrad import Device
from tinygrad.helpers import Context, SPILL

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

def _time_compile(spill, n=3):
  from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
  times = []
  for _ in range(n):
    with Context(SPILL=spill):
      t = time.perf_counter()
      AMDLLVMCompiler("gfx1100").compile(KERNEL)
      times.append(time.perf_counter() - t)
  return sum(times) / n

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDSpillFlags(unittest.TestCase):
  def test_spill_0(self):
    """Test SPILL=0 compiles."""
    with Context(SPILL=0):
      AMDLLVMCompiler("gfx1100").compile(KERNEL)

  def test_spill_1(self):
    """Test SPILL=1 compiles."""
    with Context(SPILL=1):
      AMDLLVMCompiler("gfx1100").compile(KERNEL)

  def test_timing(self):
    """Compare timing."""
    t0, t1 = _time_compile(0), _time_compile(1)
    print(f"\nSPILL=0: {t0:.3f}s, SPILL=1: {t1:.3f}s")
    self.assertTrue(t0 > 0 and t1 > 0)

if __name__ == "__main__":
  unittest.main()