#!/usr/bin/env python3
"""Test AMDGPU LLVM spill prevention flags."""
import time, unittest
from tinygrad import Device
from tinygrad.helpers import Context, SPILL

# Simple kernel that exercises register allocation
KERNEL = '''
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

def _compile(spill):
  from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
  with Context(SPILL=spill):
    compiler = AMDLLVMCompiler("gfx1100")
    lib = compiler.compile(KERNEL)
    return compiler.disassemble(lib) if hasattr(compiler, 'disassemble') else lib

def _time_compile(spill, n_iters=5):
  from tinygrad.runtime.support.compiler_amd import AMDLLVMCompiler
  times = []
  for _ in range(n_iters):
    with Context(SPILL=spill):
      start = time.perf_counter()
      compiler = AMDLLVMCompiler("gfx1100")
      compiler.compile(KERNEL)
      times.append(time.perf_counter() - start)
  return sum(times) / n_iters

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDSpillFlags(unittest.TestCase):
  def test_spill_0(self):
    """Test with SPILL=0 (no register spilling)."""
    code = _compile(0)
    self.assertTrue(len(code) > 0)

  def test_spill_1(self):
    """Test with SPILL=1 (default, allows spilling)."""
    code = _compile(1)
    self.assertTrue(len(code) > 0)

  def test_timing_compare(self):
    """Compare compile time with and without spill flags."""
    time_0 = _time_compile(0)
    time_1 = _time_compile(1)
    print(f"\nSPILL=0 avg: {time_0:.3f}s, SPILL=1 avg: {time_1:.3f}s")
    print(f"Speedup: {time_1/time_0:.2f}x" if time_0 > 0 else "N/A")
    self.assertTrue(time_0 > 0 and time_1 > 0)

if __name__ == "__main__":
  unittest.main()