import unittest
from tinygrad.device import CompileError, Device, Compiler
from tinygrad import Tensor
if Device.DEFAULT=="METAL":
  from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalProgram
@unittest.skipIf(Device.DEFAULT!="METAL", "Metal support required")
class TestMetal(unittest.TestCase):
  def test_alloc_oom(self):
    device = MetalDevice("metal")
    with self.assertRaises(MemoryError):
      device.allocator.alloc(10000000000000000000)

  def test_compile_error(self):
    compiler = MetalCompiler()
    with self.assertRaises(CompileError):
      compiler.compile("this is not valid metal")

  def test_compile_success(self):
    compiler = MetalCompiler()
    ret = compiler.compile("""
#include <metal_stdlib>
  using namespace metal;
  kernel void E_4n1(device int* data0, const device int* data1, const device int* data2,
          uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    int val0 = *(data1+0);
    int val1 = *(data1+1);
    int val2 = *(data1+2);
    int val3 = *(data1+3);
    int val4 = *(data2+0);
    int val5 = *(data2+1);
    int val6 = *(data2+2);
    int val7 = *(data2+3);
    *(data0+0) = (val0+val4);
    *(data0+1) = (val1+val5);
    *(data0+2) = (val2+val6);
    *(data0+3) = (val3+val7);
  }
""")
    assert ret is not None

  def test_failed_newLibraryWithData(self):
    device = MetalDevice("metal")
    compiler = MetalCompiler()
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void r_5(device int* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]){
  data0[0] = 0;
}
""")
    with self.assertRaises(RuntimeError):
      compiled = compiled[:40] # corrupt the compiled program
      MetalProgram(device, "r_5", compiled)

  def test_program_w_empty_compiler(self):
    device = MetalDevice("metal")
    compiler = Compiler(device)
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void r_5(device int* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]){
  data0[0] = 0;
}
""")
    MetalProgram(device, "r_5", compiled)

  def test_bad_program_w_empty_compiler(self):
    device = MetalDevice("metal")
    compiler = Compiler(device)
    # this does not raise
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void r_5(device int* data0, const device int* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]){
  invalid codes;
}
""")
    with self.assertRaises(RuntimeError):
      MetalProgram(device, "r_5", compiled)

  def test_multi_device_synchronization(self):
    NUM_DEVICES = 4
    devices = tuple(f"METAL:{i}" for i in range(NUM_DEVICES))

    tensors = []
    for device in devices:
      t = Tensor.randn(100, 100, device=device)
      tensors.append(t)

    results = []
    for i, t in enumerate(tensors):
      result = t.matmul(t)
      result.realize()
      results.append(result)

    for i in range(1, NUM_DEVICES):
      transfer = results[0].to(device=devices[i])
      transfer.realize()

    for i in range(1, NUM_DEVICES):
      transfer = results[i].to(device=devices[0])
      transfer.realize()

    for i in range(NUM_DEVICES):
      src = i
      dst = (i + 1) % NUM_DEVICES
      transfer = results[src].to(device=devices[dst])
      transfer.realize()