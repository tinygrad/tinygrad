import unittest
import numpy as np
from tinygrad.device import Device, Buffer, CompileError, Compiler
from tinygrad.dtype import dtypes
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

  def test_cross_device_transfer_synchronization(self):
    metal_device_names = [d for d in Device.devices if d.startswith("METAL")]
    if len(metal_device_names) < 2:
      self.skipTest("At least two Metal devices are required for the cross-device transfer test.")

    dev1 = Device[metal_device_names[0]]
    dev2 = Device[metal_device_names[1]]

    dev1.timeline_value = 0
    dev2.timeline_value = 0


    test_data = np.arange(1024, dtype=np.float32)
    src_buf = Buffer(device=dev1.id, size=1024, dtype=dtypes.float32).copyin(memoryview(test_data))

    # 2. Create an empty destination buffer on dev2.
    dest_buf = Buffer(device=dev2.id, size=1024, dtype=dtypes.float32)

    # 3. Perform the cross-device transfer. This will invoke the allocator's `_transfer`
    #    method, which should contain the signal/wait synchronization logic.
    dest_buf.copy(src_buf)

    # 4. Synchronize the destination device to ensure the transfer operation is complete.
    dev2.synchronize()

    # 5. Verify that the data in the destination buffer matches the source data.
    result_data = dest_buf.to('CPU').cast('f').numpy()
    np.testing.assert_allclose(result_data, test_data, atol=1e-5, rtol=1e-5, err_msg="Data mismatch after cross-device transfer")

    # 6. Verify that the timeline value was incremented correctly on the source device.
    #    The _transfer method should call get_next_timeline_value() twice.
    self.assertEqual(dev1.timeline_value, 2, f"Timeline value on source device ({dev1.dname}) was not incremented correctly.")
    self.assertEqual(dev2.timeline_value, 0, f"Timeline value on destination device ({dev2.dname}) should not be incremented.")
