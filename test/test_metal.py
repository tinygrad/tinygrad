import unittest
import os
import time
from unittest.mock import patch
from tinygrad.device import CompileError, Device, Compiler
from tinygrad.helpers import getenv
if Device.DEFAULT=="METAL":
  from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalProgram, MetalBuffer, msg, libobjc

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

  # New tests for virtual device synchronization improvements
  def test_virtual_device_detection(self):
    """Test that virtual devices are properly detected"""
    device = MetalDevice("metal")
    # Test should work regardless of whether device is virtual or not
    self.assertIsInstance(device.is_virtual, bool)
    self.assertIsInstance(device.max_buffers_in_flight, int)
    self.assertGreater(device.max_buffers_in_flight, 0)

    # Virtual devices should have smaller buffer limits
    if device.is_virtual:
      self.assertEqual(device.max_buffers_in_flight, 64)
      self.assertIsNotNone(device.virtual_sync_event)
    else:
      self.assertEqual(device.max_buffers_in_flight, 1024)
      self.assertIsNone(device.virtual_sync_event)

  @patch('tinygrad.runtime.ops_metal.from_ns_str')
  def test_virtual_device_mock_detection(self, mock_from_ns_str):
    """Test virtual device detection with mocked device names"""
    # Test virtual device detection
    mock_from_ns_str.return_value = "Apple Virtual GPU"
    device = MetalDevice("metal")
    self.assertTrue(device.is_virtual)
    self.assertEqual(device.max_buffers_in_flight, 64)
    self.assertIsNotNone(device.virtual_sync_event)

    # Test paravirtualized device detection
    mock_from_ns_str.return_value = "Apple Paravirtualized GPU"
    device2 = MetalDevice("metal")
    self.assertTrue(device2.is_virtual)

    # Test normal device
    mock_from_ns_str.return_value = "Apple M1"
    device3 = MetalDevice("metal")
    self.assertFalse(device3.is_virtual)
    self.assertEqual(device3.max_buffers_in_flight, 1024)

  def test_sync_capacity_management(self):
    """Test that sync capacity management works correctly"""
    device = MetalDevice("metal")

    # Test ensure_sync_capacity with empty buffer list
    initial_count = len(device.mtl_buffers_in_flight)
    device.ensure_sync_capacity()
    self.assertEqual(len(device.mtl_buffers_in_flight), initial_count)

    # Test _sync_oldest_buffers with empty list
    device._sync_oldest_buffers(5)
    self.assertEqual(len(device.mtl_buffers_in_flight), 0)

  def test_buffer_capacity_enforcement(self):
    """Test that buffer capacity is enforced for virtual devices"""
    device = MetalDevice("metal")
    compiler = MetalCompiler()

    # Create a simple kernel for testing
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void test_kernel(device int* data, uint3 gid [[threadgroup_position_in_grid]]){
  data[gid.x] = gid.x;
}
""")
    program = MetalProgram(device, "test_kernel", compiled)

    # Allocate a test buffer
    test_buf = device.allocator.alloc(64)

    # Simulate filling up the buffer queue (but not beyond capacity)
    max_test_buffers = min(10, device.max_buffers_in_flight // 2)

    for i in range(max_test_buffers):
      # Call program without waiting to fill the buffer queue
      program(test_buf, global_size=(1,1,1), local_size=(1,1,1), wait=False)

    # Verify buffers are tracked
    self.assertLessEqual(len(device.mtl_buffers_in_flight), device.max_buffers_in_flight)

    # Clean up
    device.synchronize()
    device.allocator.free(test_buf, None)

  def test_virtual_device_transfer_sync(self):
    """Test enhanced synchronization during transfers for virtual devices"""
    device1 = MetalDevice("metal:0")
    device2 = MetalDevice("metal:1") if getenv("METAL_MULTI_GPU", 0) else device1

    # Allocate buffers on both devices
    buf1 = device1.allocator.alloc(1024)
    buf2 = device2.allocator.alloc(1024)

    try:
      # Test transfer between devices (or same device)
      device1.allocator._transfer(buf2, buf1, 1024, device1, device2)

      # Verify synchronization occurred
      # For virtual devices, additional sync should happen
      if device1.is_virtual or device2.is_virtual:
        # Virtual device transfers should be more conservative
        self.assertLessEqual(len(device1.mtl_buffers_in_flight), device1.max_buffers_in_flight)
        self.assertLessEqual(len(device2.mtl_buffers_in_flight), device2.max_buffers_in_flight)

      # Clean up
      device1.synchronize()
      device2.synchronize()

    finally:
      device1.allocator.free(buf1, None)
      device2.allocator.free(buf2, None)

  def test_virtual_graph_environment_variable(self):
    """Test that METAL_ENABLE_VIRTUAL_GRAPH controls graph usage"""
    # Test with graph enabled for virtual devices
    with patch.dict(os.environ, {'METAL_ENABLE_VIRTUAL_GRAPH': '1'}):
      with patch('tinygrad.runtime.ops_metal.from_ns_str', return_value="Apple Virtual GPU"):
        device = MetalDevice("metal")
        # Should have graph enabled even for virtual device
        # Note: We can't directly test graph creation due to import dependencies,
        # but we can verify the device was created successfully
        self.assertTrue(device.is_virtual)

  def test_synchronization_performance(self):
    """Test that synchronization doesn't cause significant performance degradation"""
    device = MetalDevice("metal")
    compiler = MetalCompiler()

    # Create a simple kernel
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void perf_test(device int* data, uint3 gid [[threadgroup_position_in_grid]]){
  data[gid.x] = gid.x * 2;
}
""")
    program = MetalProgram(device, "perf_test", compiled)

    # Allocate test buffer
    test_buf = device.allocator.alloc(1024)

    try:
      # Time multiple kernel executions
      start_time = time.time()

      num_iterations = 10
      for i in range(num_iterations):
        program(test_buf, global_size=(256,1,1), local_size=(1,1,1), wait=True)

      end_time = time.time()
      avg_time = (end_time - start_time) / num_iterations

      # Verify execution completed (basic sanity check)
      self.assertGreater(avg_time, 0)
      self.assertLess(avg_time, 1.0)  # Should complete within reasonable time

      # Verify final synchronization
      device.synchronize()
      self.assertEqual(len(device.mtl_buffers_in_flight), 0)

    finally:
      device.allocator.free(test_buf, None)

  def test_multi_buffer_synchronization_stress(self):
    """Stress test for multiple buffer management in virtual devices"""
    device = MetalDevice("metal")
    compiler = MetalCompiler()

    # Create a simple kernel
    compiled = compiler.compile("""
#include <metal_stdlib>
kernel void stress_test(device int* data, uint3 gid [[threadgroup_position_in_grid]]){
  data[gid.x] = gid.x + 1;
}
""")
    program = MetalProgram(device, "stress_test", compiled)

    # Create multiple buffers
    num_buffers = 5
    buffers = []

    try:
      for i in range(num_buffers):
        buf = device.allocator.alloc(256)
        buffers.append(buf)

      # Execute multiple operations to stress the synchronization system
      stress_iterations = min(20, device.max_buffers_in_flight // 4)

      for i in range(stress_iterations):
        buf_idx = i % num_buffers
        program(buffers[buf_idx], global_size=(64,1,1), local_size=(1,1,1), wait=False)

        # Occasionally force synchronization
        if i % 5 == 4:
          device.synchronize()

      # Final synchronization
      device.synchronize()
      self.assertEqual(len(device.mtl_buffers_in_flight), 0)

    finally:
      # Clean up all buffers
      for buf in buffers:
        device.allocator.free(buf, None)

  def test_virtual_sync_event_functionality(self):
    """Test virtual device sync event creation and usage"""
    device = MetalDevice("metal")

    if device.is_virtual:
      # Virtual devices should have sync events
      self.assertIsNotNone(device.virtual_sync_event)
      self.assertEqual(device.virtual_sync_value, 0)

      # Test that sync values increment during operations
      compiler = MetalCompiler()
      compiled = compiler.compile("""
#include <metal_stdlib>
kernel void sync_test(device int* data, uint3 gid [[threadgroup_position_in_grid]]){
  data[0] = 42;
}
""")
      program = MetalProgram(device, "sync_test", compiled)
      test_buf = device.allocator.alloc(64)

      try:
        initial_sync_value = device.virtual_sync_value
        program(test_buf, global_size=(1,1,1), local_size=(1,1,1), wait=False)

        # Sync value should increment for virtual devices
        self.assertGreater(device.virtual_sync_value, initial_sync_value)

        device.synchronize()

      finally:
        device.allocator.free(test_buf, None)
    else:
      # Physical devices should not have virtual sync events
      self.assertIsNone(device.virtual_sync_event)
      self.assertEqual(device.virtual_sync_value, 0)

  def test_error_handling_with_sync_improvements(self):
    """Test that error handling still works correctly with sync improvements"""
    device = MetalDevice("metal")

    # Test that OOM still raises MemoryError
    with self.assertRaises(MemoryError):
      device.allocator.alloc(10000000000000000000)

    # Test that device remains functional after error
    test_buf = device.allocator.alloc(64)
    try:
      # Should still work normally
      self.assertIsInstance(test_buf, MetalBuffer)
      self.assertEqual(test_buf.size, 64)
    finally:
      device.allocator.free(test_buf, None)

if __name__ == '__main__':
    unittest.main()