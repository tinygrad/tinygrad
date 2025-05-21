import unittest
from tinygrad.device import CompileError, Device, Compiler
from tinygrad.tensor import Tensor
import random
import numpy as np
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

@unittest.skipIf(Device.DEFAULT != "METAL", "Metal support required")
class TestTensorDeviceTransfers(unittest.TestCase):
  @staticmethod
  def _generate_devices(count):
    return [f"METAL:{i}" for i in range(count)]

  @staticmethod
  def _check_tensor_match(src, dst, tol=1e-6):
    np.testing.assert_allclose(
      src.numpy(), dst.numpy(), rtol=1e-5, atol=tol,
      err_msg="Tensor values diverged on transfer"
    )

  def _prepare_random_tensors(self, device_list, size=(100, 100), seed=42):
    random.seed(seed)
    created = []
    for dev in device_list:
      t = Tensor.randn(*size, device=dev)
      t = t.matmul(t).realize()
      created.append(t)
    return created

  def test_fill_pairwise_transfers(self):
    devs = self._generate_devices(4)
    filled = {
      dev: Tensor.full((10, 10), float(i + 1), device=dev)
      for i, dev in enumerate(devs)
    }

    for src_dev in devs:
      original = filled[src_dev]
      expected_value = float(devs.index(src_dev) + 1)
      for dst_dev in devs:
        if dst_dev == src_dev:
          continue
        moved = original.to(device=dst_dev).realize()
        self.assertEqual(
          moved.shape, original.shape,
          f"Shape mismatch transferring from {src_dev} to {dst_dev}"
        )
        arr = moved.numpy()
        np.testing.assert_array_equal(
          arr, np.full_like(arr, expected_value),
          err_msg=f"Values not preserved from {src_dev} to {dst_dev}"
        )
