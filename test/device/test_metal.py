import unittest
from tinygrad import Tensor, dtypes
from tinygrad.engine.realize import compile_linear, link_linear, run_linear
from tinygrad.uop.ops import Ops
from tinygrad.device import CompileError, Device, Buffer, BufferSpec
if Device.DEFAULT=="METAL":
  from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler
@unittest.skipIf(Device.DEFAULT!="METAL", "Metal support required")
class TestMetal(unittest.TestCase):
  def test_icb_per_batch(self):
    x = Tensor.full((4,), 2).contiguous().realize()
    out = x
    for _ in range(3): out = (out + 1).contiguous()
    compiled = compile_linear(out.schedule_linear())
    icbs = [u for u in compiled.toposort() if u.op is Ops.PARAM and isinstance(u.tag, tuple) and u.tag[0] == "icb"]
    self.assertEqual(len(icbs), 1) # one batch, one icb
    self.assertEqual(len(icbs[0].tag[1]), 3) # a command per call: repeated programs still need separate commands
    self.assertEqual(icbs[0].arg.size, 4) # the icb and its commands
    linked = link_linear(compiled)
    self.assertFalse(any(u.op is Ops.PARAM and u.tag == icbs[0].tag for u in linked.toposort()))
    run_linear(linked, jit=True, wait=True)
    self.assertEqual(out.tolist(), [5] * 4)

  def test_host_copy_views(self):
    src = Buffer("CPU", 64, dtypes.uint8, initial_value=bytes(range(64)))
    dst = Buffer("METAL", 64, dtypes.uint8, initial_value=bytes(64))
    dst.view(16, dtypes.uint8, 8).ensure_allocated().copy_from(src.view(16, dtypes.uint8, 24).ensure_allocated())
    out = Buffer("CPU", 64, dtypes.uint8, preallocate=True).copy_from(dst)
    self.assertEqual(bytes(out.as_memoryview()), bytes(8) + bytes(range(24, 40)) + bytes(40))

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
      device.pipeline(compiled, "r_5")

  def test_free(self):
    size = 2**16
    device = Device['METAL']
    before = device.sysdevice.currentAllocatedSize()

    buf = device.allocator.alloc(size, BufferSpec(nolru=True))
    self.assertEqual(curr:=device.sysdevice.currentAllocatedSize(), before+size, msg=f"{curr=} - {before=}")
    device.allocator.free(buf, size, BufferSpec(nolru=True))
    self.assertEqual(curr:=device.sysdevice.currentAllocatedSize(), before, msg=f"{curr=} - {before=}")
