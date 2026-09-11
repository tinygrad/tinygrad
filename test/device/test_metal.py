import unittest, pickle, subprocess, sys, threading
from tinygrad import Tensor, TinyJit, Variable, dtypes
from tinygrad.engine.realize import compile_linear, link_linear, run_linear
from tinygrad.device import CompileError, Device, Buffer, BufferSpec, ProfileGraphEvent
from test.backend.test_profiler import helper_collect_profile
if Device.DEFAULT=="METAL":
  from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler
  from tinygrad.runtime.autogen import metal
@unittest.skipIf(Device.DEFAULT!="METAL", "Metal support required")
class TestMetal(unittest.TestCase):
  def test_residency_set_unavailable(self):
    code = """
from unittest.mock import patch
from tinygrad.runtime.ops_metal import MetalDevice, metal
with patch.object(metal.MTLDevice, 'newResidencySetWithDescriptor_error', return_value=metal.MTLResidencySet()):
  MetalDevice('METAL')
"""
    ret = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
    self.assertEqual(ret.returncode, 1, ret.stderr.decode()) # Python exception, not SIGABRT from addResidencySet(nil)
    self.assertIn("RuntimeError: METAL HCQ2 requires residency sets", ret.stderr.decode())

  def test_profile_kernel_timestamps(self):
    x = Tensor.ones(256).contiguous().realize()
    with helper_collect_profile(Device["METAL"]) as profile:
      out = ((x + 1).contiguous() * 2).contiguous().realize()
    times = [(event.sigs[e.st_id], event.sigs[e.en_id]) for event in profile if isinstance(event, ProfileGraphEvent)
             for e in event.ents if e.device == "METAL"]
    self.assertEqual(len(times), 2)
    for start, end in times:
      self.assertGreater(start, 0)
      self.assertGreater(end, start)
    self.assertGreaterEqual(times[1][0], times[0][1])
    self.assertEqual(out.tolist(), [4] * 256)

  def test_pickle_jit_fresh_process(self):
    @TinyJit
    def f(x): return x + 1
    for _ in range(3): f(Tensor([1, 2, 3, 4]))
    code = "import pickle, sys; from tinygrad import Tensor; f = pickle.load(sys.stdin.buffer); " \
           "assert f(Tensor([4, 3, 2, 1])).tolist() == [5, 4, 3, 2]"
    ret = subprocess.run([sys.executable, "-c", code], input=pickle.dumps(f), capture_output=True, timeout=60)
    self.assertEqual(ret.returncode, 0, ret.stderr.decode())

  def test_batch_dependencies(self):
    x = Tensor.full((4,), 2).contiguous().realize()
    out = x
    for _ in range(3): out = (out + 1).contiguous()
    compiled = compile_linear(out.schedule_linear())
    linked = link_linear(compiled)
    run_linear(linked, jit=True, wait=True)
    self.assertEqual(out.tolist(), [5] * 4)

  def test_async_replay_snapshots_arguments(self):
    @TinyJit
    def f(src, dst, v): return dst.assign(src + v).realize()
    srcs = [Tensor.full((256,), i, dtype=dtypes.int).contiguous().realize() for i in range(8)]
    dsts = [Tensor.zeros(256, dtype=dtypes.int).contiguous().realize() for _ in srcs]
    for _ in range(3): f(srcs[0], dsts[0], Variable("v", 1, 8).bind(1))
    dev = Device["METAL"]
    dev.synchronize()
    event = dev.sysdevice.newSharedEvent()
    gate = dev.queue.commandBuffer()
    gate.encodeWaitForEvent_value(metal.MTLEvent(event.value), 1)
    gate.commit()
    timer = threading.Timer(5, lambda: event.setSignaledValue(1)) # release a broken synchronous implementation instead of hanging the suite
    timer.start()
    try:
      for i in range(8): f(srcs[i], dsts[i], Variable("v", 1, 8).bind(i + 1))
      self.assertEqual(event.signaledValue(), 0, "submission waited for GPU completion")
    finally:
      event.setSignaledValue(1)
      timer.cancel()
      dev.synchronize()
    for i, dst in enumerate(dsts): self.assertEqual(dst.tolist(), [2 * i + 1] * 256)
    for i in range(64): f(srcs[i % 8], dsts[i % 8], Variable("v", 1, 8).bind(i // 8 + 1))
    for i, dst in enumerate(dsts): self.assertEqual(dst.tolist(), [i + 8] * 256) # wrap the pool; host reads wait for queued writes

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
