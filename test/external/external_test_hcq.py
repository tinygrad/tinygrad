import unittest, ctypes, struct, time
from tinygrad import Device, Tensor, dtypes
from tinygrad.buffer import Buffer, BufferOptions
from tinygrad.engine.schedule import create_schedule
from tinygrad.runtime.ops_amd import AMDDevice, HWCopyQueue, HWPM4Queue

def _time_queue(q, d):
  st = time.perf_counter()
  q.signal(d.timeline_signal, d.timeline_value)
  q.submit(d)
  d._wait_signal(d.timeline_signal, d.timeline_value)
  d.timeline_value += 1
  return time.perf_counter() - st

class TestHCQ(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestHCQ.d0: AMDDevice = Device["AMD"]
    #TestHCQ.d1: AMDDevice = Device["AMD:1"]
    TestHCQ.a = Tensor([0.,1.], device="AMD").realize()
    TestHCQ.b = self.a + 1
    si = create_schedule([self.b.lazydata])[-1]
    TestHCQ.runner = TestHCQ.d0.get_runner(*si.ast)
    TestHCQ.b.lazydata.buffer.allocate()
    # wow that's a lot of abstraction layers
    TestHCQ.addr = struct.pack("QQ", TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr)
    TestHCQ.addr2 = struct.pack("QQ", TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr)
    ctypes.memmove(TestHCQ.d0.kernargs_ptr, TestHCQ.addr, len(TestHCQ.addr))
    ctypes.memmove(TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.addr2, len(TestHCQ.addr2))
    TestHCQ.compute_queue = HWPM4Queue

  def setUp(self):
    TestHCQ.a.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
    TestHCQ.b.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 0))))
    TestHCQ.d0.synchronize() # wait for copyins to complete

  def test_run_1000_times_one_submit(self):
    temp_signal, temp_value = TestHCQ.d0._get_signal(value=0), 0
    q = TestHCQ.compute_queue()
    for _ in range(1000):
      q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
      q.signal(temp_signal, temp_value + 1).wait(temp_signal, temp_value + 1)
      temp_value += 1

      q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.runner.global_size, TestHCQ.runner.local_size)
      q.signal(temp_signal, temp_value + 1).wait(temp_signal, temp_value + 1)
      temp_value += 1

    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 2000.0, f"got val {val}"

  def test_run_1000_times(self):
    temp_signal = TestHCQ.d0._get_signal(value=0)
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(temp_signal, 2).wait(temp_signal, 2)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.runner.global_size,
           TestHCQ.runner.local_size)
    for _ in range(1000):
      temp_signal.value = 1
      q.submit(TestHCQ.d0)
      TestHCQ.compute_queue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 2000.0, f"got val {val}"

  def test_run_to_3(self):
    temp_signal = TestHCQ.d0._get_signal(value=0)
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(temp_signal, 1).wait(temp_signal, 1)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(temp_signal, 2).wait(temp_signal, 2)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 3.0, f"got val {val}"

  def test_wait_signal(self):
    temp_signal = TestHCQ.d0._get_signal(value=0)
    TestHCQ.compute_queue().wait(temp_signal, value=1).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=50)
    # clean up
    temp_signal.value = 1
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=100)
    TestHCQ.d0.timeline_value += 1

  def test_wait_copy_signal(self):
    temp_signal = TestHCQ.d0._get_signal(value=0)
    HWCopyQueue().wait(temp_signal, value=1).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=50)
    # clean up
    temp_signal.value = 1
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=100)
    TestHCQ.d0.timeline_value += 1

  def test_run_normal(self):
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_submit_empty_queues(self):
    TestHCQ.compute_queue().submit(TestHCQ.d0)
    HWCopyQueue().submit(TestHCQ.d0)

  def test_signal_timeout(self):
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=50)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value + 122, timeout=50)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1, timeout=50)

  def test_signal(self):
    new_timeline_value = TestHCQ.d0.timeline_value + 0xff
    TestHCQ.compute_queue().signal(TestHCQ.d0.timeline_signal, new_timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, new_timeline_value)
    TestHCQ.d0.timeline_value = new_timeline_value + 1 # update to not break runtime

  def test_copy_signal(self):
    new_timeline_value = TestHCQ.d0.timeline_value + 0xff
    HWCopyQueue().signal(TestHCQ.d0.timeline_signal, new_timeline_value).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, new_timeline_value)
    TestHCQ.d0.timeline_value = new_timeline_value + 1 # update to not break runtime

  def test_run_signal(self):
    q = TestHCQ.compute_queue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_copy_1000_times(self):
    q = HWCopyQueue()
    q.copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8)
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    for _ in range(1000):
      q.submit(TestHCQ.d0)
      HWCopyQueue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
    # confirm the signal didn't exceed the put value
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value + 1, timeout=50)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 0.0, f"got val {val}"

  def test_copy(self):
    q = HWCopyQueue()
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 1.0, f"got val {val}"

  def test_copy_bandwidth(self):
    # THEORY: the bandwidth is low here because it's only using one SDMA queue. I suspect it's more stable like this at least.
    SZ = 2_000_000_000
    a = Buffer("AMD", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    b = Buffer("AMD", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    q = HWCopyQueue()
    q.copy(a._buf.va_addr, b._buf.va_addr, SZ)
    et = _time_queue(q, TestHCQ.d0)
    gb_s = (SZ/1e9)/et
    print(f"same device copy:  {et*1e3:.2f} ms, {gb_s:.2f} GB/s")
    assert gb_s > 10 and gb_s < 1000

  def test_cross_device_copy_bandwidth(self):
    SZ = 2_000_000_000
    b = Buffer("AMD:1", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    a = Buffer("AMD", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    TestHCQ.d0._gpu_map(b._buf)
    q = HWCopyQueue()
    q.copy(a._buf.va_addr, b._buf.va_addr, SZ)
    et = _time_queue(q, TestHCQ.d0)
    gb_s = (SZ/1e9)/et
    print(f"cross device copy: {et*1e3:.2f} ms, {gb_s:.2f} GB/s")
    assert gb_s > 2 and gb_s < 50

  def test_interleave_compute_and_copy(self):
    q = TestHCQ.compute_queue()
    qc = HWCopyQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)  # b = [1, 2]
    q.signal(sig:=AMDDevice._get_signal(value=0), value=1)
    qc.wait(sig, value=1)
    qc.copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8)
    qc.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    qc.submit(TestHCQ.d0)
    time.sleep(0.02) # give it time for the wait to fail
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_cross_device_signal(self):
    d1 = Device["AMD:1"]
    q1 = TestHCQ.compute_queue()
    q2 = TestHCQ.compute_queue()
    q1.signal(sig:=AMDDevice._get_signal(value=0), value=0xfff)
    q2.wait(sig, value=0xfff)
    q2.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    q2.submit(TestHCQ.d0)
    q1.signal(d1.timeline_signal, d1.timeline_value)
    q1.submit(d1)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1
    d1._wait_signal(d1.timeline_signal, d1.timeline_value)
    d1.timeline_value += 1

  def test_timeline_signal_rollover(self):
    TestHCQ.d0.timeline_value = (1 << 32) - 20 # close value to reset
    TestHCQ.compute_queue().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1)

    for _ in range(40):
      q = TestHCQ.compute_queue()
      q.wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1)
      q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
      q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
      assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

if __name__ == "__main__":
  unittest.main()
