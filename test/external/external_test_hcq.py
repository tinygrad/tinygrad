import unittest, ctypes, struct, time
from tinygrad import Device, Tensor, dtypes
from tinygrad.buffer import Buffer, BufferOptions
from tinygrad.engine.schedule import create_schedule
from tinygrad.runtime.ops_kfd import KFDDevice, HWCopyQueue, HWComputeQueue

def _time_queue(q, d):
  st = time.perf_counter()
  q.signal(d.completion_signal)
  q.submit(d)
  d._wait_signal(d.completion_signal)
  return time.perf_counter() - st

class TestHCQ(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestHCQ.d0: KFDDevice = Device["KFD"]
    TestHCQ.d1: KFDDevice = Device["KFD:1"]
    TestHCQ.a = Tensor([0.,1.], device="KFD").realize()
    TestHCQ.b = self.a + 1
    si = create_schedule([self.b.lazydata])[-1]
    TestHCQ.runner = TestHCQ.d0.get_runner(*si.ast)
    TestHCQ.b.lazydata.buffer.allocate()
    # wow that's a lot of abstraction layers
    TestHCQ.addr = struct.pack("QQ", TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr)
    TestHCQ.addr2 = struct.pack("QQ", TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr)
    ctypes.memmove(TestHCQ.d0.kernargs_ptr, TestHCQ.addr, len(TestHCQ.addr))
    ctypes.memmove(TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.addr2, len(TestHCQ.addr2))

  def setUp(self):
    TestHCQ.a.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
    TestHCQ.b.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 0))))

  def test_run_1000_times_one_submit(self):
    q = HWComputeQueue()
    for _ in range(1000):
      q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
      q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 2000.0, f"got val {val}"

  def test_run_1000_times(self):
    q = HWComputeQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.runner.global_size,
           TestHCQ.runner.local_size, TestHCQ.d0.completion_signal)
    for _ in range(1000):
      q.submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    # confirm signal was reset
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal, timeout=50)
    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 2000.0, f"got val {val}"

  def test_run_to_3(self):
    q = HWComputeQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size, TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 3.0, f"got val {val}"

  def test_wait_signal(self):
    TestHCQ.d0.completion_signal.value = 1
    HWComputeQueue().wait(TestHCQ.d0.completion_signal).signal(TestHCQ.d0.completion_signal).submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal, timeout=50)
    # clean up
    TestHCQ.d0.completion_signal.value = 0
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal, timeout=1000)

  def test_wait_copy_signal(self):
    TestHCQ.d0.completion_signal.value = 1
    HWCopyQueue().wait(TestHCQ.d0.completion_signal).signal(TestHCQ.d0.completion_signal).submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal, timeout=50)
    # clean up
    TestHCQ.d0.completion_signal.value = 0
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal, timeout=1000)

  def test_run_normal(self):
    q = HWComputeQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size, TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_submit_empty_queues(self):
    HWComputeQueue().submit(TestHCQ.d0)
    HWCopyQueue().submit(TestHCQ.d0)

  def test_signal_timeout(self):
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal, timeout=50)

  def test_signal(self):
    HWComputeQueue().signal(TestHCQ.d0.completion_signal).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)

  def test_copy_signal(self):
    HWCopyQueue().signal(TestHCQ.d0.completion_signal).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)

  def test_run_signal(self):
    q = HWComputeQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_copy_1000_times(self):
    q = HWCopyQueue()
    q.copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8)
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    q.signal(TestHCQ.d0.completion_signal)
    for _ in range(1000):
      q.submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    # confirm signal was reset
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal, timeout=50)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 0.0, f"got val {val}"

  def test_copy(self):
    q = HWCopyQueue()
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 1.0, f"got val {val}"

  def test_copy_bandwidth(self):
    # THEORY: the bandwidth is low here because it's only using one SDMA queue. I suspect it's more stable like this at least.
    SZ = 2_000_000_000
    a = Buffer("KFD", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    b = Buffer("KFD", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    q = HWCopyQueue()
    q.copy(a._buf.va_addr, b._buf.va_addr, SZ)
    et = _time_queue(q, TestHCQ.d0)
    gb_s = (SZ/1e9)/et
    print(f"same device copy:  {et*1e3:.2f} ms, {gb_s:.2f} GB/s")
    assert gb_s > 10 and gb_s < 1000

  def test_cross_device_copy_bandwidth(self):
    SZ = 2_000_000_000
    a = Buffer("KFD", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    b = Buffer("KFD:1", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    TestHCQ.d0._gpu_map(b._buf)
    q = HWCopyQueue()
    q.copy(a._buf.va_addr, b._buf.va_addr, SZ)
    et = _time_queue(q, TestHCQ.d0)
    gb_s = (SZ/1e9)/et
    print(f"cross device copy: {et*1e3:.2f} ms, {gb_s:.2f} GB/s")
    assert gb_s > 2 and gb_s < 50

  def test_interleave_compute_and_copy(self):
    q = HWComputeQueue()
    qc = HWCopyQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)  # b = [1, 2]
    q.signal(sig:=KFDDevice._get_signal(10))
    qc.wait(sig)
    qc.copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8)
    qc.signal(TestHCQ.d0.completion_signal)
    sig.value = 1
    qc.submit(TestHCQ.d0)
    time.sleep(0.02) # give it time for the wait to fail
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)
    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_cross_device_signal(self):
    q1 = HWComputeQueue()
    q2 = HWComputeQueue()
    q1.signal(TestHCQ.d0.completion_signal)
    q2.wait(TestHCQ.d0.completion_signal)
    q2.submit(TestHCQ.d0)
    q1.submit(TestHCQ.d1)
    TestHCQ.d0._wait_signal(TestHCQ.d0.completion_signal)

if __name__ == "__main__":
  unittest.main()

