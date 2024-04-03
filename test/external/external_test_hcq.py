import unittest, ctypes, struct
from tinygrad import Device, Tensor
from tinygrad.engine.schedule import create_schedule
from tinygrad.runtime.ops_kfd import KFDDevice, HWCopyQueue, HWComputeQueue

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

  def setUp(self):
    TestHCQ.a.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
    TestHCQ.b.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 0))))

  def test_run_to_3(self):
    ctypes.memmove(TestHCQ.d0.kernargs_ptr, TestHCQ.addr, len(TestHCQ.addr))
    ctypes.memmove(TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.addr2, len(TestHCQ.addr2))
    q = HWComputeQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr+len(TestHCQ.addr), TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size, TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 3.0, f"got val {val}"

  def test_wait_signal(self):
    ctypes.memmove(TestHCQ.d0.kernargs_ptr, TestHCQ.addr, len(TestHCQ.addr))
    TestHCQ.d0.completion_signal.value = 1
    q = HWComputeQueue()
    q.wait(TestHCQ.d0.completion_signal)
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id, timeout=50)

  def test_run_normal(self):
    ctypes.memmove(TestHCQ.d0.kernargs_ptr, TestHCQ.addr, len(TestHCQ.addr))
    q = HWComputeQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size, TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_signal_timeout(self):
    q = HWComputeQueue()
    q.submit(TestHCQ.d0)
    with self.assertRaises(RuntimeError):
      TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id, timeout=50)

  def test_signal(self):
    q = HWComputeQueue()
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id)

  def test_run_signal(self):
    ctypes.memmove(TestHCQ.d0.kernargs_ptr, TestHCQ.addr, len(TestHCQ.addr))
    q = HWComputeQueue()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.global_size, TestHCQ.runner.local_size)
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_copy_signal(self):
    q = HWCopyQueue()
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id)

  def test_copy_copies(self):
    q = HWCopyQueue()
    q.copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8)
    q.signal(TestHCQ.d0.completion_signal)
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_on(TestHCQ.d0.completion_signal.event_id)
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 1.0, f"got val {val}"

if __name__ == "__main__":
  unittest.main()

