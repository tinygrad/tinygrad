import unittest, ctypes, struct
from tinygrad import Device, Tensor
from tinygrad.engine.schedule import create_schedule
from tinygrad.runtime.ops_kfd import KFDDevice, HWCopyQueue, HWComputeQueue

class TestKFD(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestKFD.d0: KFDDevice = Device["KFD"]
    TestKFD.d1: KFDDevice = Device["KFD:1"]
    TestKFD.a = Tensor([0.,0], device="KFD").realize()
    TestKFD.b = self.a + 1
    si = create_schedule([self.b.lazydata])[-1]
    TestKFD.runner = TestKFD.d0.get_runner(*si.ast)
    TestKFD.b.lazydata.buffer.allocate()
    # wow that's a lot of abstraction layers
    TestKFD.addr = struct.pack("QQ", TestKFD.b.lazydata.buffer._buf.va_addr, TestKFD.a.lazydata.buffer._buf.va_addr)
    TestKFD.addr2 = struct.pack("QQ", TestKFD.a.lazydata.buffer._buf.va_addr, TestKFD.b.lazydata.buffer._buf.va_addr)

  def setUp(self): TestKFD.b.lazydata.buffer.copyin(memoryview(bytearray(b"\x00"*8)))

  def test_run_to_3(self):
    ctypes.memmove(TestKFD.d0.kernargs_ptr, TestKFD.addr, len(TestKFD.addr))
    ctypes.memmove(TestKFD.d0.kernargs_ptr+len(TestKFD.addr), TestKFD.addr2, len(TestKFD.addr2))
    q = HWComputeQueue()
    q.exec(TestKFD.runner.clprg, TestKFD.d0.kernargs_ptr, TestKFD.runner.global_size, TestKFD.runner.local_size)
    q.exec(TestKFD.runner.clprg, TestKFD.d0.kernargs_ptr+len(TestKFD.addr), TestKFD.runner.global_size, TestKFD.runner.local_size)
    q.exec(TestKFD.runner.clprg, TestKFD.d0.kernargs_ptr, TestKFD.runner.global_size, TestKFD.runner.local_size, ctypes.addressof(TestKFD.d0.completion_signal))
    q.submit(TestKFD.d0)
    TestKFD.d0._wait_on(TestKFD.d0.completion_signal.event_id)
    assert (val:=TestKFD.b.lazydata.buffer.as_buffer().cast("f")[0]) == 3.0, f"got val {val}"

  def test_run_normal(self):
    ctypes.memmove(TestKFD.d0.kernargs_ptr, TestKFD.addr, len(TestKFD.addr))
    q = HWComputeQueue()
    q.exec(TestKFD.runner.clprg, TestKFD.d0.kernargs_ptr, TestKFD.runner.global_size, TestKFD.runner.local_size, ctypes.addressof(TestKFD.d0.completion_signal))
    q.submit(TestKFD.d0)
    TestKFD.d0._wait_on(TestKFD.d0.completion_signal.event_id)
    assert (val:=TestKFD.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_run_signal(self):
    ctypes.memmove(TestKFD.d0.kernargs_ptr, TestKFD.addr, len(TestKFD.addr))
    q = HWComputeQueue()
    q.exec(TestKFD.runner.clprg, TestKFD.d0.kernargs_ptr, TestKFD.runner.global_size, TestKFD.runner.local_size)
    q.signal(ctypes.addressof(TestKFD.d0.completion_signal))
    q.submit(TestKFD.d0)
    TestKFD.d0._wait_on(TestKFD.d0.completion_signal.event_id)
    assert (val:=TestKFD.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

if __name__ == "__main__":
  unittest.main()

