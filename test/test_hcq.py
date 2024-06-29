import unittest, ctypes, struct, time, array
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import to_mv, CI, getenv
from tinygrad.device import Buffer, BufferOptions, HCQCompatCompiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import get_runner

MOCKGPU = getenv("MOCKGPU")

@unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompatCompiled), "HCQCompat device required to run")
class TestHCQ(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestHCQ.d0 = Device[Device.DEFAULT]
    TestHCQ.a = Tensor([0.,1.], device=Device.DEFAULT).realize()
    TestHCQ.b = self.a + 1
    si = create_schedule([self.b.lazydata])[-1]

    TestHCQ.runner = get_runner(TestHCQ.d0.dname, si.ast)
    TestHCQ.b.lazydata.buffer.allocate()
    TestHCQ.addr = struct.pack("QQ", TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr)
    TestHCQ.addr2 = struct.pack("QQ", TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr)
    TestHCQ.kernargs_off = TestHCQ.runner.clprg.kernargs_offset
    TestHCQ.kernargs_size = TestHCQ.runner.clprg.kernargs_alloc_size
    ctypes.memmove(TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_off, TestHCQ.addr, len(TestHCQ.addr))
    ctypes.memmove(TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size+TestHCQ.kernargs_off, TestHCQ.addr2, len(TestHCQ.addr2))

    if Device.DEFAULT == "NV":
      # nv need to copy constbuffer there as well
      to_mv(TestHCQ.d0.kernargs_ptr, 0x160).cast('I')[:] = array.array('I', TestHCQ.runner.clprg.constbuffer_0)
      to_mv(TestHCQ.d0.kernargs_ptr+TestHCQ.kernargs_size, 0x160).cast('I')[:] = array.array('I', TestHCQ.runner.clprg.constbuffer_0)

  def setUp(self):
    TestHCQ.d0.synchronize()
    TestHCQ.a.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
    TestHCQ.b.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 0))))
    TestHCQ.d0.synchronize() # wait for copyins to complete

  # Test signals
  def test_signal(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        queue_type().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  def test_signal_update(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        q = queue_type.signal(fake_signal := TestHCQ.d0._get_signal(), 0x1000)

        q.update_signal(signal=TestHCQ.d0.timeline_signal, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        q.update_signal(value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

  # Test wait
  def test_wait(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0._get_signal()    
        TestHCQ.d0._set_signal(fake_signal, 1)
        queue_type.wait(fake_signal, 1) \
                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

  @unittest.skipIf(MOCKGPU, "Can't handle async update on MOCKGPU for now")
  def test_wait_late_set(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0._get_signal()
        queue_type.wait(fake_signal, 1) \
                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

        with self.assertRaises(RuntimeError):
          TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=500)
        
        TestHCQ.d0._set_signal(fake_signal, 1)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

if __name__ == "__main__":
  unittest.main()
