import unittest, ctypes, struct, array
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
      if MOCKGPU: TestHCQ.runner.clprg.constbuffer_0[0:2] = [2, 0] # hack for nv mockgpu only. it needs to get count of args and vals.
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
    for queue_type in [TestHCQ.d0.hw_compute_queue_t]:
      with self.subTest(name=str(queue_type)):
        q = queue_type().signal(fake_signal := TestHCQ.d0._get_signal(), 0x1000)

        q.update_signal(0, signal=TestHCQ.d0.timeline_signal, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        q.update_signal(0, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

  # Test wait
  def test_wait(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0._get_signal()
        TestHCQ.d0._set_signal(fake_signal, 1)
        queue_type().wait(fake_signal, 1) \
                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

  @unittest.skipIf(MOCKGPU, "Can't handle async update on MOCKGPU for now")
  def test_wait_late_set(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0._get_signal()
        queue_type().wait(fake_signal, 1) \
                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

        with self.assertRaises(RuntimeError):
          TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value, timeout=500)

        TestHCQ.d0._set_signal(fake_signal, 1)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

  def test_wait_update(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0._get_signal()
        q = queue_type().wait(TestHCQ.d0.timeline_signal, 0xffffffff).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

        TestHCQ.d0._set_signal(fake_signal, 0x30)

        q.update_wait(0, signal=fake_signal, value=0x30).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

  # Test exec
  def test_exec_one_kernel(self):
    TestHCQ.d0.hw_compute_queue_t().exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
                                   .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_exec_2_kernels_100_times(self):
    q = TestHCQ.d0.hw_compute_queue_t()
    q.wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
     .exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr + TestHCQ.kernargs_size, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    for _ in range(100):
      q.update_wait(0, value=TestHCQ.d0.timeline_value - 1).update_signal(3, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 200.0, f"got val {val}"

  def test_exec_update(self):
    q = TestHCQ.d0.hw_compute_queue_t()
    q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.update_exec(0, (1,1,1), (1,1,1))
    q.submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 0.0, f"got val {val}, should not be updated"

  # Test copy
  def test_copy(self):
    TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                .copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 0.0, f"got val {val}"

  def test_copy_100_times(self):
    q = TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                    .copy(TestHCQ.a.lazydata.buffer._buf.va_addr, TestHCQ.b.lazydata.buffer._buf.va_addr, 8) \
                                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    for _ in range(100):
      q.update_wait(0, value=TestHCQ.d0.timeline_value - 1).update_signal(2, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      q.submit(TestHCQ.d0)
      TestHCQ.d0.timeline_value += 1

  # Test bind api
  def test_bind(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      with self.subTest(name=str(queue_type)):
        if not hasattr(queue_type(), 'bind'): self.skipTest("queue does not support bind api")

        fake_signal = TestHCQ.d0._get_signal()
        q = queue_type().wait(TestHCQ.d0.timeline_signal, 0xffffffff).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        q.bind(TestHCQ.d0)

        TestHCQ.d0._set_signal(fake_signal, 0x30)

        q.update_wait(0, signal=fake_signal, value=0x30).submit(TestHCQ.d0)
        TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        TestHCQ.d0.signals_pool.append(fake_signal)

  # Test multidevice
  def test_multidevice_signal_wait(self):
    d1 = Device[f"{Device.DEFAULT}:1"]

    TestHCQ.d0.hw_copy_queue_t().signal(sig:=TestHCQ.d0._get_signal(value=0), value=0xfff) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    d1.hw_copy_queue_t().wait(sig, value=0xfff) \
                        .signal(d1.timeline_signal, d1.timeline_value).submit(d1)

    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    d1._wait_signal(d1.timeline_signal, d1.timeline_value)
    d1.timeline_value += 1

    TestHCQ.d0.signals_pool.append(sig)

  # Test profile api
  def test_speed_exec_time(self):
    TestHCQ.d0._prof_setup()

    sig_st, sig_en = TestHCQ.d0._get_signal(), TestHCQ.d0._get_signal()
    TestHCQ.d0.hw_compute_queue_t().timestamp(sig_st) \
                                   .exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
                                   .timestamp(sig_en) \
                                   .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = TestHCQ.d0._gpu2cpu_time(TestHCQ.d0._read_timestamp(sig_en), True) - TestHCQ.d0._gpu2cpu_time(TestHCQ.d0._read_timestamp(sig_st), True)

    TestHCQ.d0.signals_pool += [sig_st, sig_en]

    print(f"exec kernel time: {et:.2f} us")
    assert 1 <= et <= (2000 if CI else 20)

  def test_speed_copy_bandwidth(self):
    TestHCQ.d0._prof_setup()

    # THEORY: the bandwidth is low here because it's only using one SDMA queue. I suspect it's more stable like this at least.
    SZ = 2_000_000_000
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    b = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()

    sig_st, sig_en = TestHCQ.d0._get_signal(), TestHCQ.d0._get_signal()
    TestHCQ.d0.hw_copy_queue_t().timestamp(sig_st) \
                                .copy(a._buf.va_addr, b._buf.va_addr, SZ) \
                                .timestamp(sig_en) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = TestHCQ.d0._gpu2cpu_time(TestHCQ.d0._read_timestamp(sig_en), True) - TestHCQ.d0._gpu2cpu_time(TestHCQ.d0._read_timestamp(sig_st), True)
    et_ms = et / 1e3

    TestHCQ.d0.signals_pool += [sig_st, sig_en]

    gb_s = ((SZ / 1e9) / et_ms) * 1e3
    print(f"same device copy:  {et_ms:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.3 if CI else 10) <= gb_s <= 1000

  def test_speed_cross_device_copy_bandwidth(self):
    TestHCQ.d0._prof_setup()

    SZ = 2_000_000_000
    b = Buffer(f"{Device.DEFAULT}:1", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    TestHCQ.d0._gpu_map(b._buf)

    sig_st, sig_en = TestHCQ.d0._get_signal(), TestHCQ.d0._get_signal()
    TestHCQ.d0.hw_copy_queue_t().timestamp(sig_st) \
                                .copy(a._buf.va_addr, b._buf.va_addr, SZ) \
                                .timestamp(sig_en) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = TestHCQ.d0._gpu2cpu_time(TestHCQ.d0._read_timestamp(sig_en), True) - TestHCQ.d0._gpu2cpu_time(TestHCQ.d0._read_timestamp(sig_st), True)
    et_ms = et / 1e3

    TestHCQ.d0.signals_pool += [sig_st, sig_en]

    gb_s = ((SZ / 1e9) / et_ms) * 1e3
    print(f"cross device copy: {et_ms:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.3 if CI else 2) <= gb_s <= 50

  def test_timeline_signal_rollover(self):
    # NV 64bit, AMD 32bit
    TestHCQ.d0.timeline_value = (1 << 64) - 20 if Device.DEFAULT == "NV" else (1 << 32) - 20 # close value to reset
    TestHCQ.d0.hw_compute_queue_t().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1).submit(TestHCQ.d0)
    TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1)

    for _ in range(40):
      q = TestHCQ.d0.hw_compute_queue_t()
      q.wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1)
      q.exec(TestHCQ.runner.clprg, TestHCQ.d0.kernargs_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size)
      q.signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0._wait_signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1
      assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

if __name__ == "__main__":
  unittest.main()
