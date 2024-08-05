import unittest, ctypes, struct, contextlib, tempfile, pathlib, json, time, atexit, random
from tinygrad import Device, Tensor, dtypes, TinyJit
from tinygrad.helpers import CI, getenv, Context
from tinygrad.device import Buffer, BufferOptions, HCQCompiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import get_runner

MOCKGPU = getenv("MOCKGPU")

@unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "HCQ device required to run")
class TestHCQ(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestHCQ.d0 = Device[Device.DEFAULT]
    TestHCQ.a = Tensor([0.,1.], device=Device.DEFAULT).realize()
    TestHCQ.b = self.a + 1
    si = create_schedule([self.b.lazydata])[-1]

    TestHCQ.runner = get_runner(TestHCQ.d0.dname, si.ast)
    TestHCQ.b.lazydata.buffer.allocate()

    TestHCQ.kernargs_ba_ptr = TestHCQ.runner.clprg.fill_kernargs([TestHCQ.b.lazydata.buffer._buf, TestHCQ.a.lazydata.buffer._buf])
    TestHCQ.kernargs_ab_ptr = TestHCQ.runner.clprg.fill_kernargs([TestHCQ.a.lazydata.buffer._buf, TestHCQ.b.lazydata.buffer._buf])

  def setUp(self):
    TestHCQ.d0.synchronize()
    TestHCQ.a.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
    TestHCQ.b.lazydata.buffer.copyin(memoryview(bytearray(struct.pack("ff", 0, 0))))
    TestHCQ.d0.synchronize() # wait for copyins to complete

  # Test signals
  def test_signal(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        queue_type().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  def test_signal_update(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        q = queue_type().signal(TestHCQ.d0.signal_t(), 0x1000)

        q.update_signal(0, signal=TestHCQ.d0.timeline_signal, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

        q.update_signal(0, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  # Test wait
  def test_wait(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0.signal_t()
        fake_signal.value = 1
        queue_type().wait(fake_signal, 1) \
                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  @unittest.skipIf(MOCKGPU, "Can't handle async update on MOCKGPU for now")
  def test_wait_late_set(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0.signal_t()
        queue_type().wait(fake_signal, 1) \
                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

        with self.assertRaises(RuntimeError):
          TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value, timeout=500)

        fake_signal.value = 1
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)

        TestHCQ.d0.timeline_value += 1

  def test_wait_update(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0.signal_t()
        q = queue_type().wait(TestHCQ.d0.timeline_signal, 0xffffffff).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

        fake_signal.value = 0x30

        q.update_wait(0, signal=fake_signal, value=0x30).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  # Test exec
  def test_exec_one_kernel(self):
    TestHCQ.d0.hw_compute_queue_t().exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
                                   .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"

  def test_exec_2_kernels_100_times(self):
    q = TestHCQ.d0.hw_compute_queue_t()
    q.wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
     .exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ab_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    for _ in range(100):
      q.update_wait(0, value=TestHCQ.d0.timeline_value - 1).update_signal(3, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]) == 200.0, f"got val {val}"

  def test_exec_update(self):
    q = TestHCQ.d0.hw_compute_queue_t()
    q.exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.update_exec(0, (1,1,1), (1,1,1))
    q.submit(TestHCQ.d0)
    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]) == 1.0, f"got val {val}"
    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 0.0, f"got val {val}, should not be updated"

  # Test copy
  def test_copy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                .copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 1.0, f"got val {val}"

  def test_copy_long(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    sz = 64 << 20
    buf1 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferOptions(host=True, nolru=True)).ensure_allocated()
    ctypes.memset(buf2._buf.va_addr, 1, sz)

    TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                .copy(buf1._buf.va_addr, buf2._buf.va_addr, sz) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    mv_buf1 = buf1.as_buffer().cast('Q')
    for i in range(sz//8): assert mv_buf1[i] == 0x0101010101010101, f"offset {i*8} differs, not all copied, got {hex(mv_buf1[i])}"

  def test_update_copy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    q = TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                    .copy(0x0, 0x0, 8) \
                                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.update_copy(1, dest=TestHCQ.b.lazydata.buffer._buf.va_addr, src=TestHCQ.a.lazydata.buffer._buf.va_addr) \
     .submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    assert (val:=TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]) == 1.0, f"got val {val}"

  def test_update_copy_long(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    sz = 64 << 20
    buf1 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, sz, dtypes.int8, options=BufferOptions(host=True, nolru=True)).ensure_allocated()
    ctypes.memset(buf2._buf.va_addr, 1, sz)

    q = TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                    .copy(0x0, 0x0, sz) \
                                    .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.update_copy(1, buf1._buf.va_addr, buf2._buf.va_addr) \
     .submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    mv_buf1 = buf1.as_buffer().cast('Q')
    for i in range(sz//8): assert mv_buf1[i] == 0x0101010101010101, f"offset {i*8} differs, not all copied, got {hex(mv_buf1[i])}"

  # Test bind api
  def test_bind(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        fake_signal = TestHCQ.d0.signal_t()
        q = queue_type().wait(TestHCQ.d0.timeline_signal, 0xffffffff).signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)
        q.bind(TestHCQ.d0)

        fake_signal.value = 0x30

        q.update_wait(0, signal=fake_signal, value=0x30).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
        TestHCQ.d0.timeline_value += 1

  # Test multidevice
  def test_multidevice_signal_wait(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    d1 = Device[f"{Device.DEFAULT}:1"]

    TestHCQ.d0.hw_copy_queue_t().signal(sig:=TestHCQ.d0.signal_t(value=0), value=0xfff) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    d1.hw_copy_queue_t().wait(sig, value=0xfff) \
                        .signal(d1.timeline_signal, d1.timeline_value).submit(d1)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    d1.timeline_signal.wait(d1.timeline_value)
    d1.timeline_value += 1

  # Test profile api
  def test_speed_exec_time(self):
    TestHCQ.d0._prof_setup()

    sig_st, sig_en = TestHCQ.d0.signal_t(), TestHCQ.d0.signal_t()
    TestHCQ.d0.hw_compute_queue_t().timestamp(sig_st) \
                                   .exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
                                   .timestamp(sig_en) \
                                   .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = TestHCQ.d0._gpu2cpu_time(sig_en.timestamp, True) - TestHCQ.d0._gpu2cpu_time(sig_st.timestamp, True)

    print(f"exec kernel time: {et:.2f} us")
    assert 1 <= et <= (2500 if CI else 20)

  def test_speed_copy_bandwidth(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    TestHCQ.d0._prof_setup()

    # THEORY: the bandwidth is low here because it's only using one SDMA queue. I suspect it's more stable like this at least.
    SZ = 2_000_000_000
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    b = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()

    sig_st, sig_en = TestHCQ.d0.signal_t(), TestHCQ.d0.signal_t()
    TestHCQ.d0.hw_copy_queue_t().timestamp(sig_st) \
                                .copy(a._buf.va_addr, b._buf.va_addr, SZ) \
                                .timestamp(sig_en) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = TestHCQ.d0._gpu2cpu_time(sig_en.timestamp, True) - TestHCQ.d0._gpu2cpu_time(sig_st.timestamp, True)
    et_ms = et / 1e3

    gb_s = ((SZ / 1e9) / et_ms) * 1e3
    print(f"same device copy:  {et_ms:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.3 if CI else 10) <= gb_s <= 1000

  def test_speed_cross_device_copy_bandwidth(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    TestHCQ.d0._prof_setup()

    SZ = 2_000_000_000
    b = Buffer(f"{Device.DEFAULT}:1", SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    a = Buffer(Device.DEFAULT, SZ, dtypes.uint8, options=BufferOptions(nolru=True)).allocate()
    TestHCQ.d0._gpu_map(b._buf)

    sig_st, sig_en = TestHCQ.d0.signal_t(), TestHCQ.d0.signal_t()
    TestHCQ.d0.hw_copy_queue_t().timestamp(sig_st) \
                                .copy(a._buf.va_addr, b._buf.va_addr, SZ) \
                                .timestamp(sig_en) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    et = TestHCQ.d0._gpu2cpu_time(sig_en.timestamp, True) - TestHCQ.d0._gpu2cpu_time(sig_st.timestamp, True)
    et_ms = et / 1e3

    gb_s = ((SZ / 1e9) / et_ms) * 1e3
    print(f"cross device copy: {et_ms:.2f} ms, {gb_s:.2f} GB/s")
    assert (0.3 if CI else 2) <= gb_s <= 50

  def test_timeline_signal_rollover(self):
    for queue_type in [TestHCQ.d0.hw_compute_queue_t, TestHCQ.d0.hw_copy_queue_t]:
      if queue_type is None: continue

      with self.subTest(name=str(queue_type)):
        TestHCQ.d0.timeline_value = (1 << 32) - 20 # close value to reset
        queue_type().signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1).submit(TestHCQ.d0)
        TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value - 1)

        for _ in range(40):
          queue_type().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                      .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
          TestHCQ.d0.timeline_value += 1
          TestHCQ.d0.synchronize()

  def test_small_copies_from_host_buf(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(host=True, nolru=True)).ensure_allocated()

    for i in range(256):
      ctypes.memset(buf2._buf.va_addr, i, 1)

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf2._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf1.as_buffer()[0] == i

  def test_small_copies_from_host_buf_intercopy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf3 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(host=True, nolru=True)).ensure_allocated()

    for i in range(256):
      ctypes.memset(buf3._buf.va_addr, i, 1)

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf3._buf.va_addr, 1) \
                                  .copy(buf2._buf.va_addr, buf1._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf2.as_buffer()[0] == i

  def test_small_copies_from_host_buf_transfer(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    _ = Device[f"{Device.DEFAULT}:1"]

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(f"{Device.DEFAULT}:1", 1, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf3 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(host=True, nolru=True)).ensure_allocated()
    TestHCQ.d0.allocator.map(buf2._buf)

    for i in range(256):
      ctypes.memset(buf3._buf.va_addr, i, 1)

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf3._buf.va_addr, 1) \
                                  .copy(buf2._buf.va_addr, buf1._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf2.as_buffer()[0] == i

  def test_memory_barrier(self):
    a = Tensor([0, 1], device=Device.DEFAULT, dtype=dtypes.int8).realize()
    b = a + 1
    runner = get_runner(TestHCQ.d0.dname, create_schedule([b.lazydata])[-1].ast)

    buf1 = Buffer(Device.DEFAULT, 2, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 2, dtypes.int8, options=BufferOptions(cpu_access=True, nolru=True)).ensure_allocated()

    kernargs_ptr = runner.clprg.fill_kernargs([buf1._buf, buf2._buf])

    for i in range(255):
      ctypes.memset(buf2._buf.va_addr, i, 2)

      # Need memory_barrier after direct write to vram
      TestHCQ.d0.hw_compute_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                     .memory_barrier() \
                                     .exec(runner.clprg, kernargs_ptr, runner.p.global_size, runner.p.local_size) \
                                     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf1.as_buffer()[0] == (i + 1), f"has {buf1.as_buffer()[0]}, need {i + 1}"

  def test_memory_barrier_before_copy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    buf1 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(nolru=True)).ensure_allocated()
    buf3 = Buffer(Device.DEFAULT, 1, dtypes.int8, options=BufferOptions(cpu_access=True, nolru=True)).ensure_allocated()

    for i in range(256):
      ctypes.memset(buf3._buf.va_addr, i, 1)

      # Need memory_barrier after direct write to vram
      TestHCQ.d0.hw_compute_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                     .memory_barrier() \
                                     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_value += 1

      TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                  .copy(buf1._buf.va_addr, buf3._buf.va_addr, 1) \
                                  .copy(buf2._buf.va_addr, buf1._buf.va_addr, 1) \
                                  .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
      TestHCQ.d0.timeline_value += 1

      assert buf2.as_buffer()[0] == i

@contextlib.contextmanager
def helper_collect_profile(*devs, random_setup_delay=False):
  if random_setup_delay:
    devs = list(devs)
    for dev in devs: dev.synchronize()
    random.shuffle(devs)
    for dev in devs:
      dev._prof_setup()
      time.sleep(random.randint(1, 1000) / 1000)
  else:
    for dev in devs: dev._prof_setup()

  profile_dict = {}
  _, tmp = tempfile.mkstemp()
  with Context(PROFILE=1, PROFILEPATH=tmp):
    try: yield profile_dict
    finally:
      for dev in devs:
        dev.synchronize()
        dev._prof_finalize()
        atexit.unregister(dev._prof_finalize)

  for k,v in json.loads(pathlib.Path(tmp).read_text()).items(): profile_dict[k] = v
  pathlib.Path(tmp).unlink()

def helper_profile_filter_node(profile, **kwargs):
  assert len(profile) > 0, "Empty profile"
  assert 'traceEvents' in profile, "traceEvents should present"
  return [x for x in profile['traceEvents'] if all(x.get(k, None) == v for k,v in kwargs.items())]

def helper_profile_parse_pids(profile):
  pids, tids = {}, {}
  procs = helper_profile_filter_node(profile, name='process_name')
  for proc in procs: pids[proc['pid']] = proc['args']['name']
  threads = helper_profile_filter_node(profile, name='thread_name')
  for th in threads: tids[th['tid']] = th['args']['name']
  return pids, tids

def helper_profile_parse_deps(profile):
  deps = []
  for s in helper_profile_filter_node(profile, ph="s"):
    f = helper_profile_filter_node(profile, ph="f", id=s['id'])[0]

    starts, ends = [], []
    for x in helper_profile_filter_node(profile, ph="X"):
      if s['pid'] == x['pid'] and s['tid'] == x['tid'] and x['ts'] <= s['ts'] <= x['ts'] + x['dur']: starts.append(x)
      if f['pid'] == x['pid'] and f['tid'] == x['tid'] and x['ts'] <= f['ts'] <= x['ts'] + x['dur']: ends.append(x)

    assert len(starts) == 1 and len(ends) == 1, "more than one start and end possible, valid?"
    deps.append((s, f, starts[0], ends[0]))
  return deps

def helper_validate_node(node, duration_s=10, ts_age_s=30, profile=None, pid_name=None, tid_name=None):
  pids, tids = helper_profile_parse_pids(profile)
  assert abs(node['ts'] - time.perf_counter_ns() / 1e3) < ts_age_s * 1e6, "timestimp is not in 30s range"
  assert 0 < node['dur'] < duration_s * 1e6, "duration is not in 10s range"
  assert pid_name is None or pids[node['pid']] == pid_name
  assert tid_name is None or tids[node['tid']] == tid_name

@unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "HCQ device required to run")
class TestProfiler(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestProfiler.d0 = Device[Device.DEFAULT]

    TestProfiler.a = Tensor([0.,1.], device=Device.DEFAULT).realize()
    TestProfiler.b = self.a + 1
    si = create_schedule([self.b.lazydata])[-1]

    TestProfiler.runner = get_runner(TestProfiler.d0.dname, si.ast)
    TestProfiler.b.lazydata.buffer.allocate()

    TestProfiler.kernargs_ba_ptr = TestProfiler.runner.clprg.fill_kernargs([TestProfiler.b.lazydata.buffer._buf, TestProfiler.a.lazydata.buffer._buf])
    TestProfiler.kernargs_ab_ptr = TestProfiler.runner.clprg.fill_kernargs([TestProfiler.a.lazydata.buffer._buf, TestProfiler.b.lazydata.buffer._buf])

  def test_profile_kernel_run(self):
    runner_name = TestProfiler.runner.clprg.name
    with helper_collect_profile(TestProfiler.d0) as profile:
      TestProfiler.runner([TestProfiler.b.lazydata.buffer, TestProfiler.a.lazydata.buffer], var_vals={})

    kernel_node = helper_profile_filter_node(profile, name=runner_name)[0]
    helper_validate_node(kernel_node, profile=profile, pid_name=Device.DEFAULT, tid_name="COMPUTE")

  def test_profile_copyin(self):
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))

    copyin_node = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}")[0]
    helper_validate_node(copyin_node, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

  def test_profile_multiops(self):
    runner_name = TestProfiler.runner.clprg.name
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
      TestProfiler.runner([buf1, TestProfiler.a.lazydata.buffer], var_vals={})
      buf1.as_buffer()

    copyin_node = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}")[0]
    helper_validate_node(copyin_node, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

    kernel_node = helper_profile_filter_node(profile, name=runner_name)[0]
    helper_validate_node(kernel_node, profile=profile, pid_name=Device.DEFAULT, tid_name="COMPUTE")

    copyout_node = helper_profile_filter_node(profile, name=f"{Device.DEFAULT} -> CPU")[0]
    helper_validate_node(copyout_node, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

    assert copyin_node['ts'] + copyin_node['dur'] < kernel_node['ts'], "timestamp not aranged"
    assert kernel_node['ts'] + kernel_node['dur'] < copyout_node['ts'], "timestamp not aranged"

  def test_profile_multidev(self):
    d1 = Device[f"{Device.DEFAULT}:1"]
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(f"{Device.DEFAULT}:1", 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
      buf2.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))

    copyin_node_1 = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}")[0]
    helper_validate_node(copyin_node_1, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

    copyin_node_2 = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}:1")[0]
    helper_validate_node(copyin_node_2, profile=profile, pid_name=f"{Device.DEFAULT}:1", tid_name="DMA")

  @unittest.skipIf(MOCKGPU and Device.DEFAULT == "AMD", "AMD mockgpu with indirect buffers does not support queue wait interrupts")
  def test_profile_deps(self):
    d1 = Device[f"{Device.DEFAULT}:1"]

    def f(a):
      x = (a + 1).realize()
      return x, x.to(d1.dname).realize()

    a = Tensor.randn(10, 10, device=TestProfiler.d0.dname).realize()
    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      jf = TinyJit(f)
      for _ in range(3): jf(a)
      del jf

    deps = helper_profile_parse_deps(profile)
    assert len(deps) == 1, "one dep is expected, one launch"

    _, _, l, r = deps[0]
    assert l['name'].find("->") == -1, "should be kernel"
    assert r['name'] == f"{Device.DEFAULT} -> {Device.DEFAULT}:1", "should be copy"

  @unittest.skipIf(CI, "skip CI")
  def test_profile_sync(self):
    mv = memoryview(bytearray(struct.pack("ff", 0, 1)))
    expected_diff = 100000 # sleep in us

    devs = [Device[f"{Device.DEFAULT}:{i}"] for i in range(6)]
    bufs = [Buffer(f"{Device.DEFAULT}:{i}", 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated() for i in range(6)]

    # enqueue ops on different queues to check the timer sync
    cpu_time = []
    with helper_collect_profile(*devs, random_setup_delay=True) as profile:
      for i in range(6):
        x = time.perf_counter_ns()
        time.sleep(expected_diff / 1e6)
        bufs[i].copyin(mv)
        cpu_time.append(((time.perf_counter_ns() - x) / 1000) - expected_diff)

    nodes = [helper_profile_filter_node(profile, name=f"CPU -> {Device.canonicalize(f'{Device.DEFAULT}:{i}')}")[-1] for i in range(6)]
    avg_diff = []
    for i in range(1, 6):
      diff = nodes[i]['ts'] - nodes[i-1]['ts'] - cpu_time[i]
      avg_diff.append(diff - expected_diff)
      assert expected_diff * 0.998 < diff < expected_diff * 1.002, "more that 0.2% diff"

    print(f"total avg delay is {sum(avg_diff) / len(avg_diff)} us")

if __name__ == "__main__":
  unittest.main()
