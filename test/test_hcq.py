import unittest, ctypes, struct
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import CI, getenv
from tinygrad.device import Buffer, BufferOptions, HCQCompiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import get_runner, CompiledRunner
from tinygrad.codegen.kernel import Kernel, Opt, OptOps

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

    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"

  def test_exec_2_kernels_100_times(self):
    q = TestHCQ.d0.hw_compute_queue_t()
    q.wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
     .exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ab_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    for _ in range(100):
      q.update_wait(0, value=TestHCQ.d0.timeline_value - 1).update_signal(3, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
      TestHCQ.d0.timeline_value += 1

    val = TestHCQ.a.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 200.0, f"got val {val}"

  def test_exec_update(self):
    q = TestHCQ.d0.hw_compute_queue_t()
    q.exec(TestHCQ.runner.clprg, TestHCQ.kernargs_ba_ptr, TestHCQ.runner.p.global_size, TestHCQ.runner.p.local_size) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    q.update_exec(0, (1,1,1), (1,1,1))
    q.submit(TestHCQ.d0)
    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[0]
    assert val == 1.0, f"got val {val}"
    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 0.0, f"got val {val}, should not be updated"

  def test_exec_update_fuzz(self):
    a = Tensor.randint((3, 3, 3), dtype=dtypes.int, device=Device.DEFAULT).realize()
    b = a + 1
    si = create_schedule([b.lazydata])[-1]
    k = Kernel(si.ast, opts=TestHCQ.d0.renderer)
    for i in range(3): k.apply_opt(Opt(op=OptOps.LOCAL, axis=0, amt=3))

    runner = CompiledRunner(k.to_program())

    zb = Buffer(Device.DEFAULT, 3 * 3 * 3, dtypes.int, options=BufferOptions(cpu_access=True, nolru=True)).ensure_allocated()
    zt = Buffer(Device.DEFAULT, 3 * 3 * 3, dtypes.int, options=BufferOptions(cpu_access=True, nolru=True)).ensure_allocated()
    ctypes.memset(zb._buf.va_addr, 0, zb.nbytes)
    kernargs = runner.clprg.fill_kernargs([zt._buf, zb._buf])

    q = TestHCQ.d0.hw_compute_queue_t()
    q.memory_barrier() \
     .exec(runner.clprg, kernargs, (1,1,1), (1,1,1)) \
     .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value)

    for x in range(1, 4):
      for y in range(1, 4):
        for z in range(1, 4):
          ctypes.memset(zt._buf.va_addr, 0, zb.nbytes)

          q.update_exec(1, local_size=(x,y,z)) \
           .update_signal(2, value=TestHCQ.d0.timeline_value).submit(TestHCQ.d0)
          TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
          TestHCQ.d0.timeline_value += 1

          res_sum = sum(x for x in zt.as_buffer().cast("I"))
          assert x * y * z == res_sum, f"want {x * y * z}, got {res_sum}"

  # Test copy
  def test_copy(self):
    if TestHCQ.d0.hw_copy_queue_t is None: self.skipTest("device does not support copy queue")

    TestHCQ.d0.hw_copy_queue_t().wait(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value - 1) \
                                .copy(TestHCQ.b.lazydata.buffer._buf.va_addr, TestHCQ.a.lazydata.buffer._buf.va_addr, 8) \
                                .signal(TestHCQ.d0.timeline_signal, TestHCQ.d0.timeline_value).submit(TestHCQ.d0)

    TestHCQ.d0.timeline_signal.wait(TestHCQ.d0.timeline_value)
    TestHCQ.d0.timeline_value += 1

    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 1.0, f"got val {val}"

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

    val = TestHCQ.b.lazydata.buffer.as_buffer().cast("f")[1]
    assert val == 1.0, f"got val {val}"

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

    try: d1 = Device[f"{Device.DEFAULT}:1"]
    except Exception: self.skipTest("no multidevice, test skipped")

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
    assert 1 <= et <= (7000 if CI else 30)

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

    try: _ = Device[f"{Device.DEFAULT}:1"]
    except Exception: self.skipTest("no multidevice, test skipped")

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

    try: _ = Device[f"{Device.DEFAULT}:1"]
    except Exception: self.skipTest("no multidevice, test skipped")

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

if __name__ == "__main__":
  unittest.main()
