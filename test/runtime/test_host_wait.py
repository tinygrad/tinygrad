import struct, threading, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer, BufferSpec
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp
from tinygrad.engine import realize as r


class TestHostWait(unittest.TestCase):
  def setUp(self):
    self.owner, self.producer = Device["CPU"], Device["CPU:8765"]
    self.owner.synchronize()
    self.addCleanup(self.owner.pending.clear)

  def mapped(self, size=16):
    buf = Buffer("CPU", size, preallocate=True)
    buf.get_buf(self.producer.device)
    return buf

  def test_unmapped_kernel_does_not_wait(self):
    producer = Device["CPU:8767"]
    src = Tensor([0.0], device="CPU").realize()
    out = src + 1
    linked = r.link_linear(r.compile_linear(out.schedule_linear()))
    r.run_linear(linked, jit=True)
    self.assertNotIn(producer, src.uop.buffer.base.get_storage().maps)
    self.assertNotIn(producer, out.uop.buffer.base.get_storage().maps)
    self.owner.pending[producer] = 7
    with patch.object(producer, "_wait_signal", side_effect=AssertionError("unrelated wait")):
      r.run_linear(linked, jit=True)
    self.assertEqual(out.uop.buffer.host.view(fmt="f")[0], 1.0)
    self.assertEqual(self.owner.pending[producer], 7)

  def test_cpu_kernel_waits_for_pending_copy(self):
    src = Tensor([0.0], device="CPU").realize()
    out = src + 1
    linked = r.link_linear(r.compile_linear(out.schedule_linear()))
    r.run_linear(linked, jit=True)  # warm the runtime before checking that execution blocks
    cpu, producer = self.owner, self.producer
    signal = producer.timeline.host.view(fmt='Q')
    signal[1] += 1
    cpu.pending[producer] = signal[1]
    src.uop.buffer.get_buf(producer.device)
    self.addCleanup(cpu.pending.pop, producer, None)
    entered = threading.Event()
    original_wait = producer._wait_signal
    def wait(*args, **kwargs):
      entered.set()
      return original_wait(*args, **kwargs)
    t = threading.Thread(target=r.run_linear, args=(linked,), kwargs={"jit": True}, daemon=True)
    with patch.object(producer, "_wait_signal", side_effect=wait):
      t.start()
      try:
        self.assertTrue(entered.wait(5), "CPU compute must wait for the producer")
        self.assertTrue(t.is_alive())
      finally:
        src.uop.buffer.host[:] = struct.pack("f", 2.0)
        signal[0] = signal[1]
        t.join(5)
    self.assertFalse(t.is_alive())
    self.assertEqual(out.tolist(), [3.0])

  def test_selected_producer_is_deduplicated(self):
    buf, unrelated = self.mapped(), Device["CPU:8766"]
    self.owner.pending.update({self.producer: 7, unrelated: 9})
    with patch.object(self.producer, "_wait_signal") as wait:
      r.wait_for_host_buffers([buf, buf.view(4, 4)], timeout=123)
    self.assertEqual(wait.call_count, 1)
    self.assertEqual(wait.call_args.args[1:], (7, 123))
    self.assertEqual(self.owner.pending, {unrelated: 9})

  def test_view_and_storage_reuse(self):
    with Context(LRU=1):
      buf = self.mapped(19)
      maps = buf.get_storage().maps
      buf.deallocate()
      reused = Buffer("CPU", 19, preallocate=True)
      self.assertIs(reused.get_storage().maps, maps)
      self.owner.pending[self.producer] = 7
      with patch.object(self.producer, "_wait_signal") as wait:
        r.wait_for_host_buffers([reused.view(4, 8)])
      self.assertEqual(wait.call_count, 1)

  def test_failed_wait_and_newer_target(self):
    buf = self.mapped()
    self.owner.pending[self.producer] = 7
    with patch.object(self.producer, "_wait_signal", side_effect=RuntimeError("held")):
      with self.assertRaisesRegex(RuntimeError, "held"):
        r.wait_for_host_buffers([buf])
    self.assertEqual(self.owner.pending[self.producer], 7)
    def advance(*args): self.owner.pending[self.producer] = 8
    with patch.object(self.producer, "_wait_signal", side_effect=advance):
      r.wait_for_host_buffers([buf])
    self.assertEqual(self.owner.pending[self.producer], 8)

  def test_external_pointer_fallback(self):
    original = self.mapped()
    alias = Buffer("CPU", 16).allocate(external_ptr=original.host.addr)
    self.assertNotIn(self.producer, alias.get_storage().maps)
    self.owner.pending[self.producer] = 7
    with patch.object(self.owner, "synchronize") as sync:
      r.wait_for_host_buffers([alias], timeout=123)
    sync.assert_called_once_with(timeout=123)

  def test_remote_fallback(self):
    buf = Buffer("CPU", 16, preallocate=True)
    self.owner.pending[self.producer] = 7
    with patch.object(self.owner, "remote", object()), patch.object(self.owner, "synchronize") as sync:
      r.wait_for_host_buffers([buf], timeout=123)
    sync.assert_called_once_with(timeout=123)

  def test_gpu_owner_fallback(self):
    # NULL is an HCQ device and lets this routing check run without GPU hardware.
    buf = Buffer("NULL", 16)
    with patch.object(buf.allocator.dev, "synchronize") as sync:
      r.wait_for_host_buffers([buf], timeout=123)
    sync.assert_called_once_with(timeout=123)

  def test_kernel_output_is_checked(self):
    src = Tensor([0.0], device="CPU").realize()
    out = src + 1
    linked = r.link_linear(r.compile_linear(out.schedule_linear()))
    r.run_linear(linked, jit=True)
    out.uop.buffer.get_buf(self.producer.device)
    self.owner.pending[self.producer] = 7
    with patch.object(self.producer, "_wait_signal") as wait:
      r.run_linear(linked, jit=True)
    self.assertEqual(wait.call_count, 1)
    self.assertEqual(out.uop.buffer.host.view(fmt="f")[0], 1.0)

  def test_jit_uses_current_input(self):
    producer = Device["CPU:8768"]
    @TinyJit
    def f(a): return (a + 1).realize()
    a = Tensor([0.0], device="CPU").realize()
    for _ in range(3): f(a)
    b = Tensor([2.0], device="CPU").realize()
    b.uop.buffer.get_buf(producer.device)
    self.owner.pending[producer] = 7
    with patch.object(producer, "_wait_signal") as wait:
      out = f(b)
    self.assertEqual(wait.call_count, 1)
    self.assertEqual(out.uop.buffer.host.view(fmt="f")[0], 3.0)

  def test_validation_uses_host_wait(self):
    a, b = Buffer("CPU", 4, preallocate=True), self.mapped(4)
    args = [UOp.from_buffer(x, dtypes.float32) for x in (a, b, a, b)]
    call = UOp.custom_function("validate", UOp.sink()).call(*args)
    program = SimpleNamespace(arg=SimpleNamespace(globals=(0, 1), outs=(),
                              launch_dims=lambda _: ((1, 1, 1), (1, 1, 1)), vals=lambda _: ()))
    self.owner.pending[self.producer] = 7
    def cpu_runtime(*args, **kwargs): self.assertNotIn(self.producer, self.owner.pending)
    with patch.object(r, "to_program", return_value=program), patch.object(r, "get_runtime", return_value=cpu_runtime):
      with patch.object(self.producer, "_wait_signal") as wait:
        r.exec_validate(r.ExecContext(), call, call.body)
    self.assertEqual(wait.call_count, 1)


@unittest.skipUnless(Device.DEFAULT.startswith("CUDA"), "CUDA copy queue required")
class TestHostWaitCUDA(unittest.TestCase):
  def test_delayed_copy_orders_only_mapped_cpu_work(self):
    from tinygrad.runtime.autogen import cuda
    from tinygrad.runtime.ops_cuda import check
    dev = Device[Device.DEFAULT]
    source = Tensor([2.0], device=dev.device).realize()
    dest = Tensor([0.0], device="CPU").realize()
    out, unrelated = dest + 1, Tensor([0.0], device="CPU").realize() + 1
    kernels = [r.link_linear(r.compile_linear(t.schedule_linear())) for t in (out, unrelated)]
    for kernel in kernels: r.run_linear(kernel, jit=True)
    copy = r.link_linear(r.compile_linear(UOp(Ops.LINEAR, src=(dest.uop.store_call(source.uop),))))
    gate = Buffer(dev.device, 8, options=BufferSpec(host=True, cpu_access=True), initial_value=bytes(8))
    dev.synchronize()
    Device["CPU"].synchronize()
    for dependent, kernel in [(False, kernels[1]), (True, kernels[0])]:
      dest.uop.buffer.host[:] = struct.pack("f", 0.0)
      gate.host.view(fmt="Q")[0] = 0
      entered, errors = threading.Event(), []
      original_wait = dev._wait_signal
      def wait(*args, **kwargs):
        entered.set()
        return original_wait(*args, **kwargs)
      def run():
        try: r.run_linear(kernel, jit=True)
        except BaseException as error: errors.append(error)
      worker = threading.Thread(target=run, daemon=True)
      try:
        check(cuda.cuStreamWaitValue64_v2(dev.streams[1], gate._buf, 1, cuda.CU_STREAM_WAIT_VALUE_GEQ))
        r.run_linear(copy, jit=True)
        with patch.object(dev, "_wait_signal", side_effect=wait):
          worker.start()
          if dependent:
            self.assertTrue(entered.wait(5))
            self.assertTrue(worker.is_alive())
          else:
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(entered.is_set())
          gate.host.view(fmt="Q")[0] = 1
          worker.join(5)
      finally:
        gate.host.view(fmt="Q")[0] = 1
        if worker.ident is not None: worker.join(5)
        dev.synchronize()
        Device["CPU"].synchronize()
      self.assertFalse(worker.is_alive())
      self.assertEqual(errors, [])
    self.assertEqual(out.tolist(), [3.0])


if __name__ == "__main__": unittest.main()
