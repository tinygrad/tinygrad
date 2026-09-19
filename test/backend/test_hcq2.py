import unittest, contextlib, gc, struct, numpy as np
from unittest.mock import patch
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, GlobalCounters
from tinygrad.device import Buffer, Compiled
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import unwrap
from tinygrad.engine.realize import compile_linear, link_linear, run_linear
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in, hcq_compile_cache
from test.null.test_hcq2 import compiled_chain

@contextlib.contextmanager
def rt_buffers():
  calls, orig = [], Compiled.rt_buffer
  def track(dev, *args, **kwargs):
    calls.append(dev)
    return orig(dev, *args, **kwargs)
  with patch.object(Compiled, "rt_buffer", track): yield calls

@unittest.skipUnless(all_devices_in(Device.DEFAULT, HCQ_DEVS) and not Device.DEFAULT.startswith("NULL"), "hcq2 device required")
class TestHCQ2Schedule(unittest.TestCase):
  def test_amd_cmdbuf_uncached(self):
    dev = Device[Device.DEFAULT]
    if not dev.device.startswith("AMD") or not dev.is_am(): self.skipTest("AMD PCI interface required")
    for name, uncached in (("cmdbuf", True), ("kernargs", False)):
      b = UOp.placeholder((256,), dtypes.uint8, device=(dev.device,), tag=hcq2.to_name(name, "COMPUTE:0"))
      buf = unwrap(hcq2.bufferize_buf(hcq2.LinkCtx({}, use_rt=False), b)).buffer
      self.assertEqual(buf.base.options.uncached, uncached)
      self.assertEqual(buf.base.meta.mapping.uncached, uncached)

  def test_double_compile(self):
    for n in (1, 65):
      for jit in (False, True):
        with self.subTest(kernels=n, jit=jit):
          out, compiled, inputs = compiled_chain(n, jit=jit)
          linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
          before = tuple(inputs)
          with rt_buffers() as borrowed:
            for linear in (compiled, linked):
              self.assertIs(compile_linear(linear, input_uops=inputs, cache=not jit), linear)
          self.assertEqual(tuple(inputs), before)
          self.assertFalse(borrowed)
          run_linear(linked, input_uops=inputs, jit=True, wait=True)
          self.assertEqual(out.tolist(), [2 + n] * 4)

  def test_double_link(self):
    for n in (1, 65):
      for jit in (False, True):
        with self.subTest(kernels=n, jit=jit):
          out, compiled, inputs = compiled_chain(n, jit=jit)
          linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
          with rt_buffers() as borrowed:
            again = link_linear(linked, input_uops=inputs, allow_cache=not jit)
          self.assertIs(again, linked)
          self.assertFalse(borrowed)
          run_linear(again, input_uops=inputs, jit=True, wait=True)
          self.assertEqual(out.tolist(), [2 + n] * 4)

  def test_jit_new_inputs_each_call(self):
    @TinyJit
    def f(a, b): return (a * b + a).contiguous().realize()
    ins = [(Tensor.full((23,), float(i)).contiguous().realize(), Tensor.full((23,), 2.0).contiguous().realize()) for i in range(6)]
    for a, b in ins[:3]: f(a, b).tolist() # warm the jit and the copyout

    before = len(hcq_compile_cache)
    for i, (a, b) in enumerate(ins[3:], 3): self.assertEqual(f(a, b).tolist(), [i * 3.0] * 23)
    self.assertEqual(len(hcq_compile_cache), before)

  def test_jit_symbolic(self):
    @TinyJit
    def f(a): return (a + 1).sum().contiguous().realize()
    a = Tensor.rand(3, 10).contiguous().realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      np.testing.assert_allclose(f(a[:, :vi]).item(), (a[:, :i] + 1).sum().item(), atol=1e-5, rtol=1e-5)

  def test_map_cpu_buffer_preserves_contents(self):
    src = Buffer("CPU", 16, dtypes.uint8, preallocate=True)
    data = bytes(range(16))
    src.host[:] = data
    src.get_buf(Device.DEFAULT)
    self.assertEqual(bytes(src.as_memoryview()), data)

  def test_caches_hold_no_buffers(self):
    # an eager template caches without its buffers and the jit's linear compiles once uncached: freeing the tensors frees the device memory
    def step(i):
      buf = Buffer("NPY", 1024, dtypes.float32, initial_value=struct.pack("f", i) * 1024)
      x = Tensor(UOp.from_buffer(buf)).to(Device.DEFAULT).realize()
      @TinyJit
      def f(a): return (a * 2 + 1).contiguous().realize()
      for _ in range(3): out = f(x)
      self.assertEqual(out.to("CPU").tolist(), [2.0 * i + 1] * 1024)
    step(1) # warms the programs, templates and rings
    gc.collect()
    used = GlobalCounters.mem_used
    for i in range(2, 5): step(i)
    gc.collect()
    self.assertEqual(GlobalCounters.mem_used, used)

  def test_device_state_survives_as_link_refs(self):
    # a buffer the commands only address, never a param of the body, is kept by the linked call as a ref of what its getaddr resolved into
    dev = Device[Device.DEFAULT]
    names = {"AMD": () if getattr(dev, "is_aql", False) else ("scratch",), # the aql descriptor holds the scratch, nothing addresses it
             "NV": ("timeline",), "QCOM": ("_stack", "dummy"), "CUDA": ("timeline",), "NULL": ()}[Device.DEFAULT.split(":")[0]]
    @TinyJit
    def f(a): return (a * 2 + 1).contiguous().realize()
    x = Tensor.ones(16).contiguous().realize()
    for _ in range(3): f(x)
    call = f.captured.linear.src[0]
    self.assertIs(call.op, Ops.AFTER, "the linked call sits after its refs")
    refs = [u.buffer for u in call.src[1:] if u.op is Ops.BUFFER]
    for n in names: self.assertTrue(any(r is getattr(dev, n) for r in refs), f"{n} is not a ref of the call")

if __name__ == "__main__":
  unittest.main()
