import unittest, contextlib, numpy as np
from unittest.mock import patch
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import HCQ2, dedup, partition
from tinygrad.uop.ops import Ops, UOp
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, HCQ2Compiled, all_devices_in, hcq_compile_cache, link_linear_cache
from test.helpers import call_is_hcq

@contextlib.contextmanager
def rt_views():
  calls, orig = [], HCQ2Compiled.rt_view
  with patch.object(HCQ2Compiled, "rt_view", lambda s, *a, **kw: (calls.append(s), orig(s, *a, **kw))[1]): yield calls

@contextlib.contextmanager
def encoded_batches():
  batches, orig = [], hcq2.lower_and_compile
  with patch.object(hcq2, "lower_and_compile", lambda l, *a, **kw: (batches.extend(c for c in l.src if call_is_hcq(c)), orig(l, *a, **kw))[1]):
    yield batches

def patch_words(batch:UOp) -> list[UOp]:
  return [w for s in batch.src[0].toposort() if s.op is Ops.STORE and s.src[0].op is Ops.INDEX and s.src[0].src[1].op is Ops.STACK
          and s.src[1].op is Ops.STACK for w in s.src[1].src]

def rt_params(batch:UOp) -> list[str]:
  return dedup([u.arg.name for w in patch_words(batch) for u in w.toposort() if u.op is Ops.PARAM and u.arg.addrspace is AddrSpace.GLOBAL])

unittest.skipUnless(HCQ2 and all_devices_in(Device.DEFAULT, HCQ_DEVS - {"CPU"}), "non-CPU hcq2 device required")
class TestHCQ2Core(unittest.TestCase):
  def test_jit_has_no_rt_buffers(self):
    x = Tensor.ones(16).contiguous().realize()
    @TinyJit
    def f(a): return (a + 2).contiguous().realize()
    f(x)

    before = len(link_linear_cache)
    with rt_views() as calls:
      out = f(x)
      self.assertGreater(len(link_linear_cache), before)
      self.assertEqual(len(calls), 0)
      (x + 1).contiguous().realize()
      self.assertGreater(len(calls), 0)
    self.assertEqual(out.tolist(), [3.0] * 16)

  def test_jit_survives_ring_wrap(self):
    # the ring recycles with no liveness tracking, so eager work that wraps it must not land on the jit's buffers
    dev = Device[Device.DEFAULT]
    allocs = {host:dev.rt_allocator(True, host) for host in (False, True)}
    for host in allocs: dev.rt_buffer(True, host) # cache the full-sized backing buffers before temporarily shrinking their allocators
    with patch.object(allocs[False], "size", 1 << 13), patch.object(allocs[True], "size", 1 << 13):
      x = Tensor.ones(24).contiguous().realize()
      @TinyJit
      def g(a): return (a * 3 - 1).contiguous().realize()
      for _ in range(3): g(x)

      wrapped = 0
      for i in range(48):
        before = dev.rt_allocator(True, False).ptr
        (x + i).contiguous().realize()
        wrapped += dev.rt_allocator(True, False).ptr < before
        self.assertEqual(g(x).tolist(), [2.0] * 24)
      self.assertGreater(wrapped, 0)

  def test_jit_new_inputs_each_call(self):
    @TinyJit
    def f(a, b): return (a * b + a).contiguous().realize()
    ins = [(Tensor.full((23,), float(i)).contiguous().realize(), Tensor.full((23,), 2.0).contiguous().realize()) for i in range(6)]
    for a, b in ins[:3]: f(a, b).tolist() # warm the jit and the copyout

    before = len(hcq_compile_cache)
    self.assertEqual([f(a, b).tolist() for a, b in ins[3:]], [[i * 3.0] * 23 for i in range(3, 6)])
    self.assertEqual(len(hcq_compile_cache), before)

  def test_jit_symbolic(self):
    @TinyJit
    def f(a): return (a + 1).sum().contiguous().realize()
    a = Tensor.rand(3, 10).contiguous().realize()
    for i in range(1, 5):
      vi = Variable("i", 1, 10).bind(i)
      np.testing.assert_allclose(f(a[:, :vi]).item(), (a[:, :i] + 1).sum().item(), atol=1e-5, rtol=1e-5)

  def test_staged_copy_roundtrip(self):
    # a host buffer the device cannot read copies in chunks through a small ring of staging slots: every rotation must land bit-exact
    stage = Buffer("CPU", size:=1 << 16, dtypes.uint8, preallocate=True)
    for npdt in (np.uint8, np.float32):
      with self.subTest(dtype=npdt.__name__):
        n = (size // 2 // np.dtype(npdt).itemsize) * 9 + 7 # nine rotations of a two slot ring, plus a short tail
        data = np.arange(n, dtype=np.int64).astype(npdt)
        with patch.object(hcq2, "STAGING_SIZE", size), patch.object(hcq2, "STAGING_SLOTS", 2), patch.object(hcq2, "_staging", lambda: stage):
          out = Tensor(data).to(Device.DEFAULT).contiguous().realize()
          np.testing.assert_equal(out.numpy(), data)

  def test_rt_patches_are_inputs_and_vars_only(self):
    x = Tensor.rand(17, 33).contiguous().realize()
    with encoded_batches() as batches:
      @TinyJit
      def f(a): return (a.sin() * 3).contiguous().realize()
      for _ in range(3): f(x)

    jit, eager = partition(batches, lambda c: c.arg.aux.table >= 0)
    self.assertTrue(jit and eager, f"want both kinds of batch, got {len(jit)} jit and {len(eager)} eager")
    for c in batches:
      self.assertTrue(all(n.startswith(("inputs_", "timeline_")) for n in rt_params(c)), f"runtime patch reads {rt_params(c)}")
      self.assertFalse([u for w in patch_words(c) for u in w.toposort() if u.op is Ops.GETADDR], "addresses bake at link time")
    self.assertTrue(any(n.startswith("inputs_") for c in jit for n in rt_params(c)), "the jit patches its input addresses in")
    self.assertFalse(any(n.startswith("inputs_") for c in eager for n in rt_params(c)), "eager bakes its input addresses")

if __name__ == "__main__":
  unittest.main()
