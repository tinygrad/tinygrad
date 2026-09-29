import unittest, gc, struct, ctypes, threading, numpy as np
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes, GlobalCounters
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context, unwrap
from tinygrad.uop.ops import Ops, UOp, KernelInfo
from tinygrad.engine.realize import compile_linear, link_linear, lower_and_compile, run_linear
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.nir import NIRRenderer
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.c import init_c_struct_t
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.hcq2 import HCQ_DEVS, all_devices_in, hcq_compile_cache
from test.null.test_hcq2 import chain, chain_input, compiled_chain

def cpu_buf(size:int=1, dtype=dtypes.uint8, **kwargs) -> UOp: return UOp.placeholder((size,), dtype, device="CPU", **kwargs)

def lower_hcq(body:UOp) -> UOp:
  return unwrap(hcq2.lower_call(UOp.sink(body, arg=KernelInfo("test")).call(aux=hcq2.HCQInfo(("CPU",)))))

@unittest.skipUnless(all_devices_in(Device.DEFAULT, HCQ_DEVS) and not Device.DEFAULT.startswith("NULL"), "hcq2 device required")
class TestHCQ2Schedule(unittest.TestCase):
  def test_compile_and_link_are_idempotent(self):
    for jit in (False, True):
      with self.subTest(jit=jit):
        out, compiled, inputs = compiled_chain(2, jit=jit, device=Device.DEFAULT)
        linked = link_linear(compiled, input_uops=inputs, allow_cache=not jit)
        before = tuple(inputs)
        for linear in (compiled, linked): self.assertIs(compile_linear(linear, input_uops=inputs, cache=not jit), linear)
        self.assertIs(link_linear(linked, input_uops=inputs, allow_cache=not jit), linked)
        self.assertEqual(tuple(inputs), before)
        run_linear(linked, input_uops=inputs, jit=True, wait=True)
        self.assertEqual(out.tolist(), [4] * 4)

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

  def test_repeated_copy(self):
    vram, host, new = [Buffer(d, 4096, dtypes.uint8, preallocate=True) for d in (Device.DEFAULT, "CPU", "CPU")]
    new.host[:] = bytes(range(256)) * 16
    copyout, copyin = UOp.from_buffer(host).store_call(UOp.from_buffer(vram)), UOp.from_buffer(vram).store_call(UOp.from_buffer(new))
    run_linear(UOp(Ops.LINEAR, src=(copyout, copyin, copyout)), wait=True)
    self.assertEqual(bytes(host.host[:]), bytes(new.host[:]))

  @unittest.skipIf(Device.DEFAULT == "METAL", "unified memory: METAL copies on the host and maps nothing")
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

  def test_jit_has_no_rt_buffers(self):
    # a one shot link borrows ring slots, a jit's link owns its buffers: nothing it keeps may come from the ring
    dev = Device[Device.DEFAULT]
    ranges = [((b:=dev.rt_buffer(True, host))._buf, b._buf + b.nbytes) for host in (False, True)]
    x, f = chain_input(device=dev.device), TinyJit(lambda a: chain(a, 2).realize())
    for _ in range(2): f(x)
    for u in f.captured.linear.toposort():
      if u.op is Ops.BUFFER and u.addrspace is AddrSpace.GLOBAL and (buf:=u.buffer).device == dev.device:
        self.assertFalse(any(buf._buf < end and start < buf._buf + buf.nbytes for start, end in ranges))

# the fence and the ffi run on the CPU runtime, which is always available
@unittest.skipIf(isinstance(Device["CPU"].renderer, NIRRenderer), "segfaults compiling the fence loop with LVP")
class TestHCQ2Fence(unittest.TestCase):
  def setUp(self):
    self.enterContext(Context(HCQ_RUNTIME_DEV="CPU"))
    self.tl = Device["CPU"].timeline.host.view(fmt='Q')
    self.addCleanup(lambda: self.tl.__setitem__(0, self.tl[1]))

  def test_a_schedule_waits_for_its_previous_run(self):
    slots = UOp.placeholder((4,), dtypes.uint64, device=("CPU",), volatile=True, tag="slots")
    program = lower_and_compile(UOp(Ops.LINEAR, src=(lower_hcq(UOp.custom_function("hcq_fence", slots[0:2], slots[2:4])),)))
    linked = hcq2.hcq_link(program, allow_cache=False)
    (i,) = [i for i, p in enumerate(program.src[0].without_after.src[1:]) if p.arg.name == "slots"]
    slots_mv = linked.src[0].without_after.src[1 + i].buffer.host.view(fmt='Q')
    slots_mv[2], base = 7, self.tl[1]

    run_linear(linked, jit=True)
    self.assertEqual((self.tl[1], slots_mv[0], slots_mv[2]), (base + 1, base + 1, 0), "the run is announced, recorded, the signal re-armed")

    t = threading.Thread(target=run_linear, args=(linked,), kwargs={"jit": True}, daemon=True)
    t.start()
    t.join(0.2)
    self.assertTrue(t.is_alive(), "the second run must wait for the first to finish")
    self.tl[0] = base + 1
    t.join(5)
    self.assertFalse(t.is_alive())
    self.assertEqual(self.tl[1], base + 2)

@unittest.skipUnless(isinstance(Device["CPU"].renderer, CStyleLanguage), "CALL is rendered in C style only")
class TestHCQ2FFI(unittest.TestCase):
  @staticmethod
  def _run(body:UOp) -> list[Buffer]:
    linear = hcq2.hcq_link(lower_and_compile(UOp(Ops.LINEAR, src=(lower_hcq(body),))), allow_cache=False)
    run_linear(linear, jit=True)
    return [u.buffer for u in linear.src[0].without_after.src[1:] if u.op is Ops.BUFFER]

  def test_ffi_ccall(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      out = cpu_buf(dtype=dtypes.int32, slot=1, volatile=True, tag="ffi_result")
      bufs = self._run(out.index(0).store(hcq2.ccall(libc.dll.ffs, 0x10)))
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.int).host.view(fmt='i')[0], 5)

  def test_ffi_cstruct(self):
    struct_t = init_c_struct_t(16, (("u8", ctypes.c_uint8, 0), ("u16", ctypes.c_uint16, 2),
                                  ("u32", ctypes.c_uint32, 4), ("u64", ctypes.c_uint64, 8)))
    cpu_buf() # reserve slot zero for device-owned placeholders
    with Context(HCQ_RUNTIME_DEV="CPU"):
      s = hcq2.cstruct(struct_t, u8=0x12, u16=UOp.const(0x3456, dtypes.uint16), u32=0x789ABCDE, u64=0xFEDCBA9876543210)
      bufs = self._run(s.index(0).load())
    got = struct_t.from_buffer_copy(bytes(next(b for b in bufs if b.nbytes == ctypes.sizeof(struct_t)).host.view(fmt='B')))
    self.assertEqual((got.u8, got.u16, got.u32, got.u64), (0x12, 0x3456, 0x789ABCDE, 0xFEDCBA9876543210))

  def test_nested_cstruct_patches(self):
    with Context(HCQ_RUNTIME_DEV="CPU"):
      inner = hcq2.cstruct(init_c_struct_t(8, (("pad", ctypes.c_uint32, 0), ("value", ctypes.c_uint32, 4))), value=42)
      outer = hcq2.cstruct(init_c_struct_t(8, (("ptr", ctypes.c_uint64, 0),)), ptr=inner[4:8].getaddr("CPU"))
      out = cpu_buf(dtype=dtypes.uint32, tag="result")
      copied = hcq2.ccall(libc.memcpy, out.index(0), outer.bitcast(dtypes.uint64).index(0).load(), 4)
      bufs = self._run(out.after(copied).index(0).load())
    self.assertEqual(next(b for b in bufs if b.dtype is dtypes.uint32).host.view(fmt='I')[0], 42)

if __name__ == "__main__":
  unittest.main()
