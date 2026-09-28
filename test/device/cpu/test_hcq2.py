import unittest, ctypes, threading
from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import Context, unwrap
from tinygrad.uop.ops import Ops, UOp, KernelInfo
from tinygrad.engine.realize import lower_and_compile, run_linear
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.renderer.nir import LVPRenderer
from tinygrad.runtime.autogen import libc
from tinygrad.runtime.support.c import init_c_struct_t
import tinygrad.runtime.support.hcq2 as hcq2

def cpu_buf(size:int=1, dtype=dtypes.uint8, **kwargs) -> UOp: return UOp.placeholder((size,), dtype, device="CPU", **kwargs)

def lower_hcq(body:UOp) -> UOp:
  return unwrap(hcq2.lower_call(UOp.sink(body, arg=KernelInfo("test")).call(aux=hcq2.HCQInfo(("CPU",)))))

@unittest.skipIf(Device.DEFAULT != "CPU", "only run on CPU")
class TestHCQ2Fence(unittest.TestCase):
  def setUp(self):
    self.enterContext(Context(HCQ_RUNTIME_DEV="CPU"))
    if isinstance(Device.default.renderer, LVPRenderer): self.skipTest("LVP's workgroup barrier cannot implement a host fence")
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

@unittest.skipUnless(Device.DEFAULT == "CPU" and isinstance(Device.default.renderer, CStyleLanguage), "CALL is rendered in C style only")
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
