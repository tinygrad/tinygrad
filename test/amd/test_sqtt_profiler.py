import unittest, contextlib
from tinygrad import Device, Tensor, Context, TinyJit, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.device import Compiled, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.viz.serve import load_amd_counters, VizData
from tinygrad.renderer.amd.sqtt import decode, print_packets
from tinygrad.renderer.amd.dsl import s

@contextlib.contextmanager
def save_sqtt():
  Device[Device.DEFAULT].synchronize()
  profile_start = len(Compiled.profile_events)
  data = VizData()
  yield data.ctxs
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  load_amd_counters(data, [e for e in Compiled.profile_events[:profile_start] if isinstance(e, ProfileProgramEvent)] +
                          Compiled.profile_events[profile_start:])
  data.ctxs[:] = [r for r in data.ctxs if r["name"].startswith("SQTT")]

@contextlib.contextmanager
def save_sqtt_blobs():
  Device[Device.DEFAULT].synchronize()
  profile_start = len(Compiled.profile_events)
  data = []
  yield data
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  data[:] = [e for e in Compiled.profile_events[profile_start:] if isinstance(e, ProfileSQTTEvent)]

def custom_asm_cdna(A:UOp):
  import tinygrad.runtime.autogen.amd.cdna.ins as cdna
  WAVE_SIZE = 64
  insts = [cdna.s_nop(0), cdna.s_mov_b32(s[0], 10)]
  return custom_asm(A, insts+[cdna.s_endpgm()], WAVE_SIZE*2)

def custom_asm_rdna(A:UOp):
  import tinygrad.runtime.autogen.amd.rdna3.ins as rdna3
  WAVE_SIZE = 32
  insts = [rdna3.s_nop(0), rdna3.s_mov_b32(s[0], 10)]
  return custom_asm(A, insts+[rdna3.s_endpgm()], WAVE_SIZE*2)

def custom_asm(A, insts, num_threads) -> UOp:
  return UOp(Ops.PROGRAM, src=(UOp.sink(A, UOp.special(num_threads, "lidx0"), arg=KernelInfo("asm")), \
      UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS,arg=(x,dtypes.void)) for x in insts]))))

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestSQTTProfiler(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled: raise unittest.SkipTest("device must be in SQTT profiling mode")
    cls.arch = Device[Device.DEFAULT].arch

  def test_simple(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as sqtt:
      linear = t.schedule_linear()
      run_linear(linear)
    fn_name = to_program(linear.src[0].src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
    self.assertEqual(len(sqtt), 1)
    self.assertEqual(sqtt[0]["name"], f"SQTT {fn_name}")

  def test_asm(self):
    t = Tensor.empty(1)
    with save_sqtt_blobs() as sqtt:
      t.custom_kernel(fxn=custom_asm_cdna if self.arch == "gfx950" else custom_asm_rdna)[0].realize()
    for event in sqtt:
      if not event.itrace: continue
      print(f"\n=== SE {event.se} ===")
      print_packets(decode(event.blob))

  def test_multiple_runs(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as sqtt:
      linear = t.schedule_linear()
      for _ in range(N:=3): run_linear(linear)
    fn_name = to_program(linear.src[0].src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
    self.assertEqual(len(sqtt), N)
    for i in range(1, N):
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name} n{i+1}")

  def test_multiple_kernels(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    linear = t.schedule_linear()
    with save_sqtt() as sqtt:
      run_linear(linear)
    self.assertEqual(len(sqtt), len(linear.src))
    for i,call in enumerate(linear.src):
      fn_name = to_program(call.src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name}")

  def test_multiple_kernels_lower(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    linear = t.schedule_linear()
    with save_sqtt() as sqtt:
      run_linear(linear)
    self.assertEqual(len(sqtt), len(linear.src))
    for i,call in enumerate(linear.src):
      fn_name = to_program(call.src[0], renderer=Device[Device.DEFAULT].renderer).arg.function_name
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name}")

  def test_jit(self):
    @TinyJit
    def f(a): return a + 1
    t = Tensor.empty(1)
    with save_sqtt() as sqtt:
      for _ in range(N:=5):
        f(t).realize()
    self.assertEqual(len(sqtt), N)
    kernel_name = sqtt[0]["name"]
    for i,e in enumerate(sqtt[1:], start=1): self.assertEqual(e["name"], f"{kernel_name} n{i+1}")

  # TODO: can we trace SQTT for graphed kernels?
  def test_jit_graph(self, kernel_count=3*1):
    @TinyJit
    def f(a): return ((a + 1).contiguous() + 2).contiguous().sum()
    t = Tensor.empty(32)
    with save_sqtt() as sqtt:
      for _ in range(5):
        f(t).realize()
    names = [s["name"] for s in sqtt]
    k0, k1, k2 = names[:3]
    for i in range(3, len(sqtt), 3):
      n = (i // 3)+1
      self.assertEqual(names[i], f"{k0} n{n}")
      self.assertEqual(names[i+1], f"{k1} n{n}")
      self.assertEqual(names[i+2], f"{k2} n{n}")
    self.assertEqual(len(sqtt), kernel_count)

  @Context(JIT=2)
  def test_jit_multiple_kernels(self): self.test_jit_graph(kernel_count=3*5)

if __name__ == "__main__":
  unittest.main()
