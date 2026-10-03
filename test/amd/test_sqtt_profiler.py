import unittest, contextlib, functools, itertools, ctypes
from types import SimpleNamespace
from unittest.mock import Mock
from tinygrad import Device, Tensor, Context, TinyJit, dtypes
from tinygrad.dtype import AddrSpace
from test.helpers import is_hcq2_device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.device import Compiled, ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent, get_sqtt_wgp
from tinygrad.engine.realize import run_linear
from tinygrad.codegen import to_program
from tinygrad.viz.serve import load_amd_counters, VizData
from tinygrad.renderer.amd.sqtt import decode, print_packets
from tinygrad.renderer.amd.dsl import s, v
from tinygrad.helpers import getenv, HCQ2
from tinygrad.runtime.autogen import amdgpu_drm
from tinygrad.runtime.autogen.am import navi_offsets, pm4_nv, soc_11
from tinygrad.runtime.support.amd import AMDIP

class TestSQTTConfiguration(unittest.TestCase):
  def test_active_wgp_topology(self):
    info = amdgpu_drm.struct_drm_amdgpu_info_device(num_shader_engines=1, num_shader_arrays_per_engine=1)
    for mask, expected in ((0x3ff, 4), (0xff, 3), (0xc003, 1), (1, 0), (0x80000001, 1)):
      with self.subTest(mask=mask):
        info.cu_bitmap[0][0] = mask
        self.assertEqual(get_sqtt_wgp(info), expected)
    info.num_shader_engines, info.num_shader_arrays_per_engine = 5, 2
    info.cu_bitmap[0][0], info.cu_bitmap[1][1], info.cu_bitmap[0][3] = 0xf, 0xff, 0x3ff
    self.assertEqual(get_sqtt_wgp(info), 4) # maximum is in SE4/SA1, not the traced SE0/SA0
    with self.assertRaisesRegex(AssertionError, "active WGP"):
      get_sqtt_wgp(amdgpu_drm.struct_drm_amdgpu_info_device(num_shader_engines=1, num_shader_arrays_per_engine=1))

  def test_mock_topology(self):
    from test.mockgpu.amd.amddriver import DRMFileDesc
    info = amdgpu_drm.struct_drm_amdgpu_info_device()
    query = amdgpu_drm.struct_drm_amdgpu_info(query=amdgpu_drm.AMDGPU_INFO_DEV_INFO,
                                              return_pointer=ctypes.addressof(info), return_size=ctypes.sizeof(info))
    DRMFileDesc(0, None, None).ioctl(0, 0, ctypes.addressof(query))
    self.assertEqual((info.num_shader_engines, info.num_shader_arrays_per_engine), (6, 2))
    self.assertEqual([list(row) for row in info.cu_bitmap], [[0xff]*4, [0xff]*4, [0xff,0xff,0,0], [0xff,0xff,0,0]])

  @unittest.skipUnless(HCQ2, "HCQ2 required")
  def test_capture_and_dispatch_registers(self):
    from tinygrad.runtime import ops_amd
    gc = AMDIP("gc", (11,0,0), bases={0: tuple(getattr(navi_offsets, f"GC_BASE__INST0_SEG{s}", 0) for s in range(6))})
    q = ops_amd.AMDComputeQueue.__new__(ops_amd.AMDComputeQueue)
    q.dev = SimpleNamespace(sqtt_wgp=0, sqtt_ses=3, sqtt_win=4096, prof_slots=2, xccs=1, se_cnt=3, cu_cnt=54,
                            sqtt_next_cmd_id=itertools.count(), iface=SimpleNamespace(props={"cu_per_simd_array": 10}))
    q.target, q.pm4, q.soc, q.gc, q.devs = (11,0,1), pm4_nv, soc_11, gc, ("AMD",)
    q.memory_barrier, q.sqtt_userdata, q.wreg = Mock(), Mock(), Mock(wraps=q.wreg)
    q.prof_buf = Mock(return_value=UOp.const(0x100000, dtypes.uint64))
    def clear():
      q.blob, q.patches = bytearray(), []
      q.wreg.reset_mock()
    for wgp in (0, 4):
      with self.subTest(wgp=wgp):
        clear()
        q.dev.sqtt_wgp = wgp
        with Context(SQTT_SIMD_SEL=2):
          q.sqtt_start(UOp.const(0))
        masks = [c.kwargs for c in q.wreg.call_args_list if c.args[0] == gc.regSQ_THREAD_TRACE_MASK]
        self.assertEqual([(d["wgp_sel"], d["sa_sel"], d["simd_sel"]) for d in masks], [(wgp, 0, 2)] * 3)
    # In this fixture, 72 blocks request one CU per SE, plus one on SE0.
    for wgp, limit, blocks, expected in ((0, 1, 1, [1]+[0]*7), (4, 0, 1, []), (4, 1, 1, [0x100]+[0]*7),
                                         (4, 2, 1, [0x100,0,0x100]+[0]*5), (4, 1, 72, [0x102]+[1]*7), (1, 1, 144, [7]+[3]*7)):
      with self.subTest(wgp=wgp, limit=limit, blocks=blocks):
        clear()
        q.dev.sqtt_wgp = wgp
        with Context(SQTT_LIMIT_SE=limit, SQTT_ITRACE_SE_MASK=0b101):
          q.sqtt_setup_exec(SimpleNamespace(libhash=0), SimpleNamespace(global_size=(blocks,1,1)))
        self.assertEqual([c.args for c in q.wreg.call_args_list],
                         [(getattr(gc, f"regCOMPUTE_STATIC_THREAD_MGMT_SE{se}"), mask) for se, mask in enumerate(expected)])

@contextlib.contextmanager
def save_sqtt():
  Device[Device.DEFAULT].synchronize()
  profile_start = len(Compiled.profile_events)
  data = []
  yield data
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT]._at_profile_finalize()
  data[:] = [e for e in Compiled.profile_events[:profile_start] if isinstance(e, ProfileProgramEvent)]+Compiled.profile_events[profile_start:]
  if getenv("PRINT_PKTS"):
    sqtt_kernels = set()
    for event in data:
      if not isinstance(event, ProfileSQTTEvent) or not event.itrace: continue
      print(f"\n=== SE {event.se} ===")
      print_packets(decode(event.blob))
      sqtt_kernels.add(event.kern)
    for event in data:
      if not isinstance(event, ProfileProgramEvent) or event.tag not in sqtt_kernels: continue
      from test.null.test_viz import write_files, run_cli
      with write_files(profile=data) as files:
        out = run_cli(*files, "-s", f"{event.name} SQTT SE:0 PKTS", json_fmt=False)[0]["out"]
      print(out)

def map_sqtt(profile:list) -> list[dict]:
  load_amd_counters(data:=VizData(), profile)
  return [r for r in data.ctxs if r["name"].startswith("SQTT")]

def custom_asm_cdna(A:UOp):
  import tinygrad.runtime.autogen.amd.cdna.ins as cdna
  WAVE_SIZE = 64
  insts = [
    cdna.s_barrier(),
    cdna.s_getreg_b32(s[0], cdna.HWREG.HW_REG_HW_ID.value | (4 << 6) | (1 << 11)),

    cdna.s_cmp_eq_u32(s[0], 0),
    cdna.s_cbranch_scc1(16),

    cdna.s_cmp_eq_u32(s[0], 1),
    cdna.s_cbranch_scc1(9),

    cdna.s_cmp_eq_u32(s[0], 2),
    cdna.s_cbranch_scc1(3),

    # SIMD 3
    cdna.v_mov_b32_e32(v[0], 3),
    cdna.s_nop(3),
    cdna.s_endpgm(),

    # SIMD 2
    cdna.v_mov_b32_e32(v[0], 2),
    cdna.s_nop(2),
    cdna.s_nop(2),
    cdna.s_endpgm(),

    # SIMD 1
    cdna.v_mov_b32_e32(v[0], 1),
    cdna.s_nop(1),
    cdna.s_nop(1),
    cdna.s_nop(1),
    cdna.s_endpgm(),

    # SIMD 0
    cdna.v_mov_b32_e32(v[0], 0),
    cdna.s_nop(0),
    cdna.s_nop(0),
    cdna.s_nop(0),
    cdna.s_nop(0),
    cdna.s_endpgm(),
  ]
  return custom_asm(A, insts, WAVE_SIZE*4, 96*1024)

def custom_asm_rdna(A:UOp):
  import tinygrad.runtime.autogen.amd.rdna3.ins as rdna3
  WAVE_SIZE = 32
  insts = [rdna3.s_nop(0), rdna3.s_mov_b32(s[0], 10)]
  return custom_asm(A, insts+[rdna3.s_endpgm()], WAVE_SIZE*2)

def custom_asm(A, insts, num_threads, lds_size=0) -> UOp:
  lds = UOp.placeholder((lds_size,), dtypes.uint8, addrspace=AddrSpace.LOCAL) if lds_size else None
  return UOp(Ops.PROGRAM, src=(UOp.sink(A, lds, UOp.special(num_threads, "lidx0"), arg=KernelInfo("asm")), \
      UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS,arg=(x,dtypes.void)) for x in insts]))))

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestSQTTProfiler(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled: raise unittest.SkipTest("device must be in SQTT profiling mode")
    cls.arch = Device[Device.DEFAULT].arch

  def test_simple(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as data:
      linear = t.schedule_linear()
      run_linear(linear)
    fn_name = to_program(linear.src[0].src[0], renderer=Device[Device.DEFAULT].renderer).src[0].arg.function_name
    sqtt = map_sqtt(data)
    self.assertEqual(len(sqtt), 1)
    self.assertEqual(sqtt[0]["name"], f"SQTT {fn_name}")

  def test_asm(self):
    t = Tensor.empty(1)
    with save_sqtt():
      t.custom_kernel(fxn=custom_asm_cdna if self.arch == "gfx950" else custom_asm_rdna)[0].realize()

  def test_setprio(self):
    if self.arch == "gfx950":
      from tinygrad.runtime.autogen.amd.cdna import ins as isa
      hw_id, wave_size, add = isa.HWREG.HW_REG_HW_ID.value, 64, isa.s_add_u32
      barrier = [isa.s_barrier()]
    elif self.arch.startswith("gfx12"):
      from tinygrad.runtime.autogen.amd.rdna4 import ins as isa
      hw_id, wave_size, add = isa.HWREG.HW_REG_WAVE_HW_ID1.value, 32, isa.s_add_co_u32
      barrier = [isa.s_barrier_signal(ssrc0=-1), isa.s_barrier_wait(simm16=-1)]
    else: self.skipTest("tested on CDNA4 and RDNA4")
    def setprio_kernel(A, priority=0):
      insts = [
        isa.s_getreg_b32(s[0], hw_id),
        isa.s_mov_b32(s[1], 0),
        isa.s_setprio(0),
        isa.s_cmp_eq_u32(s[0], 0),
        isa.s_cbranch_scc1(1),
        isa.s_setprio(priority),
        *barrier,
      ]
      # eight waves contend for scalar issue slots
      insts += [add(s[1], s[1], 1) for _ in range(64)]
      insts += [isa.s_setprio(0), *barrier, isa.s_endpgm()]
      return custom_asm(A, insts, wave_size*8, (96 if self.arch == "gfx950" else 64)*1024)

    with Context(SQTT_LIMIT_SE=1), save_sqtt():
      Tensor.empty(1).custom_kernel(fxn=functools.partial(setprio_kernel, priority=3))[0].realize()
      Tensor.empty(1).custom_kernel(fxn=functools.partial(setprio_kernel, priority=0))[0].realize()

  def test_multiple_runs(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as data:
      linear = t.schedule_linear()
      for _ in range(N:=3): run_linear(linear)
    fn_name = to_program(linear.src[0].src[0], renderer=Device[Device.DEFAULT].renderer).src[0].arg.function_name
    sqtt = map_sqtt(data)
    self.assertEqual(len(sqtt), N)
    for i in range(1, N):
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name} n{i+1}")

  def test_multiple_kernels(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    linear = t.schedule_linear()
    with save_sqtt() as data:
      run_linear(linear)
    sqtt = map_sqtt(data)
    self.assertEqual(len(sqtt), len(linear.src))
    for i,call in enumerate(linear.src):
      fn_name = to_program(call.src[0], renderer=Device[Device.DEFAULT].renderer).src[0].arg.function_name
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name}")

  def test_multiple_kernels_lower(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    linear = t.schedule_linear()
    with save_sqtt() as data:
      run_linear(linear)
    sqtt = map_sqtt(data)
    self.assertEqual(len(sqtt), len(linear.src))
    for i,call in enumerate(linear.src):
      fn_name = to_program(call.src[0], renderer=Device[Device.DEFAULT].renderer).src[0].arg.function_name
      self.assertEqual(sqtt[i]["name"], f"SQTT {fn_name}")

  def test_jit(self):
    @TinyJit
    def f(a): return a + 1
    t = Tensor.empty(1)
    with save_sqtt() as data:
      for _ in range(N:=5):
        f(t).realize()
    sqtt = map_sqtt(data)
    self.assertEqual(len(sqtt), N)
    kernel_name = sqtt[0]["name"]
    for i,e in enumerate(sqtt[1:], start=1): self.assertEqual(e["name"], f"{kernel_name} n{i+1}")

  def test_jit_graph(self, kernel_count=3*(5 if is_hcq2_device() else 1)): # hcq2 traces the graphed kernels too
    @TinyJit
    def f(a): return ((a + 1).contiguous() + 2).contiguous().sum()
    t = Tensor.empty(32)
    with save_sqtt() as data:
      for _ in range(5):
        f(t).realize()
    sqtt = map_sqtt(data)
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
