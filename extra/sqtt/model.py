import os
os.environ["VIZ"] = "-2"
import unittest, functools
from extra.gemm.amd_asm_matmul import Kernel
from tinygrad.runtime.autogen.amd.rdna3.ins import *
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad import Tensor, Device
from tinygrad.helpers import DEBUG, getenv
from tinygrad.renderer.amd.sqtt import *

def asm_fxn(*args:tuple[UOp, ...], k:Kernel, lx=None, gx=None, name:str="asm_fxn") -> UOp:
  if lx is None: lx = getenv("LX", 1)
  if gx is None: gx = getenv("GX", 1)
  lidx = UOp.special(lx, "lidx0")
  gidx = UOp.special(gx, "gidx0")
  sink = UOp.sink(*[t.base for t in args], lidx, gidx, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=x) for x in k.finalize()]))))

skip = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "LAYOUT_HEADER", "SNAPSHOT", "TS_DELTA_S5_W2",
        "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG", "EVENT"}
def load_sqtt():
  sqtt, prg = None, None
  for e in Device[Device.DEFAULT].profile_events:
    if type(e).__name__ == "ProfileSQTTEvent" and e.se == 1: sqtt = e
    if type(e).__name__ == "ProfileProgramEvent": prg = e
  target = f"gfx{Device[Device.DEFAULT].device_props()['gfx_target_version']//1000}"
  ret = []
  for p in map_insts(sqtt.blob, prg.lib, target):
    if type(p[0]).__name__ in skip: continue
    if DEBUG >= 2: print_packets([p])
    ret.append(p)
  return ret

class TestSQTTModel(unittest.TestCase):
  def setUp(self):
    self.arch = getattr(Device[Device.DEFAULT].renderer, "arch", "")
    if not self.arch.startswith("gfx11"): self.skipTest("only rdna3")
    Device[Device.DEFAULT].profile_events.clear()

  def test_nop(self):
    k = Kernel(self.arch)
    k.emit(s_nop(0))
    k.emit(s_endpgm())
    Tensor.empty(1).custom_kernel(fxn=functools.partial(asm_fxn, k=k))[0].realize()
    mapped = load_sqtt()
    assert len(mapped) == 2
    assert {type(p) for p,_ in mapped} == {IMMEDIATE, WAVEEND}

  def test_tiny(self):
    k = Kernel(self.arch)
    for _ in range(10):
      k.emit(s_load_b128(s[4:7], s[0:1]))
      k.emit(v_mov_b32_e32(v[4], 0))
    k.emit(s_endpgm())
    Tensor.empty(1).custom_kernel(fxn=functools.partial(asm_fxn, k=k))[0].realize()
    mapped = load_sqtt()

  def test_salu(self):
    k = Kernel(self.arch)
    k.emit(s_add_u32(s[0], s[1], s[0]))
    k.emit(s_endpgm())
    Tensor.empty(1).custom_kernel(fxn=functools.partial(asm_fxn, k=k))[0].realize()
    mapped = load_sqtt()

  def test_valu(self):
    k = Kernel(self.arch)
    k.emit(v_add_f32_e32(v[0], v[1], v[0]))
    k.emit(s_endpgm())
    Tensor.empty(1).custom_kernel(fxn=functools.partial(asm_fxn, k=k))[0].realize()
    mapped = load_sqtt()

  def test_valu_salu(self):
    k = Kernel(self.arch)
    k.emit(v_add_f32_e32(v[0], v[1], v[0]))
    k.emit(s_add_u32(s[0], s[1], s[0]))
    k.emit(v_add_f32_e32(v[2], v[3], v[4]))
    k.emit(s_sub_u32(s[10], s[1], s[10]))
    k.emit(s_endpgm())
    Tensor.empty(1).custom_kernel(fxn=functools.partial(asm_fxn, k=k))[0].realize()
    mapped = load_sqtt()

    # count issued instructions
    valu_issued = sum(1 for p, _ in mapped if isinstance(p, VALUINST))
    salu_issued = sum(1 for p, _ in mapped if isinstance(p, INST) and p.op == InstOp.SALU)

    # count ALUEXEC by source type (VALU_SALU counts as both VALU and SALU)
    valu_exec = sum(1 for p, _ in mapped if isinstance(p, ALUEXEC) and p.src in {AluSrc.VALU, AluSrc.VALU_SALU})
    salu_exec = sum(1 for p, _ in mapped if isinstance(p, ALUEXEC) and p.src in {AluSrc.SALU, AluSrc.VALU_SALU})

    # verify all issued instructions have corresponding ALUEXEC
    assert valu_exec == valu_issued, f"VALU mismatch: {valu_exec} exec vs {valu_issued} issued"
    assert salu_exec == salu_issued, f"SALU mismatch: {salu_exec} exec vs {salu_issued} issued"

  def test_delay_alu(self):
    """s_delay_alu inserts cycles between dependent VALU instructions."""
    k = Kernel(self.arch)
    k.emit(v_mov_b32_e32(v[3], v[0]))           # A
    k.emit(v_lshlrev_b32_e32(v[30], 1, v[31]))  # B
    k.emit(v_lshlrev_b32_e32(v[24], 1, v[25]))  # C
    k.emit(s_delay_alu(0x00A3))                 # instID0=3 (wait 3 VALU back), skip=2, instID1=1
    k.emit(v_add_f32_e32(v[0], v[1], v[3]))     # D (depends on A, 3 back)
    k.emit(v_sub_f32_e32(v[11], v[9], v[9]))    # E
    k.emit(v_mul_f32_e32(v[10], v[13], v[11]))  # F (depends on E, 1 back)
    k.emit(s_endpgm())
    Tensor.empty(1).custom_kernel(fxn=functools.partial(asm_fxn, k=k))[0].realize()
    mapped = load_sqtt()

    # extract VALUINST issue times
    issues = [(p._time, info.inst if info else None) for p, info in mapped if isinstance(p, VALUINST)]
    base = issues[0][0]
    gaps = [issues[i][0] - issues[i-1][0] for i in range(1, len(issues))]

    # with s_delay_alu: gaps before D (index 3) and F (index 5) should be > 1
    assert gaps[2] > 1, f"expected delay before D, got gap={gaps[2]}"
    assert gaps[4] > 1, f"expected delay before F, got gap={gaps[4]}"

  def test_no_delay_alu(self):
    k = Kernel(self.arch)
    k.emit(v_mov_b32_e32(v[3], v[0]))
    k.emit(v_lshlrev_b32_e32(v[30], 1, v[31]))
    k.emit(v_lshlrev_b32_e32(v[24], 1, v[25]))
    # no s_delay_alu here
    k.emit(v_add_f32_e32(v[0], v[1], v[3]))
    k.emit(v_sub_f32_e32(v[11], v[9], v[9]))
    k.emit(v_mul_f32_e32(v[10], v[13], v[11]))
    k.emit(s_endpgm())
    Tensor.empty(1).custom_kernel(fxn=functools.partial(asm_fxn, k=k))[0].realize()
    mapped = load_sqtt()
    # extract VALUINST issue times
    issues = [(p._time, info.inst if info else None) for p, info in mapped if isinstance(p, VALUINST)]
    gaps = [issues[i][0] - issues[i-1][0] for i in range(1, len(issues))]
    # without s_delay_alu: all gaps should be 1 (back-to-back issue)
    assert all(g == 1 for g in gaps), f"expected all gaps=1, got {gaps}"

if __name__ == "__main__":
  unittest.main()
