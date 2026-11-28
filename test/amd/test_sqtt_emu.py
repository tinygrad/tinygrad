# Tests for RDNA3 mockgpu SQTT packet encoding, CU microarchitecture model, and end-to-end SQTT profiling.
import itertools, pickle, unittest
from pathlib import Path
from tinygrad import Device, Tensor
from tinygrad.device import ProfileProgramEvent
from tinygrad.helpers import DEV
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from tinygrad.renderer.amd.dsl import s, v, NULL
from tinygrad.renderer.amd.sqtt import decode, map_insts, PACKET_TYPES_RDNA3, LAYOUT_HEADER, NOP, WAVESTART, INST, IMMEDIATE, InstOp
from tinygrad.runtime.autogen.amd.rdna3 import ins as isa
from tinygrad.viz.serve import sqtt_timeline
from test.mockgpu.amd.sqtt import make_packet, encode_packets
from test.mockgpu.amd.sqtt_sim import MicroOp, RDNA3CU
from test.amd.test_asm_kernel import custom_lds_sync
from test.amd.test_sqtt_profiler import save_sqtt, custom_asm

EXAMPLES_DIR = Path(__file__).parents[2] / "extra/sqtt/examples/gfx1100"

def _run_cu(*wave_insts, simds=None, ds_addrs=tuple(range(0, 128, 4))):
  waves = []
  for insts in wave_insts:
    pcs = itertools.accumulate([0] + [inst.size() for inst in insts])
    waves.append([MicroOp(pc, inst, 0xffffffff, list(ds_addrs) if isinstance(inst, isa.DS) else []) for pc, inst in zip(pcs, insts)])
  cu = RDNA3CU(waves, simds or [0] * len(waves))
  cu.run()
  return [([x.t for x in w.ops], [x.stamp for x in w.ops]) for w in cu.waves]

def _deltas(xs: list[int]) -> list[int]: return [b - a for a, b in zip(xs, xs[1:])]

class TestSQTTEncoder(unittest.TestCase):
  def test_packet_tail_alignment(self):
    for cls in set(PACKET_TYPES_RDNA3.values()) - {LAYOUT_HEADER, NOP}:
      for pkts in ([make_packet(LAYOUT_HEADER, layout=3), make_packet(cls)],
                   [make_packet(LAYOUT_HEADER, layout=3), make_packet(IMMEDIATE), make_packet(cls)]):
        with self.subTest(cls=cls.__name__, nibbles=sum(p._size_nibbles for p in pkts)):
          self.assertEqual([type(p) for p in decode(encode_packets(pkts))], [type(p) for p in pkts])

  def test_hw_examples_roundtrip(self):
    for pkl_path in sorted(EXAMPLES_DIR.glob("*_run_0.pkl")):
      data = pickle.loads(pkl_path.read_bytes())
      for i, ev in enumerate(e for e in data if type(e).__name__ == "ProfileSQTTEvent"):
        with self.subTest(example=pkl_path.stem, se=i):
          pkts = list(decode(ev.blob))
          self.assertEqual([(type(p), p._raw) for p in decode(encode_packets(pkts))], [(type(p), p._raw) for p in pkts])

class TestRDNA3CUModel(unittest.TestCase):
  def test_valu_and_vector_queue(self):
    warm = [isa.v_add_f32_e32(v[30], v[20], v[21]), isa.s_nop(15)]
    (issue, stamp), = _run_cu(warm + [isa.v_add_f32_e32(v[k], v[20], v[21]) for k in range(8)])
    self.assertEqual(_deltas(issue[2:]), [1] * 7)
    self.assertEqual([st - it for it, st in zip(issue[2:], stamp[2:])], [7] * 8)

    (issue_q, stamp_q), = _run_cu(warm + [isa.v_fma_f32(v[0], v[0], v[1], v[2])] * 20)
    self.assertEqual(set(_deltas(stamp_q[2:])), {5})
    self.assertEqual(_deltas(issue_q[2:16]), [1] * 13)
    self.assertEqual(set(_deltas(issue_q[16:])), {5})

  def test_wmma_and_tfu(self):
    warm = [isa.v_add_f32_e32(v[30], v[20], v[21]), isa.s_nop(15)]
    (issue, _), = _run_cu(warm + [isa.v_wmma_f32_16x16x16_f16(v[0:7], v[8:15], v[16:23], v[0:7])] +
                          [isa.v_add_f32_e32(v[30 + k], v[20], v[21]) for k in range(10)])
    self.assertEqual(_deltas(issue[2:10]), [1] * 7)
    self.assertGreater(issue[10] - issue[9], 30)

    (tfu_issue, _), = _run_cu(warm + [isa.v_rcp_f32_e32(v[0], v[1]), isa.v_rcp_f32_e32(v[2], v[3])])
    self.assertEqual(tfu_issue[3] - tfu_issue[2], 4)

  def test_lds_port_queue_and_bank_conflicts(self):
    (a, a_st), (b, b_st) = _run_cu(*[[isa.s_barrier(), isa.ds_store_b32(addr=v[1], data0=v[2])]] * 2, simds=[0, 1])
    self.assertEqual(a[1], b[1])
    self.assertEqual(a_st[1] - b_st[1], 2)

    (q_issue, _), = _run_cu([isa.ds_store_b32(addr=v[1], data0=v[2])] * 30)
    self.assertEqual(_deltas(q_issue), [1] * 15 + [2] * 14)

    # All 32 lanes hitting bank 0 (stride 128B = 32 dwords) adds 31 bank-conflict cycles
    (no_conf, nc_st), = _run_cu([isa.ds_load_b32(vdst=v[0], addr=v[1]), isa.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)],
                                ds_addrs=tuple(range(0, 128, 4)))
    (conf32, c_st), = _run_cu([isa.ds_load_b32(vdst=v[0], addr=v[1]), isa.s_waitcnt_lgkmcnt(sdst=NULL, simm16=0)],
                              ds_addrs=tuple(range(0, 4096, 128)))
    self.assertEqual((conf32[1] - c_st[0]) - (no_conf[1] - nc_st[0]), 31)

  def test_icache_and_barrier(self):
    (issue, _), = _run_cu([isa.v_add_f32_e32(v[0], v[20], v[21])] * 128)
    self.assertEqual(issue[64] - issue[15], 625)

    (early, _), (late, _) = _run_cu([isa.s_barrier(), isa.v_add_f32_e32(v[0], v[20], v[21])],
                                    [isa.s_nop(15), isa.s_nop(15), isa.s_barrier(), isa.v_add_f32_e32(v[0], v[20], v[21])], simds=[0, 1])
    self.assertEqual(early[1], late[3])
    self.assertEqual(early[1] - late[2], 12)

@unittest.skipUnless(Device.DEFAULT == "AMD" and DEV.interface.startswith("MOCK"), "only runs on AMD mockgpu")
class TestSQTTEmuProfiler(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not Device["AMD"].sqtt_enabled or not Device["AMD"].arch.startswith("gfx11"):
      raise unittest.SkipTest("requires RDNA3 in SQTT profiling mode")

  def _check_timeline(self, blob: bytes, lib: bytes):
    pkts = list(decode(blob))
    self.assertIsInstance(pkts[0], LAYOUT_HEADER)
    times = [p._time for p in pkts]
    self.assertEqual(times, sorted(times))
    tl = list(sqtt_timeline(blob, lib, "gfx1100"))
    freqs = [x.key for x in tl if type(x).__name__ == "ProfilePointEvent" and x.name == "freq_hz"]
    self.assertTrue(len(freqs) > 0 and all(f > 0 for f in freqs))
    revs = [x for x in tl if type(x).__name__ == "ProfileRangeEvent"]
    non_exec = {"IMMEDIATE", "IMMEDIATE_MASK", "JUMP", "JUMP_NO", "CALL", "MESSAGE", "BARRIER", "WAVEEND", "WAVERDY"}
    insts_c = sum(1 for x in revs if "WAVE" in x.device and x.name.display_name not in non_exec and not x.name.display_name.startswith("OTHER_"))
    execs_c = sum(1 for x in revs if "EXEC" in x.device and "ALT" not in x.name.display_name)
    self.assertEqual(execs_c, insts_c)

  def test_loop_exec_mask_and_ops(self):
    insts = [
      isa.s_mov_b32(s[1], 4),
      isa.v_mov_b32_e32(v[1], 1.0),
      isa.v_rcp_f32_e32(v[2], v[1]),
      isa.v_mul_lo_u32(v[3], v[0], 3),
      isa.v_lshlrev_b64(v[4:5], 2, v[4:5]),
      isa.s_and_saveexec_b32(s[2], 0),
      isa.v_add_f32_e32(v[1], v[1], v[2]),  # EXEC == 0 -> retires as IMMEDIATE
      isa.s_mov_b32(isa.EXEC_LO, s[2]),
      isa.s_add_i32(s[1], s[1], -1),
      isa.s_cmp_eq_i32(s[1], 0),
      isa.s_cbranch_scc0(-6 & 0xFFFF),
      isa.s_endpgm(),
    ]
    with save_sqtt() as data:
      Tensor.empty(1).custom_kernel(fxn=lambda A: custom_asm(A, insts, 32))[0].realize()
    prgs = {e.tag: e for e in data if isinstance(e, ProfileProgramEvent)}
    ev = next(e for e in data if isinstance(e, ProfileSQTTEvent) and e.se == 0)
    self._check_timeline(ev.blob, prgs[ev.kern].lib)
    mapped = list(map_insts(ev.blob, prgs[ev.kern].lib, "gfx1100"))
    ops = [p.op for p, _ in mapped if isinstance(p, INST)]
    self.assertIn(InstOp.VALUT_4, ops)
    self.assertIn(InstOp.VALUB_4, ops)
    self.assertIn(InstOp.VALUB_2, ops)
    self.assertIn(InstOp.SALU_WR_EXEC, ops)
    masked_vadds = [type(p) for p, info in mapped if info is not None and info.inst.op_name == "V_ADD_F32_E32"]
    self.assertEqual(masked_vadds, [IMMEDIATE] * 4)

  def test_lds_sync_matches_hw(self):
    a = Tensor.zeros(128, dtype="uint32").contiguous().realize()
    with save_sqtt() as data:
      Tensor.custom_kernel(a, fxn=lambda A: custom_lds_sync(A, "rdna3"))[0].realize()
    prgs = {e.tag: e for e in data if isinstance(e, ProfileProgramEvent)}
    ev = next(e for e in data if isinstance(e, ProfileSQTTEvent) and e.se == 0)
    self._check_timeline(ev.blob, prgs[ev.kern].lib)

    def sram_span(blob: bytes, lib: bytes) -> int:
      w0 = [p._time for p, info in map_insts(blob, lib, "gfx1100") if info and info.wave == 0 and info.inst.op_name == "V_LSHLREV_B32_E32"]
      return w0[-1] - w0[0]

    emu_span = sram_span(ev.blob, prgs[ev.kern].lib)
    hw_data = pickle.loads((EXAMPLES_DIR / "profile_sync_run_0.pkl").read_bytes())
    hw_prgs = {x.tag: x for x in hw_data if type(x).__name__ == "ProfileProgramEvent"}
    hw_ev = next(x for x in hw_data if type(x).__name__ == "ProfileSQTTEvent" and x.itrace and any(isinstance(p, WAVESTART) for p in decode(x.blob)))
    hw_span = sram_span(hw_ev.blob, hw_prgs[hw_ev.kern].lib)
    self.assertLessEqual(abs(emu_span - hw_span), 4, f"SRAM span mismatch: emu={emu_span} vs hw={hw_span}")

if __name__ == "__main__":
  unittest.main()
