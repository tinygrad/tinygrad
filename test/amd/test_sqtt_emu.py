"""Tests for cycle-accurate RDNA3 SQTT emulation and packet encoding."""
import pickle, unittest
from pathlib import Path
from tinygrad import Device, Tensor
from tinygrad.device import ProfileProgramEvent
from tinygrad.runtime.ops_amd import ProfileSQTTEvent
from tinygrad.renderer.amd.dsl import s, v
from tinygrad.renderer.amd.sqtt import (decode, encode, map_insts, LAYOUT_HEADER, WAVESTART, WAVEEND, VALUINST, INST,
                                        IMMEDIATE_MASK, ALUEXEC, VMEMEXEC, WAVERDY, TS_DELTA_OR_MARK, InstOp, AluSrc, MemSrc)
from tinygrad.runtime.autogen.amd.rdna3 import ins as isa
from tinygrad.viz.serve import sqtt_timeline
from test.amd.test_asm_kernel import custom_lds_sync
from test.amd.test_sqtt_profiler import save_sqtt, custom_asm

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "extra/sqtt/examples/gfx1100"

class TestSQTTEncoder(unittest.TestCase):
  def test_roundtrip(self):
    pkts = [
      (0, LAYOUT_HEADER, {"layout": 3, "simd": 0, "group": 0, "sel_a": 6, "sel_b": 0, "flag4": 0}),
      (100, TS_DELTA_OR_MARK, {"rt": 1, "pl": 0, "delta": 10_000_000_000}),
      (1000, WAVESTART, {"sa": 0, "simd": 0, "wgp": 0, "wave": 0, "id7": 1}),
      (1010, INST, {"wave": 0, "op": InstOp.SALU}),
      (1011, VALUINST, {"wave": 0, "flag": 0}),
      (1014, ALUEXEC, {"src": AluSrc.SALU}),
      (1019, ALUEXEC, {"src": AluSrc.VALU}),
      (1030, IMMEDIATE_MASK, {"mask": 0b11}),
      (1040, WAVERDY, {"mask": 0b10}),
      (1050, VMEMEXEC, {"src": MemSrc.LDS}),
      (1060, WAVEEND, {"sa": 0, "simd": 0, "wgp": 0, "wave": 0}),
    ]
    blob = encode(pkts)
    self.assertEqual(len(blob) % 32, 0)
    decoded = [p for p in decode(blob) if not type(p).__name__.startswith(("NOP", "TS_DELTA_SHORT")) and
               not (isinstance(p, TS_DELTA_OR_MARK) and not p.is_marker)]
    self.assertEqual(len(decoded), len(pkts))
    for dec, (t, cls, _) in zip(decoded, pkts):
      self.assertIsInstance(dec, cls)
      self.assertEqual(dec._time, t)

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestSQTTEmuRDNA3(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled or not Device[Device.DEFAULT].arch.startswith("gfx11"):
      raise unittest.SkipTest("requires RDNA3 in SQTT profiling mode")

  def _assert_timeline_invariants(self, blob: bytes, lib: bytes):
    pkts = list(decode(blob))
    self.assertIsInstance(pkts[0], LAYOUT_HEADER)
    times = [p._time for p in pkts]
    self.assertEqual(times, sorted(times))
    timeline = list(sqtt_timeline(blob, lib, "gfx1100"))
    freqs = [x.key for x in timeline if type(x).__name__ == "ProfilePointEvent" and x.name == "freq_hz"]
    self.assertTrue(len(freqs) > 0 and all(f > 0 for f in freqs))
    revs = [x for x in timeline if type(x).__name__ == "ProfileRangeEvent"]
    non_exec_ops = {"IMMEDIATE", "IMMEDIATE_MASK", "JUMP", "JUMP_NO", "MESSAGE", "BARRIER", "BARRIER_SIGNAL",
                    "WAVEEND", "WAVEEND_RDNA4", "WAVERDY"}
    insts_c = sum(1 for x in revs if "WAVE" in x.device and x.name.display_name not in non_exec_ops and
                  not x.name.display_name.startswith("OTHER_"))
    execs_c = sum(1 for x in revs if "EXEC" in x.device and "ALT" not in x.name.display_name)
    self.assertEqual(execs_c, insts_c)
    for b in (x for x in revs if x.name.display_name == "BARRIER"):
      self.assertGreater(b.en - b.st, 1)

  def test_valu_raw_and_loop(self):
    insts = [
      isa.s_mov_b32(s[1], 6),
      isa.v_mov_b32_e32(v[1], 1),
      isa.v_add_f32_e32(v[1], v[1], v[1]),
      isa.v_add_f32_e32(v[1], v[1], v[1]),
      isa.s_add_i32(s[1], s[1], -1),
      isa.s_cmp_eq_i32(s[1], 0),
      isa.s_cbranch_scc0(-5 & 0xFFFF),
      isa.s_endpgm(),
    ]
    with save_sqtt() as data:
      Tensor.empty(1).custom_kernel(fxn=lambda A: custom_asm(A, insts, 32))[0].realize()
    prgs = {e.tag: e for e in data if isinstance(e, ProfileProgramEvent)}
    e = [e for e in data if isinstance(e, ProfileSQTTEvent) and e.itrace][0]
    self._assert_timeline_invariants(e.blob, prgs[e.kern].lib)
    mapped = list(map_insts(e.blob, prgs[e.kern].lib, "gfx1100"))
    jumps = [p.op for p, _ in mapped if isinstance(p, INST) and p.op in (InstOp.JUMP, InstOp.JUMP_NO)]
    self.assertEqual(jumps, [InstOp.JUMP] * 5 + [InstOp.JUMP_NO])
    # verify 5-cycle RAW latency between dependent v_add_f32 instructions
    vadds = [p._time for p, info in mapped if info is not None and info.inst.op_name == "V_ADD_F32_E32"]
    self.assertEqual(len(vadds), 12)
    self.assertEqual(vadds[1] - vadds[0], 5)

  def test_lds_sync_matches_hw(self):
    a = Tensor.zeros(128, dtype="uint32").contiguous().realize()
    with save_sqtt() as data:
      Tensor.custom_kernel(a, fxn=lambda A: custom_lds_sync(A, "rdna3"))[0].realize()
    prgs = {e.tag: e for e in data if isinstance(e, ProfileProgramEvent)}
    e = [e for e in data if isinstance(e, ProfileSQTTEvent) and e.itrace][0]
    self._assert_timeline_invariants(e.blob, prgs[e.kern].lib)

    # Compare SRAM section (from first v_lshlrev_b32 after SMEM wait to last v_lshlrev_b32 before global_store)
    # against real gfx1100 hardware trace in extra/sqtt/examples/gfx1100/profile_sync_run_0.pkl
    def sram_duration(blob: bytes, lib: bytes) -> int:
      w0_times = [(p._time, info.inst.op_name) for p, info in map_insts(blob, lib, "gfx1100") if info is not None and info.wave == 0]
      t_start = next(t for t, op in w0_times if op == "V_LSHLREV_B32_E32")
      t_end = [t for t, op in w0_times if op == "V_LSHLREV_B32_E32"][-1]
      return t_end - t_start

    emu_cycles = sram_duration(e.blob, prgs[e.kern].lib)
    with open(EXAMPLES_DIR / "profile_sync_run_0.pkl", "rb") as f:
      hw_data = pickle.load(f)
    hw_prgs = {x.tag: x for x in hw_data if type(x).__name__ == "ProfileProgramEvent"}
    hw_sqtt = next(x for x in hw_data if type(x).__name__ == "ProfileSQTTEvent" and x.itrace and
                   any(isinstance(p, WAVESTART) for p in decode(x.blob)))
    hw_cycles = sram_duration(hw_sqtt.blob, hw_prgs[hw_sqtt.kern].lib)
    # Emulated SRAM kernel cycle count should closely match real gfx1100 hardware (within 5%)
    self.assertLessEqual(abs(emu_cycles - hw_cycles), max(6, hw_cycles * 0.05),
                         f"SRAM cycle mismatch: emu={emu_cycles} vs hw={hw_cycles}")

  def test_tfu_salu_and_valub2(self):
    insts = [
      isa.s_mov_b32(s[2], 10),
      isa.s_add_u32(s[2], s[2], 5),
      isa.s_addc_u32(s[3], s[2], 1),
      isa.v_mov_b32_e32(v[1], 1.0),
      isa.v_rcp_f32_e32(v[2], v[1]),
      isa.v_lshlrev_b64(v[4:5], 2, v[4:5]),
      isa.s_endpgm(),
    ]
    with save_sqtt() as data:
      Tensor.empty(1).custom_kernel(fxn=lambda A: custom_asm(A, insts, 32))[0].realize()
    prgs = {e.tag: e for e in data if isinstance(e, ProfileProgramEvent)}
    e = [e for e in data if isinstance(e, ProfileSQTTEvent) and e.itrace][0]
    self._assert_timeline_invariants(e.blob, prgs[e.kern].lib)
    ops = [p.op for p, _ in map_insts(e.blob, prgs[e.kern].lib, "gfx1100") if isinstance(p, INST)]
    self.assertIn(InstOp.VALUT_4, ops)
    self.assertIn(InstOp.VALUB_2, ops)
    self.assertIn(InstOp.SALU, ops)

  def test_hw_traces_sram_accuracy(self):
    from tinygrad.viz.serve import amd_decode, get_cfg, get_elf_section
    from test.mockgpu.amd.sqtt_emu import simulate_rdna3_sqtt
    for pkl_name, max_mae in [("profile_plus_run_0.pkl", 0.5), ("profile_sync_run_0.pkl", 1.0), ("profile_gemm_run_0.pkl", 1.5)]:
      with self.subTest(trace=pkl_name):
        with open(EXAMPLES_DIR / pkl_name, "rb") as f: hw_data = pickle.load(f)
        prgs = {x.tag: x for x in hw_data if type(x).__name__ == "ProfileProgramEvent"}
        e = next(x for x in hw_data if type(x).__name__ == "ProfileSQTTEvent" and x.itrace and
                 any(isinstance(p, WAVESTART) for p in decode(x.blob)))
        prg = prgs[e.kern]
        text = get_elf_section(prg.lib, ".text")
        code, base_pc = text.content, text.header.sh_addr
        cfg = get_cfg(amd_decode(code, "rdna3"))
        n_waves = len([p for p in decode(e.blob) if isinstance(p, WAVESTART) and p.sa == 0 and p.wgp == 0])
        hw_w0 = [(p._time, info.pc - base_pc) for p, info in map_insts(e.blob, prg.lib, "gfx1100") if info is not None and info.wave == 0]
        block_starts = set(cfg["blocks"].keys())
        btrace: list[int] = []
        for _, rel_pc in hw_w0:
          if rel_pc in block_starts and (not btrace or btrace[-1] != rel_pc or rel_pc in cfg["paths"].get(btrace[-1], {})):
            btrace.append(rel_pc)
        blobs = simulate_rdna3_sqtt(code, 1, 1, 1, max(1, n_waves) * 32, 1, 1, n_se=1, itrace_se_mask=1, simd_sel=0,
                                    block_traces={(0, lw): btrace for lw in range(max(1, n_waves))})
        emu_w0 = [(p._time, info.pc - base_pc) for p, info in map_insts(blobs[0], prg.lib, "gfx1100") if info is not None and info.wave == 0]
        self.assertEqual(len(emu_w0), len(hw_w0))
        errs = [abs((hw_w0[i][0] - hw_w0[i-1][0]) - (emu_w0[i][0] - emu_w0[i-1][0]))
                for i in range(1, len(hw_w0)) if hw_w0[i][0] - hw_w0[i-1][0] < 150]
        mae = sum(errs) / len(errs)
        self.assertLessEqual(mae, max_mae, f"{pkl_name}: SRAM MAE={mae:.2f} > {max_mae}")

if __name__ == "__main__":
  unittest.main()
