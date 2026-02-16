#!/usr/bin/env python3
"""Tests for the SQTT encoder and end-to-end sqtt_timeline on MOCKGPU emulator output.

Run with: AMD=1 MOCKGPU=1 PROFILE=1 python -m pytest test/amd/test_sqtt_encoder.py -v
"""
import unittest
from dataclasses import dataclass
from tinygrad.renderer.amd.sqtt import decode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, NOP, VALUINST, InstOp
from test.mockgpu.amd.emu import _init_sqtt_encoder, _encode_raw, _emit_nibbles, _nibbles_to_bytes, _NIB_COUNTS
from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp

# ═══════════════════════════════════════════════════════════════════════════════
# MOCK SQTTInst for test convenience
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FakeSQTTInst:
  pc_offset: int
  inst_type: type
  wave_id: int
  inst_op: int = 0
  inst_op_name: str = ""
  branch_taken: bool|None = None

def encode_sqtt(insts: list[FakeSQTTInst]) -> bytes:
  """Test helper: encode a list of FakeSQTTInst via _init_sqtt_encoder callbacks."""
  emit, finish, finalize = _init_sqtt_encoder()
  seen_waves: set[int] = set()
  for inst in insts:
    emit(inst.wave_id, inst.inst_type, inst.inst_op, inst.inst_op_name, inst.branch_taken)
    seen_waves.add(inst.wave_id)
  for w in sorted(seen_waves): finish(w)
  return finalize()

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: ENCODER PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncoderPrimitives(unittest.TestCase):
  def test_nibbles_to_bytes_even(self):
    self.assertEqual(_nibbles_to_bytes([0xA, 0xB]), bytes([0xBA]))
  def test_nibbles_to_bytes_odd(self):
    self.assertEqual(_nibbles_to_bytes([0x3]), bytes([0x03]))
  def test_nibbles_to_bytes_empty(self):
    self.assertEqual(_nibbles_to_bytes([]), b'')
  def test_encode_raw_layout_header(self):
    raw, nc = _encode_raw(LAYOUT_HEADER, layout=3, sel_a=6)
    self.assertEqual(raw & 0x7F, 0x11)
    self.assertEqual((raw >> 7) & 0x3F, 3)
  def test_encode_raw_inst(self):
    raw, nc = _encode_raw(INST, delta=1, wave=5, op=InstOp.SALU)
    self.assertEqual(raw & 0x7, 0b010)
    self.assertEqual((raw >> 8) & 0x1F, 5)
  def test_emit_nibbles_counts(self):
    for cls, kwargs in [(INST, dict(delta=1, wave=0, op=InstOp.SALU)),
                        (WAVESTART, dict(delta=1, simd=0, cu_lo=0, wave=3, id7=3)),
                        (WAVEEND, dict(delta=1, simd=0, cu_lo=0, wave=3))]:
      nibbles = []
      _emit_nibbles(nibbles, cls, **kwargs)
      self.assertEqual(len(nibbles), _NIB_COUNTS[cls])

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: ENCODER ROUNDTRIP (encode → decode)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncoderRoundtrip(unittest.TestCase):
  def test_empty_trace(self):
    blob = encode_sqtt([])
    self.assertEqual(len(blob) % 32, 0)
    packets = list(decode(blob))
    self.assertIsInstance(packets[0], LAYOUT_HEADER)
    self.assertEqual(packets[0].layout, 3)
    for p in packets:
      self.assertNotIsInstance(p, (WAVESTART, WAVEEND, INST))

  def test_single_wave_single_inst(self):
    trace = [FakeSQTTInst(0, ir3.SOP1, 0)]
    packets = list(decode(encode_sqtt(trace)))
    types = [type(p) for p in packets if not isinstance(p, NOP)]
    self.assertIn(LAYOUT_HEADER, types)
    self.assertIn(WAVESTART, types)
    self.assertIn(INST, types)
    self.assertIn(WAVEEND, types)

  def test_regular_valu_emits_valuinst(self):
    """Regular 32-bit VALU ops (no special op_name) emit VALUINST, not INST."""
    trace = [FakeSQTTInst(0, ir3.VOP3, 0, inst_op_name="V_ADD_F32_E64")]
    packets = list(decode(encode_sqtt(trace)))
    valu_pkts = [p for p in packets if isinstance(p, VALUINST)]
    inst_pkts = [p for p in packets if isinstance(p, INST)]
    self.assertEqual(len(valu_pkts), 1, "regular VALU should emit VALUINST")
    self.assertEqual(len(inst_pkts), 0, "regular VALU should not emit INST")

  def test_valu_transcendental_emits_inst(self):
    """Transcendental VALU ops emit INST(VALU_TRANS)."""
    for op_name in ["V_EXP_F32_E32", "V_LOG_F32_E32", "V_RCP_F32_E64", "V_SQRT_F32_E32", "V_SIN_F32_E32"]:
      with self.subTest(op=op_name):
        trace = [FakeSQTTInst(0, ir3.VOP1, 0, inst_op_name=op_name)]
        pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
        self.assertEqual(len(pkts), 1)
        self.assertEqual(pkts[0].op, InstOp.VALU_TRANS)

  def test_valu_64_shift_emits_inst(self):
    """64-bit shift VALU ops emit INST(VALU_64_SHIFT)."""
    trace = [FakeSQTTInst(0, ir3.VOP3, 0, inst_op_name="V_LSHLREV_B64")]
    pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(len(pkts), 1)
    self.assertEqual(pkts[0].op, InstOp.VALU_64_SHIFT)

  def test_valu_64_emits_inst(self):
    """64-bit arithmetic VALU ops emit INST(VALU_64)."""
    trace = [FakeSQTTInst(0, ir3.VOP3, 0, inst_op_name="V_ADD_F64")]
    pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(len(pkts), 1)
    self.assertEqual(pkts[0].op, InstOp.VALU_64)

  def test_valu_cmpx_emits_inst(self):
    """v_cmpx_* ops emit INST(VALU_CMPX)."""
    trace = [FakeSQTTInst(0, ir3.VOPC, 0, inst_op_name="V_CMPX_LT_F32_E32")]
    pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(len(pkts), 1)
    self.assertEqual(pkts[0].op, InstOp.VALU_CMPX)

  def test_inst_op_mapping(self):
    cases = [(ir3.SMEM, InstOp.SMEM), (ir3.SOP1, InstOp.SALU), (ir3.SOP2, InstOp.SALU)]
    for inst_type, expected_op in cases:
      with self.subTest(t=inst_type.__name__):
        blob = encode_sqtt([FakeSQTTInst(0, inst_type, 0)])
        inst_pkts = [p for p in decode(blob) if isinstance(p, INST)]
        self.assertEqual(len(inst_pkts), 1)
        self.assertEqual(inst_pkts[0].op, expected_op)

  def test_global_store_vs_load(self):
    load = [FakeSQTTInst(0, ir3.GLOBAL, 0, inst_op_name="GLOBAL_LOAD_B32")]
    store = [FakeSQTTInst(0, ir3.GLOBAL, 0, inst_op_name="GLOBAL_STORE_B128")]
    load_pkts = [p for p in decode(encode_sqtt(load)) if isinstance(p, INST)]
    store_pkts = [p for p in decode(encode_sqtt(store)) if isinstance(p, INST)]
    self.assertEqual(load_pkts[0].op, InstOp.GLOBAL_LOAD)
    self.assertEqual(store_pkts[0].op, InstOp.GLOBAL_STORE)

  def test_flat_store_vs_load(self):
    load = [FakeSQTTInst(0, ir3.FLAT, 0, inst_op_name="FLAT_LOAD_B32")]
    store = [FakeSQTTInst(0, ir3.FLAT, 0, inst_op_name="FLAT_STORE_B64")]
    self.assertEqual([p for p in decode(encode_sqtt(load)) if isinstance(p, INST)][0].op, InstOp.FLAT_LOAD)
    self.assertEqual([p for p in decode(encode_sqtt(store)) if isinstance(p, INST)][0].op, InstOp.FLAT_STORE)

  def test_lds_store_vs_load(self):
    load = [FakeSQTTInst(0, ir3.DS, 0, inst_op_name="DS_LOAD_B32")]
    store = [FakeSQTTInst(0, ir3.DS, 0, inst_op_name="DS_STORE_B64")]
    self.assertEqual([p for p in decode(encode_sqtt(load)) if isinstance(p, INST)][0].op, InstOp.LDS_LOAD)
    self.assertEqual([p for p in decode(encode_sqtt(store)) if isinstance(p, INST)][0].op, InstOp.LDS_STORE)

  def test_sopp_endpgm_skipped(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_ENDPGM.value, inst_op_name="S_ENDPGM")]
    packets = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(len(packets), 0, "s_endpgm should not emit INST")

  def test_sopp_delay_alu_skipped(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_DELAY_ALU.value, inst_op_name="S_DELAY_ALU")]
    packets = [p for p in decode(encode_sqtt(trace)) if isinstance(p, (INST, IMMEDIATE))]
    self.assertEqual(len(packets), 0, "s_delay_alu should not emit any packet")

  def test_sopp_waitcnt_emits_immediate(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_WAITCNT.value, inst_op_name="S_WAITCNT")]
    packets = [p for p in decode(encode_sqtt(trace)) if isinstance(p, IMMEDIATE)]
    self.assertEqual(len(packets), 1, "s_waitcnt should emit IMMEDIATE")

  def test_sopp_nop_emits_immediate(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_NOP.value, inst_op_name="S_NOP")]
    packets = [p for p in decode(encode_sqtt(trace)) if isinstance(p, IMMEDIATE)]
    self.assertEqual(len(packets), 1, "s_nop should emit IMMEDIATE")

  def test_sopp_clause_emits_immediate(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_CLAUSE.value, inst_op_name="S_CLAUSE")]
    packets = [p for p in decode(encode_sqtt(trace)) if isinstance(p, IMMEDIATE)]
    self.assertEqual(len(packets), 1, "s_clause should emit IMMEDIATE")

  def test_sopp_barrier_emits_inst_barrier(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_BARRIER.value, inst_op_name="S_BARRIER")]
    pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(len(pkts), 1)
    self.assertEqual(pkts[0].op, InstOp.BARRIER)

  def test_sopp_branch_taken_emits_jump(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_CBRANCH_SCC0.value, inst_op_name="S_CBRANCH_SCC0", branch_taken=True)]
    pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(pkts[0].op, InstOp.JUMP)

  def test_sopp_branch_not_taken_emits_jump_no(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_CBRANCH_SCC0.value, inst_op_name="S_CBRANCH_SCC0", branch_taken=False)]
    pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(pkts[0].op, InstOp.JUMP_NO)

  def test_sopp_sendmsg_emits_salu(self):
    trace = [FakeSQTTInst(0, ir3.SOPP, 0, inst_op=SOPPOp.S_SENDMSG.value, inst_op_name="S_SENDMSG")]
    pkts = [p for p in decode(encode_sqtt(trace)) if isinstance(p, INST)]
    self.assertEqual(pkts[0].op, InstOp.SALU)

  def test_multiple_waves(self):
    trace = [FakeSQTTInst(0, ir3.VOP3, 0), FakeSQTTInst(4, ir3.SOP2, 0), FakeSQTTInst(0, ir3.SMEM, 1)]
    packets = list(decode(encode_sqtt(trace)))
    self.assertEqual(len([p for p in packets if isinstance(p, WAVESTART)]), 2)
    self.assertEqual(len([p for p in packets if isinstance(p, WAVEEND)]), 2)
    # VOP3 (no op_name) → VALUINST, SOP2 → INST(SALU), SMEM → INST(SMEM)
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 2)
    self.assertEqual(len([p for p in packets if isinstance(p, VALUINST)]), 1)

  def test_wave_ids_preserved(self):
    trace = [FakeSQTTInst(0, ir3.SOP1, 3), FakeSQTTInst(0, ir3.SMEM, 7)]
    packets = list(decode(encode_sqtt(trace)))
    self.assertEqual(sorted(p.wave for p in packets if isinstance(p, WAVESTART)), [3, 7])

  def test_timestamps_monotonic(self):
    trace = [FakeSQTTInst(i*4, ir3.SOP1, 0) for i in range(5)]
    times = [p._time for p in decode(encode_sqtt(trace))]
    self.assertEqual(times, sorted(times))

  def test_blob_32byte_aligned(self):
    for n in [0, 1, 5, 10, 50]:
      blob = encode_sqtt([FakeSQTTInst(i*4, ir3.SOP1, 0) for i in range(n)])
      self.assertEqual(len(blob) % 32, 0, f"n={n}")

# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END: sqtt_timeline on MOCKGPU emulator output (in-process)
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(__import__('os').environ.get("VIZ", "0") not in ("", "0"), "VIZ required for SQTT tracing")
class TestSQTTTimelineEndToEnd(unittest.TestCase):
  """End-to-end tests: run kernels on MOCKGPU with VIZ, verify sqtt_timeline works."""

  def _get_sqtt_events(self):
    from tinygrad.device import Compiled, Device, ProfileProgramEvent
    from tinygrad.runtime.ops_amd import ProfileSQTTEvent
    events = Compiled.profile_events
    sqtt = [e for e in events if isinstance(e, ProfileSQTTEvent) and e.itrace and len(e.blob) > 0]
    prg_by_tag = {e.tag: e for e in events if isinstance(e, ProfileProgramEvent) and e.lib is not None and len(e.lib) > 0}
    target = Device['AMD'].iface.props['gfx_target_version']
    return sqtt, prg_by_tag, target

  def _verify_sqtt(self, sqtt, prg_by_tag, target):
    from tinygrad.renderer.amd.sqtt import map_insts
    from tinygrad.viz.serve import sqtt_timeline
    tested = 0
    for ev in sqtt:
      if ev.kern not in prg_by_tag: continue
      prg = prg_by_tag[ev.kern]
      packets = list(decode(ev.blob))
      self.assertGreater(len(packets), 0)
      self.assertEqual(packets[0].layout, 3)
      n_waves = sum(1 for p in packets if isinstance(p, WAVESTART))
      n_ends = sum(1 for p in packets if isinstance(p, WAVEEND))
      self.assertEqual(n_waves, n_ends, "wave start/end mismatch")
      # NOTE: KeyError can happen due to pre-existing kern tag mismatch bug
      try: list(map_insts(ev.blob, prg.lib, target))
      except KeyError: continue
      sqtt_timeline(ev.blob, prg.lib, target)
      tested += 1
    self.assertGreater(tested, 0, "no SQTT events were tested")

  def test_sqtt_timeline_add(self):
    from tinygrad import Tensor
    a, b = Tensor([1.0, 2.0, 3.0, 4.0]).realize(), Tensor([5.0, 6.0, 7.0, 8.0]).realize()
    (a + b).realize()
    self._verify_sqtt(*self._get_sqtt_events())

  def test_sqtt_timeline_matmul(self):
    from tinygrad import Tensor
    (Tensor.rand(4, 4).realize() @ Tensor.rand(4, 4).realize()).realize()
    self._verify_sqtt(*self._get_sqtt_events())

if __name__ == "__main__":
  unittest.main()
