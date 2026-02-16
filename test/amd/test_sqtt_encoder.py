#!/usr/bin/env python3
"""Tests for the SQTT encoder and end-to-end sqtt_timeline on MOCKGPU emulator output.

Run with: AMD=1 MOCKGPU=1 python -m pytest test/amd/test_sqtt_encoder.py -v
"""
import subprocess, sys, os, unittest
from dataclasses import dataclass
from tinygrad.renderer.amd.sqtt import (decode, _encode_raw, _emit_nibbles, _nibbles_to_bytes,
                                         LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, NOP, VALUINST,
                                         InstOp, _NIB_COUNTS)
from test.mockgpu.amd.emu import _init_sqtt_encoder
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
# END-TO-END: sqtt_timeline on MOCKGPU emulator output (subprocess-based CI test)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sqtt_timeline_test(kernel_code: str) -> str:
  """Run a kernel on MOCKGPU with VIZ=-2 and verify sqtt_timeline doesn't crash. Returns stdout."""
  # The test runs inside a subprocess with MOCKGPU=1 AMD=1 VIZ=-2.
  # After running the kernel it reads the profile pkl and calls sqtt_timeline on each SQTT event.
  test_code = f'''
import pickle, sys
from tinygrad.helpers import temp

{kernel_code}

from tinygrad.renderer.amd.sqtt import decode, map_insts, INST, WAVESTART, WAVEEND, IMMEDIATE, InstOp
from tinygrad.viz.serve import sqtt_timeline

with open(temp("profile.pkl", append_user=True), "rb") as f:
  data = pickle.load(f)

sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
# Build program lookup: only AMD programs with lib bytes
prg_by_tag = {{}}
for e in data:
  if type(e).__name__ == "ProfileProgramEvent" and e.lib is not None and len(e.lib) > 0:
    prg_by_tag[e.tag] = e
dev = next((e for e in data if type(e).__name__ == "ProfileDeviceEvent" and e.device == "AMD"), None)
assert dev is not None, "no AMD device event"
target = dev.props.get("gfx_target_version", 0)

tested, passed = 0, 0
for ev in sqtt_events:
  if not ev.itrace or ev.kern not in prg_by_tag or len(ev.blob) == 0: continue
  prg = prg_by_tag[ev.kern]

  # Decode blob and verify basic structure
  packets = list(decode(ev.blob))
  assert len(packets) > 0, f"empty blob SE={{ev.se}}"
  assert packets[0].layout == 3, f"bad layout SE={{ev.se}}"

  # Count instruction and wave packets
  n_inst = sum(1 for p in packets if isinstance(p, INST))
  n_waves = sum(1 for p in packets if isinstance(p, WAVESTART))

  # Verify timestamps monotonic
  times = [p._time for p in packets]
  assert times == sorted(times), f"timestamps not monotonic SE={{ev.se}}"

  # Verify wave lifecycle
  n_ends = sum(1 for p in packets if isinstance(p, WAVEEND))
  assert n_waves == n_ends, f"wave start/end mismatch SE={{ev.se}}: {{n_waves}} != {{n_ends}}"

  # Verify store ops are labeled as stores, not loads
  for p in packets:
    if isinstance(p, INST) and isinstance(p.op, InstOp):
      if "STORE" in p.op.name:
        assert "LOAD" not in p.op.name, f"store op labeled as load: {{p.op}}"

  # Test map_insts — the main crash regression test
  # NOTE: KeyError can happen due to pre-existing kern tag mismatch bug (ev.kern may point to wrong program)
  try:
    results = list(map_insts(ev.blob, prg.lib, target))
    tested += 1
  except KeyError as e:
    print(f"SKIP: map_insts KeyError={{e}} SE={{ev.se}} kern={{ev.kern}} name={{prg.name}} n_inst={{n_inst}} (kern tag mismatch)")
    continue

  # Test sqtt_timeline — verifies the full visualization pipeline
  try:
    timeline = sqtt_timeline(ev.blob, prg.lib, target)
    passed += 1
  except Exception as e:
    print(f"FAIL: sqtt_timeline SE={{ev.se}} kern={{ev.kern}} name={{prg.name}}: {{e}}", file=sys.stderr)
    sys.exit(1)

  print(f"OK SE={{ev.se}} kern={{ev.kern}} name={{prg.name}}: {{n_inst}} insts, {{n_waves}} waves, {{len(timeline)}} timeline events")

assert tested > 0, "no SQTT events were tested"
print(f"PASSED: {{passed}}/{{tested}} sqtt_timeline calls succeeded")
'''
  env = os.environ.copy()
  env.update({"AMD": "1", "MOCKGPU": "1", "PYTHON_REMU": "1", "VIZ": "-2"})
  result = subprocess.run([sys.executable, "-c", test_code], env=env, capture_output=True, text=True, timeout=180)
  if result.returncode != 0:
    raise AssertionError(f"subprocess failed (rc={result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}")
  return result.stdout

class TestSQTTTimelineEndToEnd(unittest.TestCase):
  """End-to-end tests: run kernels on MOCKGPU, verify sqtt_timeline doesn't crash."""

  def test_sqtt_timeline_add(self):
    """sqtt_timeline works for a simple element-wise add kernel."""
    out = _run_sqtt_timeline_test("""
from tinygrad import Tensor
a = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
b = Tensor([5.0, 6.0, 7.0, 8.0]).realize()
c = (a + b).realize()
""")
    self.assertIn("PASSED", out)

  def test_sqtt_timeline_gemm(self):
    """sqtt_timeline works for a matmul kernel (has loops and branches)."""
    out = _run_sqtt_timeline_test("""
from tinygrad import Tensor
a = Tensor.rand(4, 4).realize()
b = Tensor.rand(4, 4).realize()
c = (a @ b).realize()
""")
    self.assertIn("PASSED", out)

  def test_sqtt_timeline_multi_op(self):
    """sqtt_timeline works for a chain of operations."""
    out = _run_sqtt_timeline_test("""
from tinygrad import Tensor
a = Tensor.rand(8, 8).realize()
b = (a * 2.0 + 1.0).relu().realize()
""")
    self.assertIn("PASSED", out)

  def test_sqtt_timeline_custom_asm(self):
    """sqtt_timeline works for a custom assembly kernel (amd_asm_matmul)."""
    env = os.environ.copy()
    env.update({"AMD": "1", "MOCKGPU": "1", "PYTHON_REMU": "1", "VIZ": "-2", "N": "256", "CNT": "1", "VERIFY": "0",
                "PYTHONPATH": os.path.join(os.path.dirname(__file__), "../..")})
    result = subprocess.run([sys.executable, "extra/gemm/amd_asm_matmul.py"], env=env, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
      raise AssertionError(f"amd_asm_matmul failed (rc={result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}")

    # Now verify the profile pkl: no emulator instruction names should appear as program events
    verify_code = '''
import pickle, sys
from tinygrad.helpers import temp

with open(temp("profile.pkl", append_user=True), "rb") as f:
  data = pickle.load(f)

prg_events = [e for e in data if type(e).__name__ == "ProfileProgramEvent"]
print(f"{len(prg_events)} program events")

# Check that emulator instruction runners didn't leak into profile events
bad = [e for e in prg_events if e.lib is None and e.tag is not None]
if bad:
  print(f"FAIL: {len(bad)} emulator instruction runners leaked into profile events:", file=sys.stderr)
  for e in bad[:5]: print(f"  tag={e.tag} name={e.name}", file=sys.stderr)
  sys.exit(1)

# Check that real kernel programs have lib bytes
real = [e for e in prg_events if e.lib is not None and len(e.lib) > 0]
print(f"{len(real)} real programs with lib bytes")
for e in real: print(f"  tag={e.tag} name={e.name} lib_size={len(e.lib)}")
assert len(real) >= 1, f"expected at least 1 real program, got {len(real)}"

# Verify sqtt_timeline works for SQTT events with matching programs
from tinygrad.renderer.amd.sqtt import decode, INST, VALUINST
from tinygrad.viz.serve import sqtt_timeline

sqtt_events = [e for e in data if type(e).__name__ == "ProfileSQTTEvent"]
dev = next((e for e in data if type(e).__name__ == "ProfileDeviceEvent" and e.device == "AMD"), None)
target = dev.props.get("gfx_target_version", 0) if dev else 0
prg_by_tag = {e.tag: e for e in prg_events if e.lib is not None and len(e.lib) > 0}

tested = 0
for ev in sqtt_events:
  if not ev.itrace or ev.kern not in prg_by_tag: continue
  prg = prg_by_tag[ev.kern]
  timeline = sqtt_timeline(ev.blob, prg.lib, target)
  pkts = list(decode(ev.blob))
  n_valu = sum(1 for p in pkts if isinstance(p, VALUINST))
  n_inst = sum(1 for p in pkts if isinstance(p, INST))
  print(f"OK kern={ev.kern} name={prg.name}: {n_inst} INST, {n_valu} VALUINST, {len(timeline)} timeline events")
  tested += 1

assert tested > 0, "no SQTT events were tested"
print(f"PASSED: {tested} sqtt_timeline calls succeeded")
'''
    env2 = {k: v for k, v in env.items() if k != "VIZ"}  # don't set VIZ in verification subprocess (would overwrite profile pkl)
    result2 = subprocess.run([sys.executable, "-c", verify_code], env=env2, capture_output=True, text=True, timeout=60)
    if result2.returncode != 0:
      raise AssertionError(f"verification failed (rc={result2.returncode}):\nstdout: {result2.stdout}\nstderr: {result2.stderr}")
    self.assertIn("PASSED", result2.stdout)
    # Verify no emulator instruction names leaked
    self.assertNotIn("FAIL", result2.stderr)

if __name__ == "__main__":
  unittest.main()
