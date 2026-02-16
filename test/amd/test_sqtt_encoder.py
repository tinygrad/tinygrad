#!/usr/bin/env python3
"""Tests for the SQTT encoder and end-to-end sqtt_timeline on MOCKGPU emulator output.

Run with: AMD=1 MOCKGPU=1 python -m pytest test/amd/test_sqtt_encoder.py -v
"""
import subprocess, sys, os, unittest
from dataclasses import dataclass
from tinygrad.renderer.amd.sqtt import (decode, _encode_raw, _emit_nibbles, _nibbles_to_bytes,
                                         LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, NOP,
                                         InstOp, _NIB_COUNTS)
from test.mockgpu.amd.emu import encode_sqtt
from tinygrad.runtime.autogen.amd.rdna3 import ins as ir3
from tinygrad.runtime.autogen.amd.rdna3.enum import SOPPOp

# ═══════════════════════════════════════════════════════════════════════════════
# MOCK SQTTInst (mirrors test/mockgpu/amd/emu.py SQTTInst without importing it)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FakeSQTTInst:
  pc_offset: int
  inst_type: type
  wave_id: int
  inst_op: int = 0
  inst_op_name: str = ""
  branch_taken: bool|None = None

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
    trace = [FakeSQTTInst(0, ir3.VOP3, 0)]
    packets = list(decode(encode_sqtt(trace)))
    types = [type(p) for p in packets if not isinstance(p, NOP)]
    self.assertIn(LAYOUT_HEADER, types)
    self.assertIn(WAVESTART, types)
    self.assertIn(INST, types)
    self.assertIn(WAVEEND, types)

  def test_inst_op_mapping(self):
    cases = [(ir3.VOP3, InstOp.VALU_TRANS), (ir3.SMEM, InstOp.SMEM), (ir3.SOP1, InstOp.SALU), (ir3.SOP2, InstOp.SALU)]
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
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 3)

  def test_wave_ids_preserved(self):
    trace = [FakeSQTTInst(0, ir3.VOP3, 3), FakeSQTTInst(0, ir3.SMEM, 7)]
    packets = list(decode(encode_sqtt(trace)))
    self.assertEqual(sorted(p.wave for p in packets if isinstance(p, WAVESTART)), [3, 7])

  def test_timestamps_monotonic(self):
    trace = [FakeSQTTInst(i*4, ir3.VOP3, 0) for i in range(5)]
    times = [p._time for p in decode(encode_sqtt(trace))]
    self.assertEqual(times, sorted(times))

  def test_blob_32byte_aligned(self):
    for n in [0, 1, 5, 10, 50]:
      blob = encode_sqtt([FakeSQTTInst(i*4, ir3.VOP3, 0) for i in range(n)])
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
  if not ev.itrace or ev.kern not in prg_by_tag: continue
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
  try:
    results = list(map_insts(ev.blob, prg.lib, target))
    tested += 1
  except KeyError as e:
    print(f"FAIL: map_insts KeyError={{e}} SE={{ev.se}} kern={{ev.kern}} name={{prg.name}} n_inst={{n_inst}}", file=sys.stderr)
    sys.exit(1)

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

if __name__ == "__main__":
  unittest.main()
