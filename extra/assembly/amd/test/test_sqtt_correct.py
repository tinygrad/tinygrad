#!/usr/bin/env python3
"""Tests for SQTT emulator correctness against known hardware patterns.

These tests define the CORRECT behavior based on hardware traces.
When run against hardware (SQTT_HW=1), all tests should pass.
When run against the emulator, tests FAIL where the emulator is wrong.

Run emulator tests (expect failures - shows emulator bugs):
  PYTHONPATH="." python3 extra/assembly/amd/test/test_sqtt_correct.py

Run against real hardware (all tests should pass):
  SQTT_HW=1 PYTHONPATH="." python3 extra/assembly/amd/test/test_sqtt_correct.py
"""
import os
import unittest

USE_HW = os.environ.get("SQTT_HW", "0") == "1"

# Must set SQTT env vars before importing tinygrad device
if USE_HW:
  os.environ["SQTT"] = "1"
  os.environ["PROFILE"] = "1"
  os.environ["SQTT_LIMIT_SE"] = "2"
  os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"

from extra.assembly.amd.emu import SQTTState, decode_program, exec_wave, WaveState, LDSMem
from extra.assembly.amd.sqtt import WAVESTART, WAVEEND, IMMEDIATE, VALUINST, ALUEXEC, PacketType
from extra.assembly.amd.autogen.rdna3.ins import (v_mov_b32_e32, v_add_f32_e32, v_rcp_f32_e32, v_sqrt_f32_e32,
  v_exp_f32_e32, s_mov_b32, s_nop, s_endpgm, s_delay_alu, v_mul_f32_e32)
from extra.assembly.amd.dsl import v, s

def assemble(instructions: list) -> bytes:
  return b''.join(inst.to_bytes() for inst in instructions)

def wrap_with_nops(instructions: list, nops=16) -> list:
  return instructions + [s_nop(0)]*nops + [s_endpgm()]

def get_wave_packets(packets: list) -> list:
  """Extract packets from first WAVESTART to WAVEEND on simd 0."""
  result, in_wave = [], False
  for p in packets:
    if isinstance(p, WAVESTART) and p.simd == 0:
      in_wave, result = True, [p]
    elif in_wave:
      result.append(p)
      if isinstance(p, WAVEEND): break
  return result

def get_timing_deltas(packets: list) -> list[tuple[str, int]]:
  """Extract (packet_type, delta_from_previous) for non-timing packets."""
  skip_types = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG"}
  filtered = [p for p in packets if type(p).__name__ not in skip_types]
  if not filtered: return []
  result = [(type(filtered[0]).__name__, 0)]
  for i in range(1, len(filtered)):
    result.append((type(filtered[i]).__name__, filtered[i]._time - filtered[i-1]._time))
  if result and result[-1][0] == 'WAVEEND':
    result[-1] = ('WAVEEND', 0)  # normalize WAVEEND timing
  return result

def run_emulator(instructions: list) -> list:
  """Run instructions through emulator and return wave packets."""
  instructions = wrap_with_nops(instructions)
  code = assemble(instructions)
  program = decode_program(code)
  st = WaveState()
  st.exec_mask = (1 << 32) - 1
  lds = LDSMem(bytearray(65536))
  trace = SQTTState(wave_id=0, simd=0, cu=0)
  exec_wave(program, st, lds, 32, trace)
  return get_wave_packets(trace.packets)

def run_hardware(instructions: list) -> list:
  """Run on real hardware, return most common wave packet sequence."""
  from extra.assembly.amd.test.test_sqtt_hw import compile_asm_sqtt, run_prg_sqtt_batch
  from extra.assembly.amd.sqtt import decode
  from collections import Counter

  instructions = wrap_with_nops(instructions)
  prg = compile_asm_sqtt(instructions, alu_only=True)

  for _ in range(10):
    blobs = run_prg_sqtt_batch(prg, n_runs=200)
    traces = [get_wave_packets(decode(blob)) for blob in blobs]
    traces = [t for t in traces if t]  # filter empty
    if traces:
      # Return most common pattern
      delta_sets = [tuple(get_timing_deltas(t)) for t in traces]
      most_common = Counter(delta_sets).most_common(1)[0][0]
      # Find a trace matching that pattern
      for t in traces:
        if tuple(get_timing_deltas(t)) == most_common:
          return t
  return []

def run_sqtt(instructions: list) -> list:
  """Run on HW or emulator based on SQTT_HW env var, return wave packets."""
  return run_hardware(instructions) if USE_HW else run_emulator(instructions)

def run_test(instructions: list) -> list[tuple[str, int]]:
  """Run and return timing deltas."""
  return get_timing_deltas(run_sqtt(instructions))


class TestTranscendentals(unittest.TestCase):
  """Tests for transcendental instruction (v_rcp, v_sqrt, v_exp, v_log) tracing.

  Hardware behavior:
  - Transcendentals emit INST packets (not VALUINST) - they use a separate unit
  - Issue interval is 4 cycles (vs 1 cycle for regular VALU)
  - Latency is higher (~8 cycles vs ~6 for VALU)
  """

  def test_trans_emits_inst_not_valuinst(self):
    """Transcendentals should emit INST packets, not VALUINST.

    Hardware uses a separate transcendental unit that traces as INST.
    """
    deltas = run_test([v_rcp_f32_e32(v[0], v[0])])
    types = [d[0] for d in deltas]

    self.assertIn('INST', types, "Transcendental should emit INST")
    self.assertNotIn('VALUINST', types, "Transcendental should not emit VALUINST")

  def test_trans_4_cycle_issue_interval(self):
    """Independent transcendentals issue 4 cycles apart (not 1 like VALU).

    The transcendental unit is narrower and takes 4 cycles per instruction.
    """
    deltas = run_test([
      v_rcp_f32_e32(v[0], v[0]),
      v_sqrt_f32_e32(v[1], v[1]),
      v_exp_f32_e32(v[2], v[2]),
    ])

    # Find INST packets (transcendentals)
    inst_deltas = [d for d in deltas if d[0] == 'INST']
    self.assertEqual(len(inst_deltas), 3, "Should have 3 INST packets for 3 transcendentals")

    # Hardware: 4-cycle interval between transcendentals
    for d in inst_deltas[1:]:
      self.assertEqual(d[1], 4, "Trans should issue 4 cycles apart")


class TestSALUTracing(unittest.TestCase):
  """Tests for SALU instruction tracing.

  Hardware behavior:
  - SALU instructions (s_mov, s_add, etc.) emit INST packets
  - They execute on the scalar unit, separate from VALU
  """

  def test_salu_emits_inst(self):
    """SALU instructions should emit INST packets.

    Hardware traces scalar ALU operations with INST packets.
    """
    deltas = run_test([s_mov_b32(s[4], 1)])
    types = [d[0] for d in deltas]

    self.assertIn('INST', types, "SALU should emit INST")

  def test_mixed_salu_valu_interleaving(self):
    """Mixed SALU/VALU sequence should show interleaved INST and VALUINST.

    Hardware can issue SALU and VALU in parallel on different units.
    The trace shows both instruction types interleaved.
    """
    deltas = run_test([
      s_mov_b32(s[4], 1),
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[5], 2),
      v_mov_b32_e32(v[1], 2.0),
    ])
    types = [d[0] for d in deltas]

    self.assertIn('INST', types, "Should have INST for SALU")
    self.assertIn('VALUINST', types, "Should have VALUINST for VALU")
    inst_count = types.count('INST')
    valuinst_count = types.count('VALUINST')
    self.assertEqual(inst_count, 2, "Should have 2 INST packets for 2 SALUs")
    self.assertEqual(valuinst_count, 2, "Should have 2 VALUINST packets for 2 VALUs")


class TestLongDependencyChains(unittest.TestCase):
  """Tests for long VALU dependency chains.

  Hardware behavior:
  - Chains exhaust forwarding network after ~4-5 instructions
  - Once exhausted, instructions must wait for register writeback
  - This causes ALUEXEC timing to change at the boundary
  """

  def test_5_chain_aluexec_timing(self):
    """5-instruction dependency chain has specific ALUEXEC timing pattern.

    Hardware: First 4 ALUEXECs use forwarding, 5th waits for writeback.
    This shows up as 9-cycle delta before the last ALUEXEC.
    """
    deltas = run_test([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),
      v_add_f32_e32(v[2], v[1], v[1]),
      v_add_f32_e32(v[3], v[2], v[2]),
      v_add_f32_e32(v[4], v[3], v[3]),
    ])

    aluexec_deltas = [d for d in deltas if d[0] == 'ALUEXEC']
    self.assertEqual(len(aluexec_deltas), 5, "Should have 5 ALUEXECs")

    # HW: last ALUEXEC has delta=9 (waiting for writeback after forwarding exhausted)
    self.assertEqual(aluexec_deltas[-1][1], 9, "Last ALUEXEC should wait 9 cycles")


class TestDelayALU(unittest.TestCase):
  """Tests for s_delay_alu instruction effects.

  Hardware behavior:
  - s_delay_alu tells hardware about upcoming dependencies
  - It affects when the next instruction can issue
  - The encoding specifies which previous instruction to wait for
  """

  def test_delay_alu_affects_valuinst_timing(self):
    """s_delay_alu(1) delays next VALUINST to wait for VALU_DEP_1.

    Should show ~5-cycle gap between VALUINSTs when delay_alu is used.
    """
    deltas = run_test([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(1),  # VALU_DEP_1
      v_add_f32_e32(v[1], v[0], v[0]),
    ])

    valuinst_deltas = [d for d in deltas if d[0] == 'VALUINST']
    self.assertEqual(len(valuinst_deltas), 2, "Should have 2 VALUINSTs")

    # delay_alu should cause ~5 cycle gap
    self.assertGreaterEqual(valuinst_deltas[1][1], 4, "delay_alu should cause ~5 cycle gap")

  def test_delay_alu_aluexec_position(self):
    """ALUEXEC timing after delay_alu - second ALUEXEC preceded by IMMEDIATE.

    Hardware pattern: ..., IMMEDIATE, ALUEXEC (second one)
    """
    deltas = run_test([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(1),
      v_add_f32_e32(v[1], v[0], v[0]),
    ])

    # Find position of second ALUEXEC
    aluexec_indices = [i for i, d in enumerate(deltas) if d[0] == 'ALUEXEC']
    self.assertEqual(len(aluexec_indices), 2, "Should have 2 ALUEXECs")

    # On HW, there's an IMMEDIATE before the second ALUEXEC
    second_aluexec_idx = aluexec_indices[1]
    prev_type = deltas[second_aluexec_idx - 1][0]
    self.assertEqual(prev_type, 'IMMEDIATE', "Should have IMMEDIATE before second ALUEXEC")


class TestVALUAfterTrans(unittest.TestCase):
  """Tests for VALU instructions following transcendentals.

  Hardware behavior:
  - Transcendental completes later than regular VALU
  - Following VALUs can issue while trans is in flight
  - First packet should be INST (trans), followed by VALUINSTs
  """

  def test_valu_after_trans_packet_types(self):
    """VALUs after trans: first is INST (trans), rest are VALUINST.

    Trans has ~8 cycle latency, VALU has ~6 cycle.
    Trans issues first as INST, then VALUs as VALUINST.
    """
    deltas = run_test([
      v_rcp_f32_e32(v[0], v[0]),  # trans
      v_mov_b32_e32(v[1], 1.0),   # valu
      v_mov_b32_e32(v[2], 2.0),   # valu
    ])
    types = [d[0] for d in deltas]

    # First instruction (after WAVESTART) should be INST (trans)
    self.assertEqual(types[1], 'INST', "First instruction should be INST (trans)")
    self.assertEqual(types[2], 'VALUINST', "Second should be VALUINST")
    self.assertEqual(types[3], 'VALUINST', "Third should be VALUINST")


class TestBasicPatterns(unittest.TestCase):
  """Tests for basic patterns that work correctly."""

  def test_empty_program(self):
    """Empty program (just epilogue) produces WAVESTART, IMMEDIATEs, WAVEEND."""
    deltas = run_test([])
    types = [d[0] for d in deltas]
    self.assertEqual(types[0], 'WAVESTART')
    self.assertEqual(types[-1], 'WAVEEND')
    self.assertTrue(all(t == 'IMMEDIATE' for t in types[1:-1]))

  def test_single_valu(self):
    """Single VALU should produce VALUINST + ALUEXEC."""
    deltas = run_test([v_mov_b32_e32(v[0], 1.0)])
    types = [d[0] for d in deltas]
    self.assertEqual(types.count('VALUINST'), 1)
    self.assertEqual(types.count('ALUEXEC'), 1)

  def test_independent_valus(self):
    """Independent VALUs issue 1 cycle apart."""
    deltas = run_test([v_mov_b32_e32(v[i], float(i)) for i in range(4)])
    valuinst_deltas = [d for d in deltas if d[0] == 'VALUINST']
    for vd in valuinst_deltas[1:]:
      self.assertEqual(vd[1], 1)

  def test_snop_timing(self):
    """s_nop(7) produces larger delay than s_nop(0)."""
    snop0 = run_test([s_nop(0)])
    snop7 = run_test([s_nop(7)])
    imm0 = next(d for d in snop0 if d[0] == 'IMMEDIATE')
    imm7 = next(d for d in snop7 if d[0] == 'IMMEDIATE')
    self.assertGreater(imm7[1], imm0[1])


if __name__ == "__main__":
  unittest.main()
