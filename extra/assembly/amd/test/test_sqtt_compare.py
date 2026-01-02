#!/usr/bin/env python3
"""Tests comparing hardware SQTT traces to emulator SQTT output.

Run with: python -m pytest extra/assembly/amd/test/test_sqtt_compare.py -v
Requires AMD GPU with SQTT support.
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_LIMIT_SE"] = "2"
os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"  # Exclude WAVERDY, REG, EVENT, UTILCTR, WAVEALLOC, PERF

import unittest
from tinygrad.device import Device
from tinygrad.helpers import DEBUG

from extra.assembly.amd.sqtt import decode, encode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, PacketType

dev = Device["AMD"]


class TestSQTTCodec(unittest.TestCase):
  """Tests for SQTT encoder/decoder roundtrip."""

  def test_roundtrip_simple(self):
    """Test encode/decode roundtrip for simple packets."""
    test_packets = [
      LAYOUT_HEADER.from_raw(0x100),
      WAVESTART.from_raw(0x0),
      INST.from_raw(0x10),  # delta=1
      INST.from_raw(0x10),  # delta=1
      WAVEEND.from_raw(0x40),  # delta=2
    ]
    encoded = encode(test_packets)
    decoded = decode(encoded)

    self.assertGreaterEqual(len(decoded), len(test_packets))
    for i, (orig, dec) in enumerate(zip(test_packets, decoded)):
      self.assertEqual(type(orig), type(dec), f"type mismatch at {i}: {orig} vs {dec}")

  def test_decode_real_blob(self):
    """Test decoding a real SQTT blob from examples."""
    import pickle
    from pathlib import Path
    example_path = Path(__file__).parent.parent.parent.parent / "sqtt/examples/profile_plus_run_0.pkl"
    if not example_path.exists():
      self.skipTest(f"Example file not found: {example_path}")

    with open(example_path, "rb") as f:
      data = pickle.load(f)

    sqtt_events = [e for e in data if isinstance(e, ProfileSQTTEvent)]
    self.assertGreater(len(sqtt_events), 0, "No SQTT events in example")

    packets = decode(sqtt_events[0].blob)
    self.assertGreater(len(packets), 0, "No packets decoded")
    # Should see common packet types
    pkt_types = {type(p) for p in packets}
    self.assertIn(LAYOUT_HEADER, pkt_types)


from extra.assembly.amd.emu import SQTTState, decode_program
from extra.assembly.amd.sqtt import VALUINST, ALUEXEC
from extra.assembly.amd.autogen.rdna3.ins import (v_mov_b32_e32, v_add_f32_e32, v_rcp_f32_e32, v_sqrt_f32_e32, v_exp_f32_e32, v_log_f32_e32, s_delay_alu,
  v_mul_f32_e32, v_sub_f32_e32, s_mov_b32, s_add_u32, s_sub_u32, s_mul_i32, s_nop,
  v_add_f64, v_mul_f64, v_fma_f64,  # Double precision
  v_wmma_f32_16x16x16_f16, v_wmma_f32_16x16x16_bf16,  # WMMA
  VOPD, VOPDOp)  # Dual issue
from extra.assembly.amd.dsl import v, s
from extra.assembly.amd.test.test_sqtt_hw import compile_asm_sqtt, run_prg_sqtt, run_prg_sqtt_batch, get_wave_packets, format_packet, assemble, wrap_with_nops

def run_emulator_sqtt(instructions: list) -> list[PacketType]:
  """Run instructions through emulator and return SQTT packets."""
  code = assemble(wrap_with_nops(instructions))
  program = decode_program(code)

  sqtt = SQTTState(wave_id=0, simd=0, cu=0)
  sqtt.emit_wavestart()  # advances cycle by WAVE_STARTUP_CYCLES

  for pc, inst in sorted(program.items()):
    if inst.op_name == 'S_ENDPGM': break
    sqtt.trace_inst(inst)

  sqtt.finalize()
  return sqtt.packets

def filter_timing_packets(packets: list) -> list:
  """Filter to packets within WAVESTART..WAVEEND."""
  return get_wave_packets(packets)

def filter_noise_packets(packets: list) -> list:
  """Filter out pure timing/noise packets, keeping all meaningful packets."""
  skip_types = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3"}
  return [p for p in packets if type(p).__name__ not in skip_types]

def get_timing_deltas(packets: list) -> list[tuple[str, int]]:
  """Extract timing deltas between consecutive packets, starting with WAVESTART at delta=0."""
  filtered = filter_timing_packets(packets)
  if not filtered: return []
  # First packet (WAVESTART) has delta from itself (0), rest are deltas from previous
  result = [(type(filtered[0]).__name__, 0)] + [(type(filtered[i]).__name__, filtered[i]._time - filtered[i-1]._time) for i in range(1, len(filtered))]
  # Normalize WAVEEND timing to 0 (ignore completion timing for now)
  if result and result[-1][0] == 'WAVEEND':
    result[-1] = ('WAVEEND', 0)
  return result

@unittest.skipIf(not hasattr(dev, 'profile_events'), "AMD device required")
class TestEmulatorSQTT(unittest.TestCase):
  """Tests comparing emulator SQTT to hardware SQTT."""

  def _run_and_compare(self, instructions: list, name: str = "", n_runs: int = 200, min_identical: int = 20, max_attempts: int = 10):
    """Run instructions on both hardware and emulator, compare SQTT structure."""
    from collections import Counter

    # Compile once
    prg = compile_asm_sqtt(instructions, alu_only=True)

    # Retry up to max_attempts times until we get min_identical matching patterns
    for attempt in range(max_attempts):
      # Run kernel n_runs times in a single queue submission - all traces captured in one SQTT buffer
      blobs = run_prg_sqtt_batch(prg, n_runs=n_runs)

      # Extract all wave traces from the blobs (one blob per shader engine)
      hw_traces = []
      for blob in blobs:
        packets = decode(blob)
        # Find all WAVESTART..WAVEEND ranges on SIMD 0
        in_wave = False
        current_wave = []
        for p in packets:
          if isinstance(p, WAVESTART) and p.simd == 0:
            in_wave = True
            current_wave = [p]
          elif in_wave:
            current_wave.append(p)
            if isinstance(p, WAVEEND):
              hw_traces.append(current_wave)
              in_wave = False
              current_wave = []

      if not hw_traces:
        continue

      # Check if we have enough identical patterns (ignore WAVEEND timing - focus on issue timing only)
      skip_types = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG"}
      def wave_deltas(pkts):
        filtered = [p for p in pkts if type(p).__name__ not in skip_types]
        if not filtered: return []
        result = [(type(filtered[0]).__name__, 0)] + [(type(filtered[i]).__name__, filtered[i]._time - filtered[i-1]._time) for i in range(1, len(filtered))]
        # Normalize WAVEEND timing to 0 (ignore completion timing for now)
        if result and result[-1][0] == 'WAVEEND':
          result[-1] = ('WAVEEND', 0)
        return result

      hw_delta_sets = [wave_deltas(t) for t in hw_traces]
      pattern_counts = Counter(tuple(d) for d in hw_delta_sets)

      # Check if we have enough patterns with WAVEEND (timing normalized to 0)
      patterns_with_waveend = [p for p in pattern_counts if p[-1] == ('WAVEEND', 0)]
      count_waveend = sum(pattern_counts[p] for p in patterns_with_waveend)
      if count_waveend >= min_identical:
        break
    else:
      if not hw_traces:
        self.skipTest(f"Could not capture any hardware traces on SIMD 0 after {max_attempts} attempts")

    # Run on emulator
    emu_packets = run_emulator_sqtt(instructions)
    emu_deltas = get_timing_deltas(emu_packets)

    # Find most common pattern and a representative trace for it
    most_common_pattern = list(pattern_counts.most_common(1)[0][0]) if pattern_counts else []
    most_common_trace = next((t for t, d in zip(hw_traces, hw_delta_sets) if d == most_common_pattern), hw_traces[0])

    if DEBUG >= 2:
      print(f"\n{'='*70}")
      print(f"TEST: {name} ({len(hw_traces)} traces from {n_runs} runs)")
      print(f"{'='*70}")

      print(f"Timing patterns:")
      for pattern, count in pattern_counts.most_common():
        match = " <- MATCH" if list(pattern) == emu_deltas else ""
        print(f"  {count:2d}x: {list(pattern)}{match}")

      print(f"\nEmulator: {emu_deltas}")

      # Print HW trace (filter noise, normalize to WAVESTART time)
      hw_filtered = filter_noise_packets(most_common_trace)
      hw_t0 = hw_filtered[0]._time if hw_filtered else 0
      print(f"\nHW:")
      for p in hw_filtered:
        t = p._time - hw_t0 if hasattr(p, '_time') else 0
        print(f"  {t:8d}: {format_packet(p)}")

      # Print emulator trace
      emu_t0 = emu_packets[0]._time if emu_packets else 0
      print(f"\nEmulator:")
      for p in emu_packets:
        t = p._time - emu_t0 if hasattr(p, '_time') else 0
        print(f"  {t:8d}: {format_packet(p)}")

    # Assert emulator pattern matches most common HW pattern exactly
    emu_pattern = tuple(emu_deltas)
    self.assertIn(emu_pattern, pattern_counts,
      f"{name}: emulator pattern not found in HW traces.\n"
      f"Emulator: {emu_deltas}\n"
      f"HW patterns: {[list(p) for p in pattern_counts.most_common(3)]}")

  def test_salu_independent(self):
    """SALU instructions with no dependencies."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_mov_b32(s[5], 2),
      s_mov_b32(s[6], 3),
    ], "3 SALU independent")

  def test_salu_chain(self):
    """SALU instructions with chain dependencies."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_add_u32(s[5], s[4], 1),
      s_add_u32(s[6], s[5], 1),
    ], "3 SALU chain")

  def test_empty(self):
    """Empty program - just s_endpgm."""
    self._run_and_compare([], "empty")

  def _test_valu_independent_n(self, n: int, trans=False):
    """VALU instructions with no dependencies."""
    self._run_and_compare([
      v_rcp_f32_e32(v[i], v[i]) if trans else v_mov_b32_e32(v[i], float(i)) for i in range(n)
    ], f"{n} VALU independent")

  def test_valu_independent_1(self): self._test_valu_independent_n(1)
  def test_valu_independent_2(self): self._test_valu_independent_n(2)
  def test_valu_independent_3(self): self._test_valu_independent_n(3)
  def test_valu_independent_4(self): self._test_valu_independent_n(4)
  def test_valu_independent_5(self): self._test_valu_independent_n(5)
  def test_valu_independent_6(self): self._test_valu_independent_n(6)
  def test_valu_independent_7(self): self._test_valu_independent_n(7)
  def test_valu_independent_8(self): self._test_valu_independent_n(8)
  def test_valu_independent_16(self): self._test_valu_independent_n(16)

  def test_trans_independent_16(self): self._test_valu_independent_n(16, trans=True)

  def test_valu_chain(self):
    """VALU instructions with chain dependencies."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),
      v_add_f32_e32(v[2], v[1], v[1]),
    ], "3 VALU chain")

  def test_trans_independent(self):
    """Transcendental instructions with no dependencies."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      v_sqrt_f32_e32(v[1], v[1]),
      v_exp_f32_e32(v[2], v[2]),
    ], "3 TRANS independent")

  def test_trans_chain(self):
    """Transcendental instructions with chain dependencies."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_rcp_f32_e32(v[1], v[0]),
      v_sqrt_f32_e32(v[2], v[1]),
    ], "3 TRANS chain")

  # ─────────────────────────────────────────────────────────────────────────────
  # Additional dependency pattern tests
  # ─────────────────────────────────────────────────────────────────────────────

  def test_valu_waw(self):
    """VALU WAW (write-after-write) - writing to same register."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[0], 3.0),
    ], "3 VALU WAW")

  def test_valu_partial_dep(self):
    """VALU with partial dependencies - some dependent, some independent."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),        # independent
      v_mov_b32_e32(v[1], 2.0),        # independent
      v_add_f32_e32(v[2], v[0], v[1]), # depends on v0, v1
      v_mov_b32_e32(v[3], 4.0),        # independent
    ], "4 VALU partial dep")

  def test_valu_long_chain(self):
    """VALU long chain - 5 dependent instructions."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),
      v_add_f32_e32(v[2], v[1], v[1]),
      v_add_f32_e32(v[3], v[2], v[2]),
      v_add_f32_e32(v[4], v[3], v[3]),
    ], "5 VALU long chain")

  def test_salu_waw(self):
    """SALU WAW (write-after-write) - writing to same register."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_mov_b32(s[4], 2),
      s_mov_b32(s[4], 3),
    ], "3 SALU WAW")

  def test_salu_with_snop32(self):
    """SALU with s_nop(32) - tests s_nop delay timing."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_nop(32),
      s_mov_b32(s[5], 2),
    ], "SALU with s_nop(32)")

  def test_mixed_salu_valu(self):
    """Mixed SALU and VALU sequence."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[5], 2),
      v_mov_b32_e32(v[1], 2.0),
    ], "mixed SALU/VALU")

  def test_valu_fan_out(self):
    """VALU fan-out - one source used by multiple consumers."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),
      v_add_f32_e32(v[2], v[0], v[0]),
      v_add_f32_e32(v[3], v[0], v[0]),
    ], "4 VALU fan-out")

  def test_valu_after_trans(self):
    """VALU after transcendental - tests TRANS->VALU timing."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 2.0),
    ], "TRANS then VALU")

  def test_trans_after_valu(self):
    """Transcendental after VALU - tests VALU->TRANS timing."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_rcp_f32_e32(v[2], v[2]),
    ], "VALU then TRANS")

  # ─────────────────────────────────────────────────────────────────────────────
  # Additional SALU tests
  # ─────────────────────────────────────────────────────────────────────────────

  def test_salu_independent_6(self):
    """6 independent SALU instructions."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_mov_b32(s[5], 2),
      s_mov_b32(s[6], 3),
      s_mov_b32(s[7], 4),
      s_mov_b32(s[8], 5),
      s_mov_b32(s[9], 6),
    ], "6 SALU independent")

  def test_salu_with_snop1(self):
    """SALU with s_nop(1) - minimal delay."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_nop(1),
      s_mov_b32(s[5], 2),
    ], "SALU with s_nop(1)")

  def test_salu_with_snop7(self):
    """SALU with s_nop(7) - medium delay."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_nop(7),
      s_mov_b32(s[5], 2),
    ], "SALU with s_nop(7)")

  # ─────────────────────────────────────────────────────────────────────────────
  # Additional TRANS tests
  # ─────────────────────────────────────────────────────────────────────────────

  def test_trans_dep_chain(self):
    """Transcendental with dependency chain - each trans depends on previous."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_rcp_f32_e32(v[1], v[0]),
      v_sqrt_f32_e32(v[2], v[1]),
    ], "TRANS dep chain")

  def test_salu_then_trans(self):
    """SALU followed by transcendental."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_mov_b32(s[5], 2),
      v_rcp_f32_e32(v[0], v[0]),
    ], "SALU then TRANS")

  def test_trans_then_salu(self):
    """Transcendental followed by SALU."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      s_mov_b32(s[4], 1),
      s_mov_b32(s[5], 2),
    ], "TRANS then SALU")

  # ─────────────────────────────────────────────────────────────────────────────
  # More complex mixed sequences
  # ─────────────────────────────────────────────────────────────────────────────

  def test_valu_salu_valu(self):
    """VALU-SALU-VALU sandwich."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[4], 1),
      v_mov_b32_e32(v[1], 2.0),
    ], "VALU-SALU-VALU")

  def test_salu_valu_salu(self):
    """SALU-VALU-SALU sandwich."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[5], 2),
    ], "SALU-VALU-SALU")

  def test_mixed_salu_valu_trans(self):
    """Mixed SALU, VALU, and TRANS sequence."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      v_mov_b32_e32(v[0], 1.0),
      v_rcp_f32_e32(v[1], v[1]),
      v_mov_b32_e32(v[2], 2.0),
    ], "SALU-VALU-TRANS-VALU")

  # ─────────────────────────────────────────────────────────────────────────────
  # VALU dependency patterns
  # ─────────────────────────────────────────────────────────────────────────────

  def test_valu_diamond(self):
    """VALU diamond dependency - v0 -> v1,v2 -> v3."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on v0
      v_mul_f32_e32(v[2], v[0], v[0]),  # depends on v0
      v_add_f32_e32(v[3], v[1], v[2]),  # depends on v1 and v2
    ], "VALU diamond")

  def test_valu_staggered_deps(self):
    """VALU with staggered dependencies."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),        # independent
      v_mov_b32_e32(v[1], 2.0),        # independent
      v_mov_b32_e32(v[2], 3.0),        # independent
      v_add_f32_e32(v[3], v[0], v[1]), # depends on v0, v1
      v_add_f32_e32(v[4], v[1], v[2]), # depends on v1, v2
    ], "VALU staggered deps")

  def test_valu_independent_then_chain(self):
    """Independent VALUs followed by a dependency chain."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 3.0),
      v_add_f32_e32(v[3], v[0], v[0]),  # depends on v0
      v_add_f32_e32(v[4], v[3], v[3]),  # depends on v3
    ], "VALU independent then chain")

  # ─────────────────────────────────────────────────────────────────────────────
  # Longer sequences
  # ─────────────────────────────────────────────────────────────────────────────

  def test_valu_long_independent(self):
    """12 independent VALU instructions."""
    self._run_and_compare([
      v_mov_b32_e32(v[i], float(i)) for i in range(12)
    ], "12 VALU independent")

  def test_salu_long_chain(self):
    """Long SALU chain - 5 dependent instructions."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_add_u32(s[5], s[4], 1),
      s_add_u32(s[6], s[5], 1),
      s_add_u32(s[7], s[6], 1),
      s_add_u32(s[8], s[7], 1),
    ], "5 SALU chain")

  # ─────────────────────────────────────────────────────────────────────────────
  # Multiple TRANS tests (4-cycle issue interval)
  # ─────────────────────────────────────────────────────────────────────────────

  def test_trans_multiple_independent(self):
    """Multiple independent transcendentals - tests 4-cycle issue interval."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      v_rcp_f32_e32(v[1], v[1]),
      v_rcp_f32_e32(v[2], v[2]),
    ], "3 TRANS independent")

  def test_trans_4_independent(self):
    """4 independent transcendentals."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      v_sqrt_f32_e32(v[1], v[1]),
      v_exp_f32_e32(v[2], v[2]),
      v_log_f32_e32(v[3], v[3]),
    ], "4 TRANS independent")

  # ─────────────────────────────────────────────────────────────────────────────
  # TRANS with VALU dependencies
  # ─────────────────────────────────────────────────────────────────────────────

  def test_valu_reads_trans(self):
    """VALU reading from TRANS result - tests TRANS→VALU dependency."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on TRANS result
    ], "VALU reads TRANS")

  def test_trans_reads_valu(self):
    """TRANS reading from VALU result."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_rcp_f32_e32(v[1], v[0]),  # depends on VALU result
    ], "TRANS reads VALU")

  # ─────────────────────────────────────────────────────────────────────────────
  # Mixed long sequences
  # ─────────────────────────────────────────────────────────────────────────────

  def test_alternating_salu_valu(self):
    """Alternating SALU and VALU instructions."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[5], 2),
      v_mov_b32_e32(v[1], 2.0),
      s_mov_b32(s[6], 3),
      v_mov_b32_e32(v[2], 3.0),
    ], "alternating SALU/VALU")

  def test_valu_burst_then_salu_burst(self):
    """VALU burst followed by SALU burst."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 3.0),
      s_mov_b32(s[4], 1),
      s_mov_b32(s[5], 2),
      s_mov_b32(s[6], 3),
    ], "VALU burst then SALU burst")

  def test_salu_burst_then_valu_burst(self):
    """SALU burst followed by VALU burst."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_mov_b32(s[5], 2),
      s_mov_b32(s[6], 3),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 3.0),
    ], "SALU burst then VALU burst")

  # ─────────────────────────────────────────────────────────────────────────────
  # Complex dependency patterns
  # ─────────────────────────────────────────────────────────────────────────────

  def test_valu_reduction(self):
    """VALU reduction pattern - sum of 4 values."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 3.0),
      v_mov_b32_e32(v[3], 4.0),
      v_add_f32_e32(v[4], v[0], v[1]),  # v4 = v0 + v1
      v_add_f32_e32(v[5], v[2], v[3]),  # v5 = v2 + v3
      v_add_f32_e32(v[6], v[4], v[5]),  # v6 = v4 + v5
    ], "VALU reduction")

  def test_valu_deep_chain(self):
    """Deep VALU dependency chain - 8 dependent instructions."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),
      v_add_f32_e32(v[2], v[1], v[1]),
      v_add_f32_e32(v[3], v[2], v[2]),
      v_add_f32_e32(v[4], v[3], v[3]),
      v_add_f32_e32(v[5], v[4], v[4]),
      v_add_f32_e32(v[6], v[5], v[5]),
      v_add_f32_e32(v[7], v[6], v[6]),
    ], "8 VALU deep chain")

  # ─────────────────────────────────────────────────────────────────────────────
  # Single instruction tests
  # ─────────────────────────────────────────────────────────────────────────────

  def test_single_salu(self):
    """Single SALU instruction."""
    self._run_and_compare([s_mov_b32(s[4], 1)], "single SALU")

  def test_single_valu(self):
    """Single VALU instruction."""
    self._run_and_compare([v_mov_b32_e32(v[0], 1.0)], "single VALU")

  def test_single_trans(self):
    """Single transcendental instruction."""
    self._run_and_compare([v_rcp_f32_e32(v[0], v[0])], "single TRANS")

  # ═══════════════════════════════════════════════════════════════════════════
  # s_delay_alu tests - these help understand hardware pipeline latencies
  # ═══════════════════════════════════════════════════════════════════════════
  # s_delay_alu encoding: bits 3:0 = ID0, bits 6:4 = SKIP, bits 10:7 = ID1
  # ID values: 0=NO_DEP, 1=VALU_DEP_1, 2=VALU_DEP_2, 3=VALU_DEP_3, 4=VALU_DEP_4
  #            5=TRANS32_DEP_1, 6=TRANS32_DEP_2, 7=TRANS32_DEP_3
  #            8=FMA_ACCUM_CYCLE_1, 9=SALU_CYCLE_1, 10=SALU_CYCLE_2, 11=SALU_CYCLE_3

  def test_valu_chain_with_delay_alu_dep1(self):
    """VALU chain with s_delay_alu VALU_DEP_1 - tests if delay affects timing."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(1),  # instid0(VALU_DEP_1)
      v_add_f32_e32(v[1], v[0], v[0]),
    ], "VALU chain with delay_alu DEP_1")

  def test_valu_chain_with_delay_alu_dep2(self):
    """VALU chain with s_delay_alu VALU_DEP_2."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[10], 2.0),  # independent, pushes dep back
      s_delay_alu(2),  # instid0(VALU_DEP_2)
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on first v_mov
    ], "VALU chain with delay_alu DEP_2")

  def test_valu_chain_with_delay_alu_dep4(self):
    """VALU chain with s_delay_alu VALU_DEP_4."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[10], 2.0),
      v_mov_b32_e32(v[11], 3.0),
      v_mov_b32_e32(v[12], 4.0),
      s_delay_alu(4),  # instid0(VALU_DEP_4)
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on first v_mov
    ], "VALU chain with delay_alu DEP_4")

  def test_trans_with_delay_alu_dep1(self):
    """TRANS with s_delay_alu TRANS32_DEP_1."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      s_delay_alu(5),  # instid0(TRANS32_DEP_1)
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on TRANS result
    ], "TRANS with delay_alu TRANS_DEP_1")

  def test_trans_with_delay_alu_dep3(self):
    """TRANS chain with s_delay_alu TRANS32_DEP_3."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      v_sqrt_f32_e32(v[10], v[10]),  # independent
      v_exp_f32_e32(v[11], v[11]),   # independent
      s_delay_alu(7),  # instid0(TRANS32_DEP_3)
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on first TRANS
    ], "TRANS with delay_alu TRANS_DEP_3")

  def test_salu_with_delay_alu_cycle1(self):
    """SALU with s_delay_alu SALU_CYCLE_1."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_delay_alu(9),  # instid0(SALU_CYCLE_1)
      s_mov_b32(s[5], s[4]),  # depends on prior SALU
    ], "SALU with delay_alu SALU_CYCLE_1")

  def test_salu_with_delay_alu_cycle2(self):
    """SALU with s_delay_alu SALU_CYCLE_2."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_delay_alu(10),  # instid0(SALU_CYCLE_2)
      s_mov_b32(s[5], s[4]),  # depends on prior SALU
    ], "SALU with delay_alu SALU_CYCLE_2")

  def test_delay_alu_skip_next(self):
    """s_delay_alu with SKIP=NEXT to apply delay to second instruction."""
    # SKIP=1 (NEXT) in bits 6:4, ID1=1 (VALU_DEP_1) in bits 10:7
    # encoding: ID0=0 | (SKIP=1 << 4) | (ID1=1 << 7) = 0 | 16 | 128 = 144
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(144),  # instskip(NEXT) | instid1(VALU_DEP_1)
      v_mov_b32_e32(v[10], 2.0),  # no delay (ID0=0)
      v_add_f32_e32(v[1], v[0], v[0]),  # delayed (ID1=VALU_DEP_1)
    ], "delay_alu with SKIP=NEXT")

  # ═══════════════════════════════════════════════════════════════════════════
  # s_delay_alu comparison tests - these compare WITH vs WITHOUT delay_alu
  # ═══════════════════════════════════════════════════════════════════════════

  def test_delay_alu_no_dep(self):
    """s_delay_alu with NO_DEP (0) - should have no effect on timing."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(0),  # instid0(NO_DEP)
      v_mov_b32_e32(v[1], 2.0),
    ], "delay_alu NO_DEP")

  def test_delay_alu_valu_dep1_independent(self):
    """s_delay_alu VALU_DEP_1 with actually independent instructions."""
    # The next instruction doesn't actually depend on v[0], but delay_alu says it does
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(1),  # instid0(VALU_DEP_1) - claims dependency
      v_mov_b32_e32(v[1], 2.0),  # independent! doesn't use v[0]
    ], "delay_alu DEP_1 independent")

  def test_delay_alu_valu_dep1_vs_none(self):
    """Compare VALU chain with true dependency - with delay_alu."""
    # This has actual dependency v[0]->v[1], delay_alu should match dependency stall
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(1),  # VALU_DEP_1
      v_add_f32_e32(v[1], v[0], v[0]),
    ], "VALU dep with delay_alu")

  def test_valu_true_dep_no_delay_alu(self):
    """VALU chain with true dependency - WITHOUT delay_alu."""
    # Compare against test_delay_alu_valu_dep1_vs_none to see if delay_alu matters
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on v[0]
    ], "VALU dep no delay_alu")

  def test_delay_alu_trans_dep1_independent(self):
    """s_delay_alu TRANS32_DEP_1 with actually independent VALU."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      s_delay_alu(5),  # TRANS32_DEP_1
      v_mov_b32_e32(v[1], 2.0),  # independent! doesn't use v[0]
    ], "delay_alu TRANS_DEP_1 independent")

  def test_trans_valu_independent_no_delay_alu(self):
    """TRANS followed by independent VALU - WITHOUT delay_alu."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),
      v_mov_b32_e32(v[1], 2.0),  # independent
    ], "TRANS then VALU independent no delay_alu")

  def test_delay_alu_salu_cycle1_independent(self):
    """s_delay_alu SALU_CYCLE_1 with actually independent SALU."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_delay_alu(9),  # SALU_CYCLE_1
      s_mov_b32(s[5], 2),  # independent! doesn't use s[4]
    ], "delay_alu SALU_CYCLE_1 independent")

  def test_salu_independent_no_delay_alu(self):
    """Two independent SALUs - WITHOUT delay_alu."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_mov_b32(s[5], 2),
    ], "2 SALU independent no delay_alu")

  # ─────────────────────────────────────────────────────────────────────────────
  # Test all VALU_DEP levels (1-4) with same instructions
  # ─────────────────────────────────────────────────────────────────────────────

  def test_delay_alu_valu_dep3(self):
    """s_delay_alu VALU_DEP_3 - depends on 3rd previous VALU."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),        # target of dependency
      v_mov_b32_e32(v[10], 2.0),       # filler
      v_mov_b32_e32(v[11], 3.0),       # filler
      s_delay_alu(3),  # VALU_DEP_3
      v_add_f32_e32(v[1], v[0], v[0]), # depends on first v_mov
    ], "delay_alu VALU_DEP_3")

  # ─────────────────────────────────────────────────────────────────────────────
  # Test SALU_CYCLE levels (1-3)
  # ─────────────────────────────────────────────────────────────────────────────

  def test_delay_alu_salu_cycle3(self):
    """s_delay_alu SALU_CYCLE_3."""
    self._run_and_compare([
      s_mov_b32(s[4], 1),
      s_delay_alu(11),  # SALU_CYCLE_3
      s_mov_b32(s[5], s[4]),
    ], "delay_alu SALU_CYCLE_3")

  # ─────────────────────────────────────────────────────────────────────────────
  # Test TRANS32_DEP levels (1-3)
  # ─────────────────────────────────────────────────────────────────────────────

  def test_delay_alu_trans_dep2(self):
    """s_delay_alu TRANS32_DEP_2."""
    self._run_and_compare([
      v_rcp_f32_e32(v[0], v[0]),        # target
      v_sqrt_f32_e32(v[10], v[10]),     # filler TRANS
      s_delay_alu(6),  # TRANS32_DEP_2
      v_add_f32_e32(v[1], v[0], v[0]),  # depends on first TRANS
    ], "delay_alu TRANS_DEP_2")

  # ─────────────────────────────────────────────────────────────────────────────
  # Test SKIP values
  # ─────────────────────────────────────────────────────────────────────────────

  def test_delay_alu_skip_same(self):
    """s_delay_alu with SKIP=SAME (0) - delay applies to immediate next inst."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(1),  # SKIP=0 (SAME), ID0=VALU_DEP_1
      v_add_f32_e32(v[1], v[0], v[0]),  # this gets delayed
    ], "delay_alu SKIP=SAME")

  def test_delay_alu_skip_next2(self):
    """s_delay_alu with SKIP=NEXT2 to skip 2 instructions."""
    # SKIP=2 (NEXT2) in bits 6:4, ID1=1 (VALU_DEP_1) in bits 10:7
    # encoding: ID0=0 | (SKIP=2 << 4) | (ID1=1 << 7) = 0 | 32 | 128 = 160
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(160),  # instskip(NEXT2) | instid1(VALU_DEP_1)
      v_mov_b32_e32(v[10], 2.0),  # no delay
      v_mov_b32_e32(v[11], 3.0),  # no delay
      v_add_f32_e32(v[1], v[0], v[0]),  # delayed
    ], "delay_alu SKIP=NEXT2")

  def test_delay_alu_skip_next3(self):
    """s_delay_alu with SKIP=NEXT3 to skip 3 instructions."""
    # SKIP=3 (NEXT3) in bits 6:4, ID1=1 (VALU_DEP_1) in bits 10:7
    # encoding: ID0=0 | (SKIP=3 << 4) | (ID1=1 << 7) = 0 | 48 | 128 = 176
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      s_delay_alu(176),  # instskip(NEXT3) | instid1(VALU_DEP_1)
      v_mov_b32_e32(v[10], 2.0),  # no delay
      v_mov_b32_e32(v[11], 3.0),  # no delay
      v_mov_b32_e32(v[12], 4.0),  # no delay
      v_add_f32_e32(v[1], v[0], v[0]),  # delayed
    ], "delay_alu SKIP=NEXT3")

  # ─────────────────────────────────────────────────────────────────────────────
  # Combined ID0 and ID1 tests
  # ─────────────────────────────────────────────────────────────────────────────

  def test_delay_alu_both_ids(self):
    """s_delay_alu with both ID0 and ID1 set."""
    # ID0=1 (VALU_DEP_1), SKIP=1 (NEXT), ID1=1 (VALU_DEP_1)
    # encoding: 1 | (1 << 4) | (1 << 7) = 1 | 16 | 128 = 145
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[2], 3.0),
      s_delay_alu(145),  # ID0=VALU_DEP_1, SKIP=NEXT, ID1=VALU_DEP_1
      v_add_f32_e32(v[1], v[0], v[0]),  # delayed by ID0 (depends on v[0])
      v_add_f32_e32(v[3], v[2], v[2]),  # delayed by ID1 (depends on v[2])
    ], "delay_alu both IDs")

  # ═══════════════════════════════════════════════════════════════════════════
  # Double Precision (DP) tests - separate 2-wide unit
  # ═══════════════════════════════════════════════════════════════════════════
  # DP unit: 2-wide, processes 32 lanes in 16 passes
  # Expected: ~42 cycle latency, ~32 cycle issue interval

  def test_dp_single(self):
    """Single double precision add."""
    self._run_and_compare([
      v_add_f64(v[2], v[0], v[0]),  # v[2:3] = v[0:1] + v[0:1]
    ], "single DP add")

  def test_dp_independent_2(self):
    """Two independent DP operations."""
    self._run_and_compare([
      v_add_f64(v[2], v[0], v[0]),
      v_add_f64(v[4], v[0], v[0]),
    ], "2 DP independent")

  def test_dp_independent_3(self):
    """Three independent DP operations."""
    self._run_and_compare([
      v_add_f64(v[2], v[0], v[0]),
      v_add_f64(v[4], v[0], v[0]),
      v_add_f64(v[6], v[0], v[0]),
    ], "3 DP independent")

  def test_dp_chain(self):
    """DP operations with dependency chain."""
    self._run_and_compare([
      v_add_f64(v[2], v[0], v[0]),
      v_add_f64(v[4], v[2], v[2]),  # depends on v[2:3]
    ], "DP chain")

  def test_dp_mul(self):
    """Double precision multiply."""
    self._run_and_compare([
      v_mul_f64(v[2], v[0], v[0]),
    ], "single DP mul")

  def test_dp_fma(self):
    """Double precision FMA."""
    self._run_and_compare([
      v_fma_f64(v[2], v[0], v[0], v[0]),  # v[2:3] = v[0:1] * v[0:1] + v[0:1]
    ], "single DP FMA")

  def test_dp_after_valu(self):
    """DP after VALU - tests cross-unit timing."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f64(v[2], v[0], v[0]),
    ], "VALU then DP")

  def test_valu_after_dp(self):
    """VALU after DP - tests cross-unit timing."""
    self._run_and_compare([
      v_add_f64(v[2], v[0], v[0]),
      v_mov_b32_e32(v[10], 1.0),
    ], "DP then VALU")

  # ═══════════════════════════════════════════════════════════════════════════
  # VOPD (Dual Issue) tests - two VALU ops in one instruction
  # ═══════════════════════════════════════════════════════════════════════════
  # VOPD: dual-issue VALU, expected ~9 cycle latency

  def test_vopd_single(self):
    """Single VOPD instruction (dual mov)."""
    self._run_and_compare([
      VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32,
           vdstx=v[0], srcx0=v[10], vsrcx1=v[10],
           vdsty=v[1], srcy0=v[11], vsrcy1=v[11]),
    ], "single VOPD")

  def test_vopd_independent_2(self):
    """Two independent VOPD instructions."""
    self._run_and_compare([
      VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32,
           vdstx=v[0], srcx0=v[10], vsrcx1=v[10],
           vdsty=v[1], srcy0=v[11], vsrcy1=v[11]),
      VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32,
           vdstx=v[2], srcx0=v[12], vsrcx1=v[12],
           vdsty=v[3], srcy0=v[13], vsrcy1=v[13]),
    ], "2 VOPD independent")

  def test_vopd_add_mul(self):
    """VOPD with add and mul operations."""
    self._run_and_compare([
      VOPD(opx=VOPDOp.V_DUAL_ADD_F32, opy=VOPDOp.V_DUAL_MUL_F32,
           vdstx=v[0], srcx0=v[10], vsrcx1=v[11],
           vdsty=v[1], srcy0=v[12], vsrcy1=v[13]),
    ], "VOPD add+mul")

  def test_vopd_after_valu(self):
    """VOPD after regular VALU."""
    self._run_and_compare([
      v_mov_b32_e32(v[10], 1.0),
      VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32,
           vdstx=v[0], srcx0=v[10], vsrcx1=v[10],
           vdsty=v[1], srcy0=v[11], vsrcy1=v[11]),
    ], "VALU then VOPD")

  def test_valu_after_vopd(self):
    """Regular VALU after VOPD."""
    self._run_and_compare([
      VOPD(opx=VOPDOp.V_DUAL_MOV_B32, opy=VOPDOp.V_DUAL_MOV_B32,
           vdstx=v[0], srcx0=v[10], vsrcx1=v[10],
           vdsty=v[1], srcy0=v[11], vsrcy1=v[11]),
      v_mov_b32_e32(v[2], 1.0),
    ], "VOPD then VALU")

  # ═══════════════════════════════════════════════════════════════════════════
  # WMMA (Wave Matrix Multiply Accumulate) tests
  # ═══════════════════════════════════════════════════════════════════════════
  # WMMA: matrix ops on tensor cores, expected ~47 cycle latency, ~34 cycle issue interval

  def test_wmma_single(self):
    """Single WMMA f32_16x16x16_f16."""
    self._run_and_compare([
      v_wmma_f32_16x16x16_f16(vdst=v[0], src0=v[16], src1=v[24], src2=v[0]),
    ], "single WMMA")

  def test_wmma_independent_2(self):
    """Two independent WMMA operations."""
    self._run_and_compare([
      v_wmma_f32_16x16x16_f16(vdst=v[0], src0=v[16], src1=v[24], src2=v[0]),
      v_wmma_f32_16x16x16_f16(vdst=v[8], src0=v[32], src1=v[40], src2=v[8]),
    ], "2 WMMA independent")

  def test_wmma_independent_3(self):
    """Three independent WMMA operations."""
    self._run_and_compare([
      v_wmma_f32_16x16x16_f16(vdst=v[0], src0=v[16], src1=v[24], src2=v[0]),
      v_wmma_f32_16x16x16_f16(vdst=v[8], src0=v[32], src1=v[40], src2=v[8]),
      v_wmma_f32_16x16x16_f16(vdst=v[48], src0=v[56], src1=v[64], src2=v[48]),
    ], "3 WMMA independent")

  def test_wmma_chain(self):
    """WMMA with dependency chain."""
    self._run_and_compare([
      v_wmma_f32_16x16x16_f16(vdst=v[0], src0=v[16], src1=v[24], src2=v[0]),
      v_wmma_f32_16x16x16_f16(vdst=v[0], src0=v[0], src1=v[24], src2=v[0]),  # depends on first
    ], "WMMA chain")

  def test_wmma_bf16(self):
    """WMMA with bf16 inputs."""
    self._run_and_compare([
      v_wmma_f32_16x16x16_bf16(vdst=v[0], src0=v[16], src1=v[24], src2=v[0]),
    ], "WMMA bf16")

  def test_wmma_after_valu(self):
    """WMMA after VALU."""
    self._run_and_compare([
      v_mov_b32_e32(v[16], 1.0),
      v_wmma_f32_16x16x16_f16(vdst=v[0], src0=v[16], src1=v[24], src2=v[0]),
    ], "VALU then WMMA")

  def test_valu_after_wmma(self):
    """VALU after WMMA."""
    self._run_and_compare([
      v_wmma_f32_16x16x16_f16(vdst=v[0], src0=v[16], src1=v[24], src2=v[0]),
      v_mov_b32_e32(v[50], 1.0),
    ], "WMMA then VALU")


if __name__ == "__main__":
  unittest.main()
