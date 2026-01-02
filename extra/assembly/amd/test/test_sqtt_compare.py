#!/usr/bin/env python3
"""Tests comparing hardware SQTT traces to emulator SQTT output.

Run with: python -m pytest extra/assembly/amd/test/test_sqtt_compare.py -v
Requires AMD GPU with SQTT support.
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_LIMIT_SE"] = "2"
os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"  # exclude WAVERDY, REG, EVENT, UTILCTR, WAVEALLOC, PERF

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
from extra.assembly.amd.autogen.rdna3.ins import v_mov_b32_e32, v_add_f32_e32, v_rcp_f32_e32, v_sqrt_f32_e32, v_exp_f32_e32, s_mov_b32, s_add_u32
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
  return [(type(filtered[0]).__name__, 0)] + [(type(filtered[i]).__name__, filtered[i]._time - filtered[i-1]._time) for i in range(1, len(filtered))]

@unittest.skipIf(not hasattr(dev, 'profile_events'), "AMD device required")
class TestEmulatorSQTT(unittest.TestCase):
  """Tests comparing emulator SQTT to hardware SQTT."""

  def _run_and_compare(self, instructions: list, name: str = "", n_runs: int = 100, min_identical: int = 25, max_attempts: int = 10):
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

      # Check if we have enough identical patterns
      skip_types = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3"}
      def wave_deltas(pkts):
        filtered = [p for p in pkts if type(p).__name__ not in skip_types]
        if not filtered: return []
        return [(type(filtered[0]).__name__, 0)] + [(type(filtered[i]).__name__, filtered[i]._time - filtered[i-1]._time) for i in range(1, len(filtered))]

      hw_delta_sets = [wave_deltas(t) for t in hw_traces]
      pattern_counts = Counter(tuple(d) for d in hw_delta_sets)

      if pattern_counts and pattern_counts.most_common(1)[0][1] >= min_identical:
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

  def _test_valu_independent_n(self, n: int):
    """VALU instructions with no dependencies."""
    self._run_and_compare([
      v_mov_b32_e32(v[i], float(i)) for i in range(n)
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


if __name__ == "__main__":
  unittest.main()
