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
from extra.assembly.amd.autogen.rdna3.ins import v_mov_b32_e32, v_add_f32_e32, s_mov_b32, s_add_u32, s_endpgm
from extra.assembly.amd.dsl import v, s
from extra.assembly.amd.test.test_sqtt_hw import run_asm_sqtt, decode_all_blobs, get_wave_packets, format_packet

def assemble(instructions: list) -> bytes:
  return b''.join(inst.to_bytes() for inst in instructions)

def run_emulator_sqtt(instructions: list) -> list[PacketType]:
  """Run instructions through emulator and return SQTT packets."""
  code = assemble(instructions + [s_endpgm()])
  program = decode_program(code)

  sqtt = SQTTState(wave_id=0, simd=0, cu=0)
  sqtt.emit_wavestart()
  # cycle stays at 0 - emulator doesn't model startup jitter

  for pc, inst in sorted(program.items()):
    if inst.op_name == 'S_ENDPGM': break
    sqtt.trace_inst(inst)

  sqtt.finalize()
  return sqtt.packets

def compare_sqtt_structure(hw_packets: list, emu_packets: list) -> list[str]:
  """Compare SQTT packet structure (types and order), return list of differences."""
  diffs = []

  # Filter to instruction-related packets only
  hw_filtered = [p for p in hw_packets if isinstance(p, (WAVESTART, WAVEEND, INST, VALUINST, ALUEXEC))]
  emu_filtered = [p for p in emu_packets if isinstance(p, (WAVESTART, WAVEEND, INST, VALUINST, ALUEXEC))]

  # Compare types
  hw_types = [type(p).__name__ for p in hw_filtered]
  emu_types = [type(p).__name__ for p in emu_filtered]

  if hw_types != emu_types:
    diffs.append(f"Packet types differ:\n  HW:  {hw_types}\n  EMU: {emu_types}")

  # Compare InstOp values for INST packets
  hw_ops = [(type(p).__name__, p.op if isinstance(p, INST) else None) for p in hw_filtered]
  emu_ops = [(type(p).__name__, p.op if isinstance(p, INST) else None) for p in emu_filtered]

  for i, (hw, emu) in enumerate(zip(hw_ops, emu_ops)):
    if hw != emu:
      diffs.append(f"Packet {i}: HW={hw}, EMU={emu}")

  return diffs


def filter_timing_packets(packets: list) -> list:
  """Filter to packets relevant for timing comparison."""
  return [p for p in packets if isinstance(p, (WAVESTART, WAVEEND, INST, VALUINST, ALUEXEC))]

def get_timing_deltas(packets: list) -> list[tuple[str, int]]:
  """Extract timing deltas between consecutive packets (includes startup as first delta)."""
  filtered = filter_timing_packets(packets)
  if not filtered: return []
  return [(type(filtered[i]).__name__, filtered[i]._time - filtered[i-1]._time) for i in range(1, len(filtered))]

def get_post_startup_deltas(packets: list) -> list[tuple[str, int]]:
  """Extract timing deltas after startup (skips WAVESTART->first instruction jitter)."""
  deltas = get_timing_deltas(packets)
  return deltas[1:] if deltas else []  # skip first delta which is startup jitter

@unittest.skipIf(not hasattr(dev, 'profile_events'), "AMD device required")
class TestEmulatorSQTT(unittest.TestCase):
  """Tests comparing emulator SQTT to hardware SQTT."""

  def _run_and_compare(self, instructions: list, name: str = "", n_traces: int = 20):
    """Run instructions on both hardware and emulator, compare SQTT structure."""
    from collections import Counter

    # Capture n_traces valid hardware traces on SIMD 0
    hw_traces = []
    attempts = 0
    max_attempts = 500
    while len(hw_traces) < n_traces and attempts < max_attempts:
      attempts += 1
      blobs = run_asm_sqtt(instructions, alu_only=True)
      packets = decode_all_blobs(blobs)
      wave_pkts = get_wave_packets(packets)
      ws = next((p for p in wave_pkts if isinstance(p, WAVESTART)), None)
      if ws and ws.simd == 0:
        hw_traces.append(wave_pkts)

    if not hw_traces:
      self.skipTest(f"Could not capture hardware trace on SIMD 0 after {max_attempts} attempts")

    # Run on emulator
    emu_packets = run_emulator_sqtt(instructions)
    emu_deltas = get_timing_deltas(emu_packets, skip_startup=True)
    emu_times = get_times(emu_packets)

    # Analyze hardware timing patterns (skip startup time which has jitter)
    hw_delta_sets = [get_timing_deltas(t, skip_startup=True) for t in hw_traces]
    hw_times_sets = [get_times(t) for t in hw_traces]
    pattern_counts = Counter(tuple(d) for d in hw_delta_sets)

    # Find most common pattern and a representative trace for it
    most_common_pattern = list(pattern_counts.most_common(1)[0][0]) if pattern_counts else []
    most_common_trace = next((t for t, d in zip(hw_traces, hw_delta_sets) if d == most_common_pattern), hw_traces[0])

    # Compute startup time jitter
    startup_times = [get_startup_time(t) for t in hw_traces]
    from collections import Counter as Ctr
    startup_counts = Ctr(startup_times)

    # Get most common startup time to use for emulator
    most_common_startup = startup_counts.most_common(1)[0][0] if startup_counts else 405

    if DEBUG >= 2:
      print(f"\n{'='*70}")
      print(f"TEST: {name} ({len(hw_traces)}/{n_traces} traces in {attempts} attempts)")
      print(f"{'='*70}")

      print(f"Jitter analysis (startup={most_common_startup}):")
      for pattern, count in pattern_counts.most_common():
        match = "✓ MATCH" if list(pattern) == emu_deltas else ""
        print(f"  {count:2d}x: {list(pattern)} {match}")

      # Pretty print most common HW trace and emulator trace side by side
      # Adjust emulator times to match HW startup
      hw_t0 = most_common_trace[0]._time
      emu_t0 = emu_packets[0]._time
      hw_startup = get_startup_time(most_common_trace)
      emu_startup = get_startup_time(emu_packets)
      
      hw_filtered = [p for p in most_common_trace if isinstance(p, (WAVESTART, WAVEEND, INST, VALUINST, ALUEXEC))]
      emu_filtered = [p for p in emu_packets if isinstance(p, (WAVESTART, WAVEEND, INST, VALUINST, ALUEXEC))]
      
      print(f"\n{'HW':^40} | {'Emulator':^40}")
      print(f"{'-'*40} | {'-'*40}")
      
      max_len = max(len(hw_filtered), len(emu_filtered))
      for i in range(max_len):
        if i < len(hw_filtered):
          hp = hw_filtered[i]
          hw_time = hp._time - hw_t0
          hw_str = f"{hw_time:6d}: {type(hp).__name__}"
        else:
          hw_str = ""
        
        if i < len(emu_filtered):
          ep = emu_filtered[i]
          # Adjust emulator time to match HW startup (WAVESTART stays at 0)
          if isinstance(ep, WAVESTART):
            emu_time = 0
          else:
            emu_time = ep._time - emu_t0 - emu_startup + hw_startup
          emu_str = f"{emu_time:6d}: {type(ep).__name__}"
        else:
          emu_str = ""
        
        print(f"{hw_str:40} | {emu_str:40}")

    # Assert emulator matches most common hardware pattern
    self.assertEqual(emu_deltas, most_common_pattern, f"{name}: emulator doesn't match most common HW pattern")

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

  def test_valu_independent(self):
    """VALU instructions with no dependencies."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 3.0),
    ], "3 VALU independent")

  def test_valu_chain(self):
    """VALU instructions with chain dependencies."""
    self._run_and_compare([
      v_mov_b32_e32(v[0], 1.0),
      v_add_f32_e32(v[1], v[0], v[0]),
      v_add_f32_e32(v[2], v[1], v[1]),
    ], "3 VALU chain")


if __name__ == "__main__":
  unittest.main()
