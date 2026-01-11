#!/usr/bin/env python3
"""Tests for SQTT emulator correctness against known hardware patterns.

NOTE: This file only tests NOP and VALU behavior. For WMMA/DP/trans tests,
see test_sqtt_compare.py.

Run emulator tests: PYTHONPATH="." python3 extra/assembly/amd/test/test_sqtt_correct.py
Run hardware tests: SQTT_HW=1 PYTHONPATH="." python3 extra/assembly/amd/test/test_sqtt_correct.py
"""
import os
import unittest

USE_HW = os.environ.get("SQTT_HW", "0") == "1"

if USE_HW:
  os.environ["SQTT"] = "1"
  os.environ["PROFILE"] = "1"
  os.environ["SQTT_LIMIT_SE"] = "2"
  os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"

from extra.assembly.amd.emu import SQTTState, decode_program, exec_wave, WaveState, LDSMem
from extra.assembly.amd.sqtt import WAVESTART, WAVEEND
from extra.assembly.amd.autogen.rdna3.ins import v_mov_b32_e32, v_add_f32_e32, s_nop, s_endpgm
from extra.assembly.amd.dsl import v

def assemble(instructions: list) -> bytes:
  return b''.join(inst.to_bytes() for inst in instructions)

def wrap_with_nops(instructions: list, nops=16) -> list:
  return instructions + [s_nop(0)]*nops + [s_endpgm()]

def get_wave_packets(packets: list) -> list:
  result, in_wave = [], False
  for p in packets:
    if isinstance(p, WAVESTART) and p.simd == 0:
      in_wave, result = True, [p]
    elif in_wave:
      result.append(p)
      if isinstance(p, WAVEEND): break
  return result

def get_timing_deltas(packets: list) -> list[tuple[str, int]]:
  skip_types = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3", "REG"}
  filtered = [p for p in packets if type(p).__name__ not in skip_types]
  if not filtered: return []
  result = [(type(filtered[0]).__name__, 0)]
  for i in range(1, len(filtered)):
    result.append((type(filtered[i]).__name__, filtered[i]._time - filtered[i-1]._time))
  return result

def run_emulator(instructions: list) -> list:
  code = assemble(instructions)
  program = decode_program(code)
  st = WaveState()
  st.exec_mask = (1 << 32) - 1
  lds = LDSMem(bytearray(65536))
  trace = SQTTState(wave_id=0, simd=0, cu=0)
  exec_wave(program, st, lds, 32, trace)
  return get_wave_packets(trace.packets)

def get_all_waves(packets: list) -> list[list]:
  """Extract all WAVESTART..WAVEEND ranges on simd 0."""
  waves, in_wave, current = [], False, []
  for p in packets:
    if isinstance(p, WAVESTART) and p.simd == 0:
      in_wave, current = True, [p]
    elif in_wave:
      current.append(p)
      if isinstance(p, WAVEEND):
        waves.append(current)
        in_wave, current = False, []
  return waves

def run_hardware(instructions: list) -> list:
  from extra.assembly.amd.test.test_sqtt_hw import compile_asm_sqtt, run_prg_sqtt_batch
  from extra.assembly.amd.sqtt import decode
  from collections import Counter

  prg = compile_asm_sqtt(instructions, alu_only=True)

  for _ in range(10):
    blobs = run_prg_sqtt_batch(prg, n_runs=200)
    # Extract all waves from all blobs
    traces = []
    for blob in blobs:
      traces.extend(get_all_waves(decode(blob)))
    if not traces:
      continue
    # Find most common pattern
    delta_sets = [tuple(get_timing_deltas(t)) for t in traces]
    most_common = Counter(delta_sets).most_common(1)[0][0]
    for t in traces:
      if tuple(get_timing_deltas(t)) == most_common:
        return t
  return []

def run_sqtt(instructions: list, nops: int = 16) -> list:
  instructions = wrap_with_nops(instructions, nops=nops)
  return run_hardware(instructions) if USE_HW else run_emulator(instructions)

def get_deltas(instructions: list) -> tuple[list[int], list[int]]:
  """Run and return (issue deltas, exec deltas).
  Issue = IMMEDIATE + VALUINST, Exec = ALUEXEC.
  Deltas are between consecutive packets of same stream."""
  deltas = get_timing_deltas(run_sqtt(instructions))
  time = 0
  issue_times, exec_times = [], []
  for ptype, delta in deltas:
    time += delta
    if ptype in ('IMMEDIATE', 'VALUINST'):
      issue_times.append(time)
    elif ptype == 'ALUEXEC':
      exec_times.append(time)
  issue = [issue_times[i] - issue_times[i-1] for i in range(1, len(issue_times))]
  execd = [exec_times[i] - exec_times[i-1] for i in range(1, len(exec_times))]
  return issue, execd

# Hardware ALUEXEC delta patterns:
#   chain: forwarding (6,5,5...) then stalls when exhausted (9,9,9...)
#   ind: no dependencies, exec follows issue by 1 cycle
#   snop: n+4 baseline, but +4 extra for 11 <= n <= 22
CHAIN_ISSUE = {
  2:  [1],
  3:  [1, 1],
  4:  [1, 1, 1],
  5:  [1, 1, 1, 1],
  6:  [1, 1, 1, 1, 1],
  7:  [1, 1, 1, 1, 1, 1],
  8:  [1, 1, 1, 1, 1, 1, 1],
  12: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  14: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  15: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],              # issue stalls start here
  16: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5],
  18: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 5, 5],
  20: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 5, 5, 5, 5],
}
CHAIN_EXEC = {
  2:  [6],
  3:  [6, 5],
  4:  [6, 5, 5],
  5:  [6, 5, 5, 9],
  6:  [6, 5, 5, 9, 9],
  7:  [6, 5, 5, 5, 9, 9],
  8:  [6, 5, 5, 5, 9, 9, 9],
  12: [6, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9],
  14: [6, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9],
  15: [6, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9],
  16: [6, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9],
  18: [6, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9],
  20: [6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9],
}
IND_EXEC = {
  2: [1],
  3: [1, 1],
  4: [1, 1, 1],
  5: [1, 1, 1, 1],
  6: [1, 1, 1, 1, 1],
  7: [1, 1, 1, 1, 1, 1],
  8: [1, 1, 1, 1, 1, 1, 1],
}
SNOP_EXEC = {
  0: 4,   1: 5,   2: 6,   3: 7,   4: 8,   5: 9,   6: 10,  7: 11,  8: 12,  9: 13,  10: 14,
  11: 19, 12: 20, 13: 21, 14: 22, 15: 23, 16: 24, 17: 25, 18: 26, 19: 27, 20: 28, 21: 29, 22: 30,  # +4 extra
  23: 27, 24: 28, 25: 29, 26: 30, 27: 31, 28: 32, 29: 33, 30: 34, 31: 35,
  32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43,
  40: 44, 41: 45, 42: 46, 43: 47, 44: 48, 45: 49, 46: 50, 47: 51,
  48: 52, 49: 53, 50: 54, 51: 55, 52: 56, 53: 57, 54: 58, 55: 59,
  56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65, 62: 66, 63: 67,
}

class TestVALUChains(unittest.TestCase):
  """VALU dependency chains."""
  def _chain(self, n):
    instrs = [v_mov_b32_e32(v[0], 1.0)] + [v_add_f32_e32(v[i], v[i-1], v[i-1]) for i in range(1, n)]
    issue, execd = get_deltas(instrs)
    self.assertEqual(issue[:n-1], CHAIN_ISSUE[n])
    self.assertEqual(execd, CHAIN_EXEC[n])

  def test_chain_2(self): self._chain(2)
  def test_chain_3(self): self._chain(3)
  def test_chain_4(self): self._chain(4)
  def test_chain_5(self): self._chain(5)
  def test_chain_6(self): self._chain(6)
  def test_chain_7(self): self._chain(7)
  def test_chain_8(self): self._chain(8)
  def test_chain_12(self): self._chain(12)
  def test_chain_14(self): self._chain(14)
  def test_chain_15(self): self._chain(15)  # issue stalls start here
  def test_chain_16(self): self._chain(16)
  def test_chain_18(self): self._chain(18)
  def test_chain_20(self): self._chain(20)


class TestVALUIndependent(unittest.TestCase):
  """Independent VALU instructions."""
  def _ind(self, n):
    instrs = [v_mov_b32_e32(v[i], float(i)) for i in range(n)]
    issue, execd = get_deltas(instrs)
    self.assertEqual(issue[:n-1], [1]*(n-1))
    self.assertEqual(execd, IND_EXEC[n])

  def test_ind_2(self): self._ind(2)
  def test_ind_3(self): self._ind(3)
  def test_ind_4(self): self._ind(4)
  def test_ind_5(self): self._ind(5)
  def test_ind_6(self): self._ind(6)
  def test_ind_7(self): self._ind(7)
  def test_ind_8(self): self._ind(8)


class TestForwardingGap(unittest.TestCase):
  """Producer + N independent instructions + consumer - tests forwarding window."""
  def _last_exec_delta(self, n_gap):
    instrs = [v_mov_b32_e32(v[0], 1.0)]
    instrs += [v_mov_b32_e32(v[10+i], float(i)) for i in range(n_gap)]
    instrs += [v_add_f32_e32(v[1], v[0], v[0])]
    _, execd = get_deltas(instrs)
    return execd[-1]

  def test_gap0(self): self.assertEqual(self._last_exec_delta(0), 6)
  def test_gap1(self): self.assertEqual(self._last_exec_delta(1), 5)
  def test_gap2(self): self.assertEqual(self._last_exec_delta(2), 4)
  def test_gap3(self): self.assertEqual(self._last_exec_delta(3), 3)
  def test_gap4(self): self.assertEqual(self._last_exec_delta(4), 3)
  def test_gap5(self): self.assertEqual(self._last_exec_delta(5), 4)  # anomaly
  def test_gap6(self): self.assertEqual(self._last_exec_delta(6), 3)
  def test_gap7(self): self.assertEqual(self._last_exec_delta(7), 3)
  def test_gap8(self): self.assertEqual(self._last_exec_delta(8), 3)
  def test_gap9(self): self.assertEqual(self._last_exec_delta(9), 3)
  def test_gap10(self): self.assertEqual(self._last_exec_delta(10), 3)
  def test_gap11(self): self.assertEqual(self._last_exec_delta(11), 3)
  def test_gap12(self): self.assertEqual(self._last_exec_delta(12), 3)


class TestVALULatency(unittest.TestCase):
  """VALU latency depends on VGPR source reads.
  6 cycles: no VGPR source (constant only), stays 6 regardless of warmup
  8-11 cycles: VGPR source read, decreases with warmup (11->10->9->8)
  s_nop(0) after VALU immediately drops VGPR read latency to 8
  """
  def _get_latency(self, instrs):
    if not isinstance(instrs, list): instrs = [instrs]
    packets = run_sqtt(instrs)
    deltas = get_timing_deltas(packets)
    time, valu_times, exec_times = 0, [], []
    for ptype, delta in deltas:
      time += delta
      if ptype == 'VALUINST': valu_times.append(time)
      if ptype == 'ALUEXEC': exec_times.append(time)
    return exec_times[-1] - valu_times[-1] if valu_times and exec_times else None

  # 6-cycle latency: no VGPR source (constant), always 6
  def test_const_single(self): self.assertEqual(self._get_latency(v_mov_b32_e32(v[0], 1.0)), 6)
  def test_const_literal(self): self.assertEqual(self._get_latency(v_mov_b32_e32(v[0], 565.0)), 6)
  def test_const_after_const(self): self.assertEqual(self._get_latency([v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[1], 2.0)]), 6)
  def test_const_after_nop(self): self.assertEqual(self._get_latency([v_mov_b32_e32(v[0], 1.0), s_nop(0), v_mov_b32_e32(v[1], 2.0)]), 6)

  # VGPR read latency: cold start = 9
  def test_vgpr_cold(self): self.assertEqual(self._get_latency(v_mov_b32_e32(v[0], v[1])), 9)

  # VGPR read latency: warmup decreases 11->10->9->8
  def _vgpr_after_n_const(self, n):
    return self._get_latency([v_mov_b32_e32(v[i], float(i)) for i in range(n)] + [v_mov_b32_e32(v[10], v[99])])
  def test_vgpr_after_1_const(self): self.assertEqual(self._vgpr_after_n_const(1), 11)
  def test_vgpr_after_2_const(self): self.assertEqual(self._vgpr_after_n_const(2), 10)
  def test_vgpr_after_3_const(self): self.assertEqual(self._vgpr_after_n_const(3), 9)
  def test_vgpr_after_4_const(self): self.assertEqual(self._vgpr_after_n_const(4), 8)
  def test_vgpr_after_5_const(self): self.assertEqual(self._vgpr_after_n_const(5), 8)
  def test_vgpr_after_6_const(self): self.assertEqual(self._vgpr_after_n_const(6), 9)  # anomaly
  def test_vgpr_after_7_const(self): self.assertEqual(self._vgpr_after_n_const(7), 8)
  def test_vgpr_after_8_const(self): self.assertEqual(self._vgpr_after_n_const(8), 8)

  # s_nop(0) immediately drops VGPR read latency to 8
  def test_vgpr_nop_warmup(self): self.assertEqual(self._get_latency([v_mov_b32_e32(v[0], 1.0), s_nop(0), v_mov_b32_e32(v[1], v[99])]), 8)


class TestChainWithNop(unittest.TestCase):
  """Dependency chain with s_nop between instructions."""
  def _test(self, nop_val, expected_issue, expected_exec):
    issue, execd = get_deltas([v_mov_b32_e32(v[0], 1.0), s_nop(nop_val), v_add_f32_e32(v[1], v[0], v[0])])
    self.assertEqual(issue[:2], expected_issue)
    self.assertEqual(execd, expected_exec)

  def test_nop0(self): self._test(0, [3, 1], [6])
  def test_nop1(self): self._test(1, [4, 1], [7])
  def test_nop2(self): self._test(2, [5, 1], [9])
  def test_nop3(self): self._test(3, [6, 1], [9])
  def test_nop4(self): self._test(4, [11, 1], [10])
  def test_nop5(self): self._test(5, [12, 1], [11])


class TestIndWithNop(unittest.TestCase):
  """Independent instructions with s_nop between."""
  def _test(self, nop_val, expected_issue, expected_exec):
    issue, execd = get_deltas([v_mov_b32_e32(v[0], 1.0), s_nop(nop_val), v_mov_b32_e32(v[1], 2.0)])
    self.assertEqual(issue[:2], expected_issue)
    self.assertEqual(execd, expected_exec)

  def test_nop0(self): self._test(0, [3, 1], [4])
  def test_nop1(self): self._test(1, [4, 1], [5])
  def test_nop3(self): self._test(3, [6, 1], [7])
  def test_nop4(self): self._test(4, [11, 1], [8])
  def test_nop5(self): self._test(5, [12, 1], [9])


class TestChain3NopMid(unittest.TestCase):
  """3-instruction chain with s_nop in middle."""
  def _test(self, nop_val, expected_issue, expected_exec):
    issue, execd = get_deltas([
      v_mov_b32_e32(v[0], 1.0), v_add_f32_e32(v[1], v[0], v[0]),
      s_nop(nop_val), v_add_f32_e32(v[2], v[1], v[1])])
    self.assertEqual(issue[:3], expected_issue)
    self.assertEqual(execd, expected_exec)

  def test_nop0(self): self._test(0, [1, 3, 1], [6, 5])
  def test_nop1(self): self._test(1, [1, 4, 1], [6, 5])
  def test_nop2(self): self._test(2, [1, 5, 1], [6, 5])
  def test_nop3(self): self._test(3, [1, 10, 1], [6, 5])


class TestInd3NopMid(unittest.TestCase):
  """3 independent instructions with s_nop in middle."""
  def _test(self, nop_val, expected_issue, expected_exec):
    issue, execd = get_deltas([
      v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[1], 2.0),
      s_nop(nop_val), v_mov_b32_e32(v[2], 3.0)])
    self.assertEqual(issue[:3], expected_issue)
    self.assertEqual(execd, expected_exec)

  def test_nop0(self): self._test(0, [1, 3, 1], [1, 4])
  def test_nop1(self): self._test(1, [1, 4, 1], [1, 5])
  def test_nop2(self): self._test(2, [1, 5, 1], [1, 6])
  def test_nop3(self): self._test(3, [1, 10, 1], [1, 7])


class TestSNopDelay(unittest.TestCase):
  """Single s_nop delay between two independent v_movs."""
  def _test(self, n):
    _, execd = get_deltas([v_mov_b32_e32(v[0], 1.0), s_nop(n), v_mov_b32_e32(v[1], 2.0)])
    self.assertEqual(execd, [SNOP_EXEC[n]])

  def test_snop_0(self): self._test(0)
  def test_snop_1(self): self._test(1)
  def test_snop_2(self): self._test(2)
  def test_snop_3(self): self._test(3)
  def test_snop_4(self): self._test(4)
  def test_snop_5(self): self._test(5)
  def test_snop_6(self): self._test(6)
  def test_snop_7(self): self._test(7)
  def test_snop_10(self): self._test(10)
  def test_snop_11(self): self._test(11)  # +4 extra starts here
  def test_snop_15(self): self._test(15)
  def test_snop_22(self): self._test(22)  # +4 extra ends here
  def test_snop_23(self): self._test(23)
  def test_snop_31(self): self._test(31)
  def test_snop_32(self): self._test(32)
  def test_snop_63(self): self._test(63)


class TestVALUExecWithNop(unittest.TestCase):
  """Single VALU followed by s_nop - measures VALUINST to ALUEXEC delay."""
  def _get_delay(self, instrs, nops=16):
    deltas = get_timing_deltas(run_sqtt(instrs, nops=nops))
    time, valu_time, exec_time = 0, None, None
    for ptype, delta in deltas:
      time += delta
      if ptype == 'VALUINST' and valu_time is None: valu_time = time
      if ptype == 'ALUEXEC' and exec_time is None: exec_time = time
    return exec_time - valu_time

  # Boundary: s_nop(0-3) = 6 cycles, s_nop(4+) = 10 cycles
  def test_nop0(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(0)]), 6)
  def test_nop1(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(1)]), 6)
  def test_nop2(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(2)]), 6)
  def test_nop3(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(3)]), 6)
  def test_nop4(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(4)]), 10)
  def test_nop5(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(5)]), 10)
  def test_nop6(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(6)]), 10)
  def test_nop7(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(7)]), 10)
  def test_nop8(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(8)]), 10)
  def test_nop9(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(9)]), 10)
  def test_nop10(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(10)]), 10)
  # No nop = slow path, one s_nop(0) padding = fast path
  def test_no_padding(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0)], nops=0), 10)
  def test_one_padding(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0)], nops=1), 6)
  # Multiple s_nop(0)s don't accumulate - still fast path
  def test_nop0_x2(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(0), s_nop(0)]), 6)
  # First nop determines path: s_nop(0) then s_nop(4) = fast, s_nop(4) then s_nop(0) = slow
  def test_nop0_nop4(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(0), s_nop(4)]), 6)
  def test_nop4_nop0(self): self.assertEqual(self._get_delay([v_mov_b32_e32(v[0], 1.0), s_nop(4), s_nop(0)]), 10)


if __name__ == "__main__":
  unittest.main()
