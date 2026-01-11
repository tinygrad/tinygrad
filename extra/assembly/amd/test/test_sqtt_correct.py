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
from extra.assembly.amd.autogen.rdna3.ins import v_mov_b32_e32, v_add_f32_e32, s_nop, s_endpgm, s_delay_alu
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

# ************************************ tests ************************************

class TestVALUChains(unittest.TestCase):
  """VALU dependency chains."""
  def _chain(self, n, expected_issue, expected_exec):
    instrs = [v_mov_b32_e32(v[0], 1.0)] + [v_add_f32_e32(v[i], v[i-1], v[i-1]) for i in range(1, n)]
    issue, execd = get_deltas(instrs)
    self.assertEqual(issue[:n-1], expected_issue)
    self.assertEqual(execd, expected_exec)

  def test_chain_2(self):
    self._chain(2, [1], [6])
  def test_chain_3(self):
    self._chain(3, [1, 1], [6, 5])
  def test_chain_4(self):
    self._chain(4, [1, 1, 1], [6, 5, 5])
  def test_chain_5(self):
    self._chain(5, [1, 1, 1, 1], [6, 5, 5, 9])
  def test_chain_6(self):
    self._chain(6, [1, 1, 1, 1, 1], [6, 5, 5, 9, 9])
  def test_chain_7(self):
    self._chain(7, [1, 1, 1, 1, 1, 1], [6, 5, 5, 5, 9, 9])
  def test_chain_8(self):
    self._chain(8, [1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 5, 9, 9, 9])
  def test_chain_12(self):
    self._chain(12, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9])
  def test_chain_14(self):
    self._chain(14, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9])
  def test_chain_15(self):  # issue stalls start here
    self._chain(15, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3], [6, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9])
  def test_chain_16(self):
    self._chain(16, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5], [6, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9])
  def test_chain_18(self):
    self._chain(18, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 5, 5], [6, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9])
  def test_chain_20(self):
    self._chain(20, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 5, 5, 5, 5, 5], [6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9])


class TestVALUChainsWithNops(unittest.TestCase):
  """VALU dependency chains with nops before to isolate warmup effects."""
  def _chain_with_nops(self, n, num_nops=5):
    instrs = [s_nop(0) for _ in range(num_nops)]
    instrs += [v_mov_b32_e32(v[0], 1.0)] + [v_add_f32_e32(v[i], v[i-1], v[i-1]) for i in range(1, n)]
    issue, execd = get_deltas(instrs)
    return execd

  def test_chain_2_with_nops(self): self.assertEqual(self._chain_with_nops(2), [6])
  def test_chain_3_with_nops(self): self.assertEqual(self._chain_with_nops(3), [6, 5])
  def test_chain_4_with_nops(self): self.assertEqual(self._chain_with_nops(4), [6, 5, 5])
  def test_chain_5_with_nops(self): self.assertEqual(self._chain_with_nops(5), [6, 5, 5, 9])
  def test_chain_6_with_nops(self): self.assertEqual(self._chain_with_nops(6), [6, 5, 5, 9, 9])
  def test_chain_7_with_nops(self): self.assertEqual(self._chain_with_nops(7), [6, 5, 5, 5, 9, 9])
  def test_chain_8_with_nops(self): self.assertEqual(self._chain_with_nops(8), [6, 5, 5, 5, 9, 9, 9])


class TestVALUIndependent(unittest.TestCase):
  """Independent VALU instructions."""
  def _ind(self, n, expected_exec):
    instrs = [v_mov_b32_e32(v[i], float(i)) for i in range(n)]
    issue, execd = get_deltas(instrs)
    self.assertEqual(issue[:n-1], [1]*(n-1))
    self.assertEqual(execd, expected_exec)

  def test_ind_2(self): self._ind(2, [1])
  def test_ind_3(self): self._ind(3, [1, 1])
  def test_ind_4(self): self._ind(4, [1, 1, 1])
  def test_ind_5(self): self._ind(5, [1, 1, 1, 1])
  def test_ind_6(self): self._ind(6, [1, 1, 1, 1, 1])
  def test_ind_7(self): self._ind(7, [1, 1, 1, 1, 1, 1])
  def test_ind_8(self): self._ind(8, [1, 1, 1, 1, 1, 1, 1])


class TestForwardingGap(unittest.TestCase):
  """Producer + N independent instructions + consumer - tests forwarding window."""
  def _exec_deltas(self, n_gap):
    instrs = [v_mov_b32_e32(v[0], 1.0)]
    instrs += [v_mov_b32_e32(v[10+i], float(i)) for i in range(n_gap)]
    instrs += [v_add_f32_e32(v[1], v[0], v[0])]
    _, execd = get_deltas(instrs)
    return execd

  def test_gap0(self): self.assertEqual(self._exec_deltas(0), [6])
  def test_gap1(self): self.assertEqual(self._exec_deltas(1), [1, 5])
  def test_gap2(self): self.assertEqual(self._exec_deltas(2), [1, 1, 4])
  def test_gap3(self): self.assertEqual(self._exec_deltas(3), [1, 1, 1, 3])
  def test_gap4(self): self.assertEqual(self._exec_deltas(4), [1, 1, 1, 1, 3])
  def test_gap5(self): self.assertEqual(self._exec_deltas(5), [1, 1, 1, 1, 1, 4])  # anomaly
  def test_gap6(self): self.assertEqual(self._exec_deltas(6), [1, 1, 1, 1, 1, 1, 3])
  def test_gap7(self): self.assertEqual(self._exec_deltas(7), [1, 1, 1, 1, 1, 1, 1, 3])
  def test_gap8(self): self.assertEqual(self._exec_deltas(8), [1, 1, 1, 1, 1, 1, 1, 1, 3])
  def test_gap9(self): self.assertEqual(self._exec_deltas(9), [1, 1, 1, 1, 1, 1, 1, 1, 1, 3])


class TestVALULatency(unittest.TestCase):
  """VALU latency depends on VGPR source reads.
  6 cycles: no VGPR source (constant only), stays 6 regardless of warmup
  8-11 cycles: VGPR source read, decreases with warmup (11->10->9->8)
  s_nop(0) after VALU immediately drops VGPR read latency to 8
  Anomalies:
    - 7 consecutive VALUs (no s_nop) causes +1 cycle penalty
    - n=0 or n=3 const VALUs + nop + vgpr = 9 cycles (not 8)
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

  # s_nop + vgpr read: latency depends on # of const VALUs before nop
  def _n_const_nop_vgpr(self, n):
    """N const VALUs + s_nop(0) + vgpr read."""
    instrs = [v_mov_b32_e32(v[i], float(i)) for i in range(n)]
    instrs += [s_nop(0)]
    instrs += [v_mov_b32_e32(v[10], v[99])]
    return self._get_latency(instrs)
  def test_0_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(0), 9)
  def test_1_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(1), 8)
  def test_2_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(2), 8)
  def test_3_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(3), 9)  # anomaly
  def test_4_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(4), 8)
  def test_5_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(5), 8)
  def test_6_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(6), 8)
  def test_7_const_nop_vgpr(self): self.assertEqual(self._n_const_nop_vgpr(7), 8)


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
  """Single s_nop delay between two independent v_movs.
  s_nop(n) delays n+1 cycles, plus +4 extra for n in [11, 22].
  Exec delta = n + 4 (baseline) + 4 (if 11 <= n <= 22)."""
  def _test(self, n, expected):
    _, execd = get_deltas([v_mov_b32_e32(v[0], 1.0), s_nop(n), v_mov_b32_e32(v[1], 2.0)])
    self.assertEqual(execd, [expected])

  def test_snop_0(self):  self._test(0, 4)
  def test_snop_1(self):  self._test(1, 5)
  def test_snop_2(self):  self._test(2, 6)
  def test_snop_3(self):  self._test(3, 7)
  def test_snop_4(self):  self._test(4, 8)
  def test_snop_5(self):  self._test(5, 9)
  def test_snop_6(self):  self._test(6, 10)
  def test_snop_7(self):  self._test(7, 11)
  def test_snop_10(self): self._test(10, 14)
  def test_snop_11(self): self._test(11, 19)  # +4 extra starts here
  def test_snop_15(self): self._test(15, 23)
  def test_snop_22(self): self._test(22, 30)  # +4 extra ends here
  def test_snop_23(self): self._test(23, 27)
  def test_snop_31(self): self._test(31, 35)
  def test_snop_32(self): self._test(32, 36)
  def test_snop_63(self): self._test(63, 67)


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


class TestDelayALU(unittest.TestCase):
  """s_delay_alu behavior - helps understand hardware pipeline latencies.

  s_delay_alu(simm16) where simm16 encodes:
    instid0[3:0] = dependency on VALU N instructions back (1-4), 0=none
    skip[6:4] = skip count for second dependency
    instid1[10:7] = second dependency

  Key insight: s_delay_alu tells hardware to wait for a previous VALU to complete.
  The hardware determines how many cycles to stall based on pipeline state.
  """
  def _exec_delta(self, instrs):
    """Return exec delta for last instruction."""
    _, execd = get_deltas(instrs)
    return execd[-1] if execd else None

  # Direct dependency (producer -> consumer), instid0=1 means "wait for VALU 1 back"
  def test_direct_no_delay(self):
    # Without s_delay_alu: 6 cycles
    self.assertEqual(self._exec_delta([v_mov_b32_e32(v[0], 1.0), v_add_f32_e32(v[1], v[0], v[0])]), 6)

  def test_direct_delay1(self):
    # With s_delay_alu(instid0=1): 7 cycles (+1 from the delay instruction)
    self.assertEqual(self._exec_delta([v_mov_b32_e32(v[0], 1.0), s_delay_alu(simm16=1), v_add_f32_e32(v[1], v[0], v[0])]), 7)

  def test_direct_delay2(self):
    # instid0=2 doesn't apply (only 1 VALU back), so no extra delay
    self.assertEqual(self._exec_delta([v_mov_b32_e32(v[0], 1.0), s_delay_alu(simm16=2), v_add_f32_e32(v[1], v[0], v[0])]), 6)

  def test_direct_delay3(self):
    self.assertEqual(self._exec_delta([v_mov_b32_e32(v[0], 1.0), s_delay_alu(simm16=3), v_add_f32_e32(v[1], v[0], v[0])]), 6)

  def test_direct_delay4(self):
    self.assertEqual(self._exec_delta([v_mov_b32_e32(v[0], 1.0), s_delay_alu(simm16=4), v_add_f32_e32(v[1], v[0], v[0])]), 6)

  # With 1 independent instruction between producer and consumer
  def test_gap1_delay1(self):
    # instid0=1 waits for the independent instruction (not the producer)
    instrs = [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[5], 5.0), s_delay_alu(simm16=1), v_add_f32_e32(v[1], v[0], v[0])]
    self.assertEqual(self._exec_delta(instrs), 8)

  def test_gap1_delay2(self):
    # instid0=2 waits for the producer (2 VALUs back)
    instrs = [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[5], 5.0), s_delay_alu(simm16=2), v_add_f32_e32(v[1], v[0], v[0])]
    self.assertEqual(self._exec_delta(instrs), 6)

  def test_gap1_delay3(self):
    # instid0=3 doesn't apply (only 2 VALUs back)
    instrs = [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[5], 5.0), s_delay_alu(simm16=3), v_add_f32_e32(v[1], v[0], v[0])]
    self.assertEqual(self._exec_delta(instrs), 5)

  # With 2 independent instructions between
  def test_gap2_delay1(self):
    instrs = [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[5], 5.0), v_mov_b32_e32(v[6], 6.0),
              s_delay_alu(simm16=1), v_add_f32_e32(v[1], v[0], v[0])]
    self.assertEqual(self._exec_delta(instrs), 7)

  def test_gap2_delay2(self):
    instrs = [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[5], 5.0), v_mov_b32_e32(v[6], 6.0),
              s_delay_alu(simm16=2), v_add_f32_e32(v[1], v[0], v[0])]
    self.assertEqual(self._exec_delta(instrs), 7)

  def test_gap2_delay3(self):
    # instid0=3 waits for the producer (3 VALUs back)
    instrs = [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[5], 5.0), v_mov_b32_e32(v[6], 6.0),
              s_delay_alu(simm16=3), v_add_f32_e32(v[1], v[0], v[0])]
    self.assertEqual(self._exec_delta(instrs), 5)

  def test_gap2_delay4(self):
    instrs = [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[5], 5.0), v_mov_b32_e32(v[6], 6.0),
              s_delay_alu(simm16=4), v_add_f32_e32(v[1], v[0], v[0])]
    self.assertEqual(self._exec_delta(instrs), 4)


if __name__ == "__main__":
  unittest.main()
