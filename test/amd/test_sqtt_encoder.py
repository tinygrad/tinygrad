#!/usr/bin/env python3
"""Tests for SQTT encoder: verifies the emulator produces correct SQTT traces for known kernels.

Run with: DEV=MOCK+AMD python -m pytest test/amd/test_sqtt_encoder.py -v
"""
import ctypes, unittest
from tinygrad.helpers import Context
from tinygrad.renderer.amd.sqtt import decode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, IMMEDIATE, VALUINST, InstOp, TS_DELTA_OR_MARK
from test.mockgpu.amd.sqtt_enc import make_encoder
from tinygrad.runtime.autogen.amd.rdna3.ins import *

def _run_kernel(instructions: list, lx=1, ly=1, lz=1, gx=1, gy=1, gz=1, args_ptr=0) -> bytes:
  """Assemble instructions, run on emulator with PROFILE=1, return the SQTT blob."""
  from test.mockgpu.amd.emu import run_asm, sqtt_traces
  code = b''.join(inst.to_bytes() for inst in instructions)
  buf = (ctypes.c_char * len(code))(*code)
  lib = ctypes.addressof(buf)
  sqtt_traces.clear()
  with Context(PROFILE=1):
    run_asm(lib, len(code), gx, gy, gz, lx, ly, lz, args_ptr)
  assert len(sqtt_traces) == 1, f"expected 1 trace, got {len(sqtt_traces)}"
  return sqtt_traces.pop()

class TestSQTTEncoder(unittest.TestCase):

  def test_simple_salu(self):
    """A simple s_mov + s_endpgm kernel emits SALU INST packet."""
    blob = _run_kernel([s_mov_b32(s[0], 42), s_endpgm()])
    packets = list(decode(blob))
    inst_pkts = [p for p in packets if isinstance(p, INST)]
    self.assertEqual(len(inst_pkts), 1)
    self.assertEqual(inst_pkts[0].op, InstOp.SALU)

  def test_valu_emits_valuinst(self):
    """Regular VALU ops emit VALUINST packets."""
    blob = _run_kernel([v_mov_b32_e32(v[0], 0), v_add_f32_e32(v[1], v[0], v[0]), s_endpgm()])
    packets = list(decode(blob))
    valu_pkts = [p for p in packets if isinstance(p, VALUINST)]
    self.assertEqual(len(valu_pkts), 2)
    # no INST packets for regular VALU
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 0)

  def test_waitcnt_emits_immediate(self):
    """s_waitcnt and s_nop emit IMMEDIATE packets."""
    blob = _run_kernel([s_nop(simm16=0), s_waitcnt(simm16=0), s_endpgm()])
    imm_pkts = [p for p in decode(blob) if isinstance(p, IMMEDIATE)]
    self.assertEqual(len(imm_pkts), 2)  # s_nop + s_waitcnt

  def test_endpgm_skipped(self):
    """s_endpgm does not emit any packet."""
    blob = _run_kernel([s_endpgm()])
    packets = list(decode(blob))
    self.assertEqual(len([p for p in packets if isinstance(p, INST)]), 0)
    self.assertEqual(len([p for p in packets if isinstance(p, IMMEDIATE)]), 0)

  def test_wave_lifecycle(self):
    """Every WAVESTART has a matching WAVEEND."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_endpgm()])
    packets = list(decode(blob))
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVESTART)), sum(1 for p in packets if isinstance(p, WAVEEND)))

  def test_layout_header(self):
    """First packet is LAYOUT_HEADER with layout=3."""
    blob = _run_kernel([s_endpgm()])
    packets = list(decode(blob))
    self.assertIsInstance(packets[0], LAYOUT_HEADER)
    self.assertEqual(packets[0].layout, 3)

  def test_blob_32byte_aligned(self):
    """SQTT blob is 32-byte aligned."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_mov_b32(s[1], 1), s_endpgm()])
    self.assertEqual(len(blob) % 32, 0)

  def test_multiple_waves(self):
    """Multiple wavefronts each get their own WAVESTART/WAVEEND."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_endpgm()], lx=64)  # 64 threads = 2 waves (WAVE_SIZE=32)
    packets = list(decode(blob))
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVESTART)), 2)
    self.assertEqual(sum(1 for p in packets if isinstance(p, WAVEEND)), 2)

  def test_branch_taken_and_not_taken(self):
    """A loop with s_cbranch_scc1 emits JUMP when taken, JUMP_NO on final iteration."""
    # s[0] = 2; loop: s[0] -= 1; cmp s[0] != 0 (SCC=1 if true); cbranch_scc1 loop; endpgm
    # iteration 1: s[0]=2→1, SCC=1 (1!=0), branch taken (JUMP)
    # iteration 2: s[0]=1→0, SCC=0 (0==0), branch not taken (JUMP_NO)
    blob = _run_kernel([s_mov_b32(s[0], 2), s_sub_u32(s[0], s[0], 1), s_cmp_lg_u32(s[0], 0), s_cbranch_scc1(simm16=-3), s_endpgm()])
    inst_pkts = [p for p in decode(blob) if isinstance(p, INST)]
    ops = [p.op for p in inst_pkts]
    self.assertIn(InstOp.JUMP, ops)
    self.assertIn(InstOp.JUMP_NO, ops)

  def test_timestamps_monotonic(self):
    """Timestamps are monotonically non-decreasing."""
    blob = _run_kernel([s_mov_b32(s[0], 0), s_mov_b32(s[1], 1), s_mov_b32(s[2], 2), s_endpgm()])
    times = [p._time for p in decode(blob)]
    self.assertEqual(times, sorted(times))

  def test_no_trace_without_profile(self):
    """No SQTT trace is emitted when PROFILE=0."""
    from test.mockgpu.amd.emu import run_asm, sqtt_traces
    code = s_endpgm().to_bytes()
    buf = (ctypes.c_char * len(code))(*code)
    sqtt_traces.clear()
    with Context(PROFILE=0):
      run_asm(ctypes.addressof(buf), len(code), 1, 1, 1, 1, 1, 1, 0)
    self.assertEqual(len(sqtt_traces), 0)

class TestSQTTEncoderTiming(unittest.TestCase):
  def test_explicit_ticks(self):
    emit, finish, finalize = make_encoder()
    emit(0, s_mov_b32(s[0], 1), None, 0)
    emit(0, v_mov_b32_e32(v[0], 2), None, 19)
    emit(0, s_nop(0), None, 19)
    finish(0, 200)
    packets = list(decode(finalize()))
    events = [p for p in packets if isinstance(p, (WAVESTART, INST, VALUINST, IMMEDIATE, WAVEEND))]
    self.assertEqual([p._time for p in events], [0, 0, 19, 19, 200])
    self.assertEqual(sum(isinstance(p, TS_DELTA_OR_MARK) for p in packets), 2)

  def test_delta_boundaries(self):
    for gap in (0, 1, 3, 4, 7, 8, 15, 16, 65535, 1 << 32, (1 << 36) - 1):
      with self.subTest(gap=gap):
        emit, finish, finalize = make_encoder()
        emit(0, s_mov_b32(s[0], 1), None, 0)
        emit(0, s_mov_b32(s[1], 2), None, gap)
        finish(0, gap)
        packets = list(decode(finalize()))
        self.assertEqual([p._time for p in packets if isinstance(p, INST)], [0, gap])
        self.assertEqual([p._time for p in packets if isinstance(p, WAVEEND)], [gap])

  def test_interleaved_waves_share_one_clock(self):
    emit, finish, finalize = make_encoder()
    for wave, tick in ((0, 4), (1, 4), (0, 10), (1, 12)):
      emit(wave, s_mov_b32(s[0], 1), None, tick)
    finish(1, 13)
    finish(0, 20)
    packets = list(decode(finalize()))
    self.assertEqual([(p.wave, p._time) for p in packets if isinstance(p, INST)], [(0, 4), (1, 4), (0, 10), (1, 12)])
    self.assertEqual([(p.wave, p._time) for p in packets if isinstance(p, WAVEEND)], [(1, 13), (0, 20)])

  def test_bad_timestamp_does_not_advance_clock(self):
    for bad in (-1, 0, 9, 1.5, True, "12", 10 + (1 << 36)):
      with self.subTest(timestamp=bad):
        emit, finish, finalize = make_encoder()
        emit(0, s_mov_b32(s[0], 1), None, 10)
        with self.assertRaises(ValueError): emit(0, s_mov_b32(s[1], 2), None, bad)
        emit(0, s_mov_b32(s[1], 2), None, 11)
        finish(0, 11)
        self.assertEqual([p._time for p in decode(finalize()) if isinstance(p, INST)], [10, 11])

  def test_untimed_packet_order_unchanged(self):
    emit, finish, finalize = make_encoder()
    emit(0, s_mov_b32(s[0], 1), None)
    emit(0, s_nop(0), None)
    finish(0)
    packets = [p for p in decode(finalize()) if isinstance(p, (WAVESTART, INST, IMMEDIATE, WAVEEND))]
    self.assertEqual([p._time for p in packets], [1, 2, 3, 4])

  def test_skipped_instruction_advances_explicit_time(self):
    for inst in (s_delay_alu(0), s_endpgm()):
      with self.subTest(instruction=inst):
        emit, finish, finalize = make_encoder()
        emit(0, s_mov_b32(s[0], 1), None, 0)
        emit(0, inst, None, 10)
        with self.assertRaises(ValueError): emit(0, s_mov_b32(s[1], 2), None, 5)
        emit(0, s_mov_b32(s[1], 2), None)
        finish(0, 12)
        packets = list(decode(finalize()))
        self.assertEqual([p._time for p in packets if isinstance(p, INST)], [0, 11])
        self.assertEqual([p._time for p in packets if isinstance(p, WAVEEND)], [12])

  def test_reused_wave_slot(self):
    emit, finish, finalize = make_encoder()
    for tick in (0, 10):
      emit(0, s_mov_b32(s[0], 1), None, tick)
      finish(0, tick+1)
    packets = list(decode(finalize()))
    self.assertEqual([p._time for p in packets if isinstance(p, WAVESTART)], [0, 10])
    self.assertEqual([p._time for p in packets if isinstance(p, WAVEEND)], [1, 11])

  def test_hardware_fixture_instruction_ticks_round_trip(self):
    from test.amd.test_sqtt_timing import upstream_traces
    from tinygrad.renderer.amd.sqtt import map_insts
    count = 0
    for name, trace, program in upstream_traces():
      with self.subTest(fixture=name, se=trace.se):
        packets = list(decode(trace.blob))
        simd = packets[0].simd
        original = [(p, i) for p, i in map_insts(trace.blob, program.lib, "gfx1100") if i is not None and
                    (not isinstance(p, WAVEEND) or p.simd == simd)]
        if not original: continue  # Some recorded SEs contain no instruction observations.
        emit, finish, finalize = make_encoder()
        for packet, info in original:
          taken = packet.op == InstOp.JUMP if isinstance(packet, INST) and packet.op in (InstOp.JUMP, InstOp.JUMP_NO) else None
          emit(info.wave, info.inst, taken, packet._time)
          if isinstance(packet, WAVEEND): finish(info.wave, packet._time)
        def identity(packet, info): return info.wave, info.pc, info.inst.to_bytes(), packet._time
        actual = [identity(p, i) for p, i in map_insts(finalize(), program.lib, "gfx1100") if i is not None]
        self.assertEqual(actual, [identity(p, i) for p, i in original])
        count += len(actual)
    self.assertGreater(count, 1000)

if __name__ == "__main__":
  unittest.main()
