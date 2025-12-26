#!/usr/bin/env python3
"""Test emu2.py (pseudocode-based emulator) against emu.py (hand-coded emulator)."""
import unittest
import ctypes
from extra.assembly.rdna3.emu import WaveState as WaveState1, decode_program as decode1, step_wave as step1
from extra.assembly.rdna3.emu2 import WaveState as WaveState2, decode_program as decode2, step_wave as step2
from extra.assembly.rdna3.autogen import (
  SOP2, SOP1, VOP2, VOP1,
  SOP2Op, SOP1Op, VOP2Op, VOP1Op
)

def make_sop2(op: int, sdst: int, ssrc0: int, ssrc1: int) -> bytes:
  """Create a SOP2 instruction."""
  inst = SOP2(op=op, sdst=sdst, ssrc0=ssrc0, ssrc1=ssrc1)
  return inst.to_bytes()

def make_sop1(op: int, sdst: int, ssrc0: int) -> bytes:
  """Create a SOP1 instruction."""
  inst = SOP1(op=op, sdst=sdst, ssrc0=ssrc0)
  return inst.to_bytes()

def make_vop2(op: int, vdst: int, src0: int, vsrc1: int) -> bytes:
  """Create a VOP2 instruction."""
  from extra.assembly.rdna3.lib import v
  inst = VOP2(op=op, vdst=v[vdst], src0=src0, vsrc1=v[vsrc1])
  return inst.to_bytes()

def make_vop1(op: int, vdst: int, src0: int) -> bytes:
  """Create a VOP1 instruction."""
  from extra.assembly.rdna3.lib import v
  inst = VOP1(op=op, vdst=v[vdst], src0=src0)
  return inst.to_bytes()

class TestEmu2VsEmu(unittest.TestCase):
  """Compare emu2 (pseudocode) against emu (hand-coded) for various instructions."""

  def _compare_scalar_op(self, code: bytes, s0_val: int, s1_val: int = 0, scc_in: int = 0):
    """Run a scalar operation through both emulators and compare results."""
    prog1, prog2 = decode1(code), decode2(code)
    lds = bytearray(65536)

    # Set up state for emu1
    st1 = WaveState1()
    st1.sgpr[2] = s0_val & 0xffffffff  # ssrc0
    st1.sgpr[3] = s1_val & 0xffffffff  # ssrc1
    st1.scc = scc_in

    # Set up state for emu2
    st2 = WaveState2()
    st2.sgpr[2] = s0_val & 0xffffffff  # ssrc0
    st2.sgpr[3] = s1_val & 0xffffffff  # ssrc1
    st2.scc = scc_in

    # Run both emulators
    r1 = step1(prog1, st1, lds, 1)
    r2 = step2(prog2, st2, lds, 1)

    # Compare results
    self.assertEqual(r1, r2, f"Return codes differ: {r1} vs {r2}")
    self.assertEqual(st1.sgpr[4], st2.sgpr[4], f"Result register differs: {st1.sgpr[4]:#x} vs {st2.sgpr[4]:#x}")
    self.assertEqual(st1.scc, st2.scc, f"SCC differs: {st1.scc} vs {st2.scc}")

  def _compare_vector_op(self, code: bytes, lane_vals: list[tuple[int, int]], d_init: int = 0):
    """Run a vector operation through both emulators and compare results."""
    prog1, prog2 = decode1(code), decode2(code)
    lds = bytearray(65536)

    n_lanes = len(lane_vals)

    # Set up state for emu1
    st1 = WaveState1()
    st1.exec_mask = (1 << n_lanes) - 1
    for i, (s0, s1) in enumerate(lane_vals):
      st1.vgpr[i][0] = s0 & 0xffffffff  # src0 as VGPR0
      st1.vgpr[i][1] = s1 & 0xffffffff  # vsrc1 as VGPR1
      st1.vgpr[i][2] = d_init & 0xffffffff  # vdst as VGPR2

    # Set up state for emu2
    st2 = WaveState2()
    st2.exec_mask = (1 << n_lanes) - 1
    for i, (s0, s1) in enumerate(lane_vals):
      st2.vgpr[i][0] = s0 & 0xffffffff
      st2.vgpr[i][1] = s1 & 0xffffffff
      st2.vgpr[i][2] = d_init & 0xffffffff

    # Run both emulators
    r1 = step1(prog1, st1, lds, n_lanes)
    r2 = step2(prog2, st2, lds, n_lanes)

    # Compare results
    self.assertEqual(r1, r2, f"Return codes differ: {r1} vs {r2}")
    for i in range(n_lanes):
      self.assertEqual(st1.vgpr[i][2], st2.vgpr[i][2],
                      f"Lane {i} vdst differs: {st1.vgpr[i][2]:#x} vs {st2.vgpr[i][2]:#x}")

  # ═══════════════════════════════════════════════════════════════════════════════
  # SOP2 Tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_s_add_u32(self):
    code = make_sop2(SOP2Op.S_ADD_U32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 100, 50)
    self._compare_scalar_op(code, 0xffffffff, 2)  # overflow

  def test_s_sub_u32(self):
    code = make_sop2(SOP2Op.S_SUB_U32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 100, 50)
    self._compare_scalar_op(code, 50, 100)  # borrow

  def test_s_and_b32(self):
    code = make_sop2(SOP2Op.S_AND_B32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 0xff00ff00, 0x00ff00ff)
    self._compare_scalar_op(code, 0xffffffff, 0xffffffff)

  def test_s_or_b32(self):
    code = make_sop2(SOP2Op.S_OR_B32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 0xff00ff00, 0x00ff00ff)

  def test_s_xor_b32(self):
    code = make_sop2(SOP2Op.S_XOR_B32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 0xff00ff00, 0x00ff00ff)

  def test_s_lshl_b32(self):
    code = make_sop2(SOP2Op.S_LSHL_B32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 1, 4)
    self._compare_scalar_op(code, 0xffffffff, 16)

  def test_s_lshr_b32(self):
    code = make_sop2(SOP2Op.S_LSHR_B32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 0x80000000, 4)

  def test_s_ashr_i32(self):
    code = make_sop2(SOP2Op.S_ASHR_I32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 0x80000000, 4)  # negative
    self._compare_scalar_op(code, 0x7fffffff, 4)  # positive

  def test_s_mul_i32(self):
    code = make_sop2(SOP2Op.S_MUL_I32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 100, 50)
    self._compare_scalar_op(code, 0xffffffff, 2)  # -1 * 2

  def test_s_min_u32(self):
    code = make_sop2(SOP2Op.S_MIN_U32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 100, 50)
    self._compare_scalar_op(code, 50, 100)

  def test_s_max_u32(self):
    code = make_sop2(SOP2Op.S_MAX_U32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 100, 50)
    self._compare_scalar_op(code, 50, 100)

  def test_s_cselect_b32(self):
    code = make_sop2(SOP2Op.S_CSELECT_B32, sdst=4, ssrc0=2, ssrc1=3)
    self._compare_scalar_op(code, 100, 50, scc_in=1)
    self._compare_scalar_op(code, 100, 50, scc_in=0)

  # ═══════════════════════════════════════════════════════════════════════════════
  # SOP1 Tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_s_mov_b32(self):
    code = make_sop1(SOP1Op.S_MOV_B32, sdst=4, ssrc0=2)
    self._compare_scalar_op(code, 42)
    self._compare_scalar_op(code, 0xdeadbeef)

  def test_s_not_b32(self):
    code = make_sop1(SOP1Op.S_NOT_B32, sdst=4, ssrc0=2)
    self._compare_scalar_op(code, 0)
    self._compare_scalar_op(code, 0xffffffff)

  def test_s_abs_i32(self):
    code = make_sop1(SOP1Op.S_ABS_I32, sdst=4, ssrc0=2)
    self._compare_scalar_op(code, 0xffffffff)  # -1
    self._compare_scalar_op(code, 100)

  # ═══════════════════════════════════════════════════════════════════════════════
  # VOP2 Tests (using VGPR sources: src0=VGPR0 (256), vsrc1=VGPR1 (1))
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_v_add_f32(self):
    import struct
    def f2i(f): return struct.unpack('<I', struct.pack('<f', f))[0]
    # VOP2: src0 is encoded, vsrc1 is VGPR index
    code = make_vop2(VOP2Op.V_ADD_F32, vdst=2, src0=256, vsrc1=1)  # src0=v0, vsrc1=v1
    self._compare_vector_op(code, [(f2i(1.0), f2i(2.0)), (f2i(-1.0), f2i(3.0))])

  def test_v_sub_f32(self):
    import struct
    def f2i(f): return struct.unpack('<I', struct.pack('<f', f))[0]
    code = make_vop2(VOP2Op.V_SUB_F32, vdst=2, src0=256, vsrc1=1)
    self._compare_vector_op(code, [(f2i(5.0), f2i(2.0)), (f2i(1.0), f2i(3.0))])

  def test_v_mul_f32(self):
    import struct
    def f2i(f): return struct.unpack('<I', struct.pack('<f', f))[0]
    code = make_vop2(VOP2Op.V_MUL_F32, vdst=2, src0=256, vsrc1=1)
    self._compare_vector_op(code, [(f2i(2.0), f2i(3.0)), (f2i(-2.0), f2i(4.0))])

  def test_v_and_b32(self):
    code = make_vop2(VOP2Op.V_AND_B32, vdst=2, src0=256, vsrc1=1)
    self._compare_vector_op(code, [(0xff00ff00, 0x00ff00ff)])

  def test_v_or_b32(self):
    code = make_vop2(VOP2Op.V_OR_B32, vdst=2, src0=256, vsrc1=1)
    self._compare_vector_op(code, [(0xff00ff00, 0x00ff00ff)])

  def test_v_xor_b32(self):
    code = make_vop2(VOP2Op.V_XOR_B32, vdst=2, src0=256, vsrc1=1)
    self._compare_vector_op(code, [(0xff00ff00, 0x00ff00ff)])

  def test_v_lshlrev_b32(self):
    code = make_vop2(VOP2Op.V_LSHLREV_B32, vdst=2, src0=256, vsrc1=1)
    self._compare_vector_op(code, [(4, 1), (16, 0xffffffff)])

  def test_v_lshrrev_b32(self):
    code = make_vop2(VOP2Op.V_LSHRREV_B32, vdst=2, src0=256, vsrc1=1)
    self._compare_vector_op(code, [(4, 0x80000000), (16, 0xffffffff)])

  # ═══════════════════════════════════════════════════════════════════════════════
  # VOP1 Tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_v_mov_b32(self):
    code = make_vop1(VOP1Op.V_MOV_B32, vdst=2, src0=256)  # v2 = v0
    self._compare_vector_op(code, [(0xdeadbeef, 0)])

  def test_v_not_b32(self):
    code = make_vop1(VOP1Op.V_NOT_B32, vdst=2, src0=256)
    self._compare_vector_op(code, [(0, 0), (0xffffffff, 0)])

if __name__ == "__main__":
  unittest.main()
