#!/usr/bin/env python3
import math, unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f


class TestSpecialValues(unittest.TestCase):
  """Tests for special float values - inf, nan, zero handling."""

  def test_v_mul_f32_zero_times_inf(self):
    """V_MUL_F32: 0 * inf = NaN."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[1], s[0]),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_add_f32_inf_minus_inf(self):
    """V_ADD_F32: inf + (-inf) = NaN."""
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      s_mov_b32(s[1], 0xff800000),  # -inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_fma_f32_with_inf(self):
    """V_FMA_F32: 1.0 * inf + 0 = inf."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertTrue(math.isinf(result) and result > 0)

  def test_v_exp_f32_large_negative(self):
    """V_EXP_F32 of large negative value (2^-100) returns very small number."""
    instructions = [
      s_mov_b32(s[0], f2i(-100.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # V_EXP_F32 computes 2^x, so 2^-100 is ~7.9e-31 (very small but not 0)
    result = i2f(st.vgpr[0][1])
    self.assertLess(result, 1e-20)  # Just verify it's very small

  def test_v_exp_f32_large_positive(self):
    """V_EXP_F32 of large positive value (2^100) returns very large number."""
    instructions = [
      s_mov_b32(s[0], f2i(100.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # V_EXP_F32 computes 2^x, so 2^100 is ~1.27e30 (very large)
    result = i2f(st.vgpr[0][1])
    self.assertGreater(result, 1e20)  # Just verify it's very large


class TestNewPcodeHelpers(unittest.TestCase):
  """Tests for newly added pcode helper functions (SAD, BYTE_PERMUTE, BF16)."""

  def test_v_sad_u8_basic(self):
    """V_SAD_U8: Sum of absolute differences of 4 bytes."""
    # s0 = 0x05040302, s1 = 0x04030201, s2 = 10 -> diff = 1+1+1+1 = 4, result = 14
    instructions = [
      s_mov_b32(s[0], 0x05040302),
      s_mov_b32(s[1], 0x04030201),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 10),
      v_sad_u8(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 14, f"Expected 14, got {result}")

  def test_v_sad_u8_identical_bytes(self):
    """V_SAD_U8: When both operands are identical, SAD = 0 + accumulator."""
    instructions = [
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),  # Same as v0
      v_mov_b32_e32(v[2], 42),    # Accumulator
      v_sad_u8(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 42, f"Expected 42, got {result}")

  def test_v_sad_u16_basic(self):
    """V_SAD_U16: Sum of absolute differences of 2 half-words."""
    # s0 = 0x00020003, s1 = 0x00010001 -> diff = |2-1| + |3-1| = 1 + 2 = 3
    instructions = [
      s_mov_b32(s[0], 0x00020003),
      s_mov_b32(s[1], 0x00010001),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_sad_u16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 3, f"Expected 3, got {result}")

  def test_v_sad_u32_basic(self):
    """V_SAD_U32: Absolute difference of 32-bit values."""
    # s0 = 100, s1 = 30 -> diff = 70, s2 = 5 -> result = 75
    instructions = [
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 30),
      v_mov_b32_e32(v[2], 5),
      v_sad_u32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 75, f"Expected 75, got {result}")

  def test_v_msad_u8_masked(self):
    """V_MSAD_U8: Skip bytes where reference (s1) is 0."""
    # s0 = 0x10101010, s1 = 0x00010001, s2 = 0
    # Only bytes 0 and 2 of s1 are non-zero, so only those contribute
    # diff = |0x10-0x01| + |0x10-0x01| = 15 + 15 = 30
    instructions = [
      s_mov_b32(s[0], 0x10101010),
      s_mov_b32(s[1], 0x00010001),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),
      v_msad_u8(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 30, f"Expected 30, got {result}")

  def test_v_perm_b32_select_bytes(self):
    """V_PERM_B32: Select bytes from combined {s0, s1}."""
    # Combined = {S0, S1} where S1 is bytes 0-3, S0 is bytes 4-7
    # s0 = 0x03020100 -> bytes 4-7 of combined
    # s1 = 0x07060504 -> bytes 0-3 of combined
    # Combined = 0x03020100_07060504
    # selector = 0x00010203 -> select bytes 3,2,1,0 from combined = 0x04,0x05,0x06,0x07
    instructions = [
      s_mov_b32(s[0], 0x03020100),
      s_mov_b32(s[1], 0x07060504),
      s_mov_b32(s[2], 0x00010203),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_perm_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 0x04050607, f"Expected 0x04050607, got 0x{result:08x}")

  def test_v_perm_b32_select_high_bytes(self):
    """V_PERM_B32: Select bytes from high word (s0)."""
    # Combined = {S0, S1} where S1 is bytes 0-3, S0 is bytes 4-7
    # s0 = 0x03020100 -> bytes 4-7 of combined
    # s1 = 0x07060504 -> bytes 0-3 of combined
    # selector = 0x04050607 -> select bytes 7,6,5,4 from combined = 0x00,0x01,0x02,0x03
    instructions = [
      s_mov_b32(s[0], 0x03020100),
      s_mov_b32(s[1], 0x07060504),
      s_mov_b32(s[2], 0x04050607),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_perm_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 0x00010203, f"Expected 0x00010203, got 0x{result:08x}")

  def test_v_perm_b32_constant_values(self):
    """V_PERM_B32: Test constant 0x00 (sel=12) and 0xFF (sel>=13)."""
    # selector = 0x0C0D0E0F -> bytes: 12=0x00, 13=0xFF, 14=0xFF, 15=0xFF
    instructions = [
      s_mov_b32(s[0], 0x12345678),
      s_mov_b32(s[1], 0xABCDEF01),
      s_mov_b32(s[2], 0x0C0D0E0F),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_perm_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    # byte 0: sel=0x0F >= 13 -> 0xFF
    # byte 1: sel=0x0E >= 13 -> 0xFF
    # byte 2: sel=0x0D >= 13 -> 0xFF
    # byte 3: sel=0x0C = 12 -> 0x00
    self.assertEqual(result, 0x00FFFFFF, f"Expected 0x00FFFFFF, got 0x{result:08x}")

  def test_v_perm_b32_sign_extend(self):
    """V_PERM_B32: Test sign extension selectors 8-11."""
    # Combined = {S0, S1} where S1 is bytes 0-3, S0 is bytes 4-7
    # s0 = 0x00008000 -> byte 5 (0x80) has sign bit set
    # s1 = 0x80000080 -> bytes 1 (0x00) and 3 (0x80) have sign bits, byte 0 (0x80) has sign bit
    # Combined = 0x00008000_80000080
    # selector = 0x08090A0B -> sign of bytes 1,3,5,7
    # byte 0: sel=0x0B -> sign of byte 7 (0x00) -> 0x00
    # byte 1: sel=0x0A -> sign of byte 5 (0x80) -> 0xFF
    # byte 2: sel=0x09 -> sign of byte 3 (0x80) -> 0xFF
    # byte 3: sel=0x08 -> sign of byte 1 (0x00) -> 0x00
    instructions = [
      s_mov_b32(s[0], 0x00008000),
      s_mov_b32(s[1], 0x80000080),
      s_mov_b32(s[2], 0x08090A0B),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_perm_b32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    self.assertEqual(result, 0x00FFFF00, f"Expected 0x00FFFF00, got 0x{result:08x}")

  def test_v_dot2_f32_bf16_basic(self):
    """V_DOT2_F32_BF16: Dot product of two bf16 pairs accumulated into f32."""
    from extra.assembly.amd.pcode import _ibf16
    # A = packed (2.0, 3.0) as bf16, B = packed (4.0, 5.0) as bf16
    # Result = 2*4 + 3*5 + acc = 8 + 15 + 0 = 23.0
    a_lo, a_hi = _ibf16(2.0), _ibf16(3.0)
    b_lo, b_hi = _ibf16(4.0), _ibf16(5.0)
    a_packed = (a_hi << 16) | a_lo
    b_packed = (b_hi << 16) | b_lo
    instructions = [
      s_mov_b32(s[0], a_packed),
      s_mov_b32(s[1], b_packed),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], 0),  # accumulator = 0
      v_dot2_f32_bf16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertAlmostEqual(result, 23.0, places=1, msg=f"Expected 23.0, got {result}")


class TestQuadmaskWqm(unittest.TestCase):
  """Tests for S_QUADMASK and S_WQM instructions."""

  def test_s_quadmask_b32_all_quads_active(self):
    """S_QUADMASK_B32: All quads have at least one active lane."""
    # Input: 0xFFFFFFFF (all bits set) -> all 8 quads active -> result = 0xFF
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.sgpr[1]
    self.assertEqual(result, 0xFF, f"Expected 0xFF, got 0x{result:x}")
    self.assertEqual(st.scc, 1, "SCC should be 1 (result != 0)")

  def test_s_quadmask_b32_alternating_quads(self):
    """S_QUADMASK_B32: Every other quad has lanes active."""
    # Input: 0x0F0F0F0F -> quads 0,2,4,6 active (bits 0-3, 8-11, 16-19, 24-27)
    # Result: bits 0,2,4,6 set = 0x55
    instructions = [
      s_mov_b32(s[0], 0x0F0F0F0F),
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.sgpr[1]
    self.assertEqual(result, 0x55, f"Expected 0x55, got 0x{result:x}")

  def test_s_quadmask_b32_no_quads_active(self):
    """S_QUADMASK_B32: No quads have active lanes."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.sgpr[1]
    self.assertEqual(result, 0, f"Expected 0, got 0x{result:x}")
    self.assertEqual(st.scc, 0, "SCC should be 0 (result == 0)")

  def test_s_quadmask_b32_single_lane_per_quad(self):
    """S_QUADMASK_B32: Single lane active in each quad."""
    # Input: 0x11111111 -> bit 0 of each nibble set -> all 8 quads active
    instructions = [
      s_mov_b32(s[0], 0x11111111),
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.sgpr[1]
    self.assertEqual(result, 0xFF, f"Expected 0xFF, got 0x{result:x}")

  def test_s_wqm_b32_all_active(self):
    """S_WQM_B32: Whole quad mode - if any lane in quad is active, activate all."""
    # Input: 0x11111111 -> one lane per quad -> output all quads fully active = 0xFFFFFFFF
    instructions = [
      s_mov_b32(s[0], 0x11111111),
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.sgpr[1]
    self.assertEqual(result, 0xFFFFFFFF, f"Expected 0xFFFFFFFF, got 0x{result:x}")
    self.assertEqual(st.scc, 1, "SCC should be 1 (result != 0)")

  def test_s_wqm_b32_alternating_quads(self):
    """S_WQM_B32: Only some quads have active lanes."""
    # Input: 0x0000000F -> only quad 0 has lanes -> output = 0x0000000F (quad 0 all active)
    instructions = [
      s_mov_b32(s[0], 0x00000001),  # single lane in quad 0
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.sgpr[1]
    self.assertEqual(result, 0x0000000F, f"Expected 0x0000000F, got 0x{result:x}")

  def test_s_wqm_b32_zero(self):
    """S_WQM_B32: No lanes active."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.sgpr[1]
    self.assertEqual(result, 0, f"Expected 0, got 0x{result:x}")
    self.assertEqual(st.scc, 0, "SCC should be 0 (result == 0)")


if __name__ == "__main__":
  unittest.main()
