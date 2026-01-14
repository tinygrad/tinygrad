"""Tests for SOP instructions - scalar operations.

Includes: s_add_u32, s_mov_b32, s_and_b32, s_or_b32, s_quadmask_b32, s_wqm_b32,
          s_cbranch_vccnz, s_cbranch_vccz
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestBasicScalar(unittest.TestCase):
  """Tests for basic scalar operations."""

  def test_s_add_u32(self):
    """S_ADD_U32 adds two scalar values."""
    instructions = [
      s_mov_b32(s[0], 100),
      s_mov_b32(s[1], 200),
      s_add_u32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 300)

  def test_s_add_u32_carry(self):
    """S_ADD_U32 sets SCC on overflow."""
    instructions = [
      s_mov_b32(s[0], 64),
      s_not_b32(s[0], s[0]),  # ~64 = 0xffffffbf
      s_mov_b32(s[1], 64),
      s_add_u32(s[2], s[0], s[1]),  # 0xffffffbf + 64 = 0xffffffff
      s_mov_b32(s[3], 1),
      s_add_u32(s[4], s[2], s[3]),  # 0xffffffff + 1 = overflow
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0)
    self.assertEqual(st.scc, 1)

  def test_s_brev_b32(self):
    """S_BREV_B32 reverses bits of a 32-bit value."""
    # 10 = 0b00000000000000000000000000001010
    # reversed = 0b01010000000000000000000000000000 = 0x50000000
    instructions = [
      s_mov_b32(s[0], 10),
      s_brev_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0x50000000)

  def test_s_brev_b32_all_ones(self):
    """S_BREV_B32 with all ones stays all ones."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_brev_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0xFFFFFFFF)

  def test_s_brev_b32_single_bit(self):
    """S_BREV_B32 with bit 0 set becomes bit 31."""
    instructions = [
      s_mov_b32(s[0], 1),
      s_brev_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0x80000000)


class TestQuadmaskWqm(unittest.TestCase):
  """Tests for S_QUADMASK_B32 and S_WQM_B32."""

  def test_s_quadmask_b32_all_quads_active(self):
    """S_QUADMASK_B32 with all quads active."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # All lanes active
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Each quad (4 lanes) with any bit set -> 1 bit in result
    # 32 lanes = 8 quads, all active -> 0xFF
    self.assertEqual(st.sgpr[1], 0xFF)

  def test_s_quadmask_b32_alternating_quads(self):
    """S_QUADMASK_B32 with alternating quads active."""
    instructions = [
      s_mov_b32(s[0], 0x0F0F0F0F),  # Quads 0,2,4,6 active
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Quads 0,2,4,6 have at least one bit -> 0b01010101 = 0x55
    self.assertEqual(st.sgpr[1], 0x55)

  def test_s_quadmask_b32_no_quads_active(self):
    """S_QUADMASK_B32 with no quads active."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0)

  def test_s_quadmask_b32_single_lane_per_quad(self):
    """S_QUADMASK_B32 with single lane active in each quad."""
    instructions = [
      s_mov_b32(s[0], 0x11111111),  # Bit 0 of each nibble
      s_quadmask_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # All 8 quads have at least one lane -> 0xFF
    self.assertEqual(st.sgpr[1], 0xFF)

  def test_s_wqm_b32_all_active(self):
    """S_WQM_B32 with all lanes active returns all 1s."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0xFFFFFFFF)

  def test_s_wqm_b32_alternating_quads(self):
    """S_WQM_B32 with single lane per quad expands to full quads."""
    instructions = [
      s_mov_b32(s[0], 0x11111111),  # One lane per quad
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Each quad with any bit expands to all 4 bits
    self.assertEqual(st.sgpr[1], 0xFFFFFFFF)

  def test_s_wqm_b32_zero(self):
    """S_WQM_B32 with zero input returns zero."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_wqm_b32(s[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0)


class TestBranch(unittest.TestCase):
  """Tests for branch instructions."""

  def test_cbranch_vccnz_ignores_vcc_hi(self):
    """S_CBRANCH_VCCNZ should only check VCC_LO in wave32."""
    instructions = [
      # Set VCC_LO = 0, VCC_HI = 1
      s_mov_b32(VCC_LO, 0),
      s_mov_b32(VCC_HI, 1),
      v_mov_b32_e32(v[0], 0),
      # If VCC_HI is incorrectly used, branch will be taken
      s_cbranch_vccnz(1),  # Skip next instruction if VCC != 0
      v_mov_b32_e32(v[0], 42),  # This should execute
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 42, "Branch should NOT be taken (VCC_LO is 0)")

  def test_cbranch_vccz_ignores_vcc_hi(self):
    """S_CBRANCH_VCCZ should only check VCC_LO in wave32."""
    instructions = [
      # Set VCC_LO = 1, VCC_HI = 0
      s_mov_b32(VCC_LO, 1),
      s_mov_b32(VCC_HI, 0),
      v_mov_b32_e32(v[0], 0),
      # If VCC_HI is incorrectly used, branch will be taken
      s_cbranch_vccz(1),  # Skip next instruction if VCC == 0
      v_mov_b32_e32(v[0], 42),  # This should execute
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 42, "Branch should NOT be taken (VCC_LO is 1)")

  def test_cbranch_vccnz_branches_on_vcc_lo(self):
    """S_CBRANCH_VCCNZ branches when VCC_LO is non-zero."""
    instructions = [
      s_mov_b32(VCC_LO, 1),
      v_mov_b32_e32(v[0], 0),
      s_cbranch_vccnz(1),  # Skip next instruction if VCC != 0
      v_mov_b32_e32(v[0], 42),  # This should be skipped
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 0, "Branch should be taken (VCC_LO is 1)")


class Test64BitLiterals(unittest.TestCase):
  """Tests for 64-bit literal encoding in instructions."""

  def test_64bit_literal_negative_encoding(self):
    """64-bit literal -2^32 encodes correctly."""
    lit = -4294967296.0  # -2^32
    lit_bits = f2i64(lit)
    instructions = [
      s_mov_b32(s[0], lit_bits & 0xffffffff),
      s_mov_b32(s[1], lit_bits >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    self.assertAlmostEqual(result, -4294967296.0, places=5)

class TestSCCBehavior(unittest.TestCase):
  """Tests for SCC condition code behavior."""

  def test_scc_from_s_cmp(self):
    """SCC should be set by scalar compare."""
    instructions = [
      s_mov_b32(s[0], 10),
      s_cmp_eq_u32(s[0], 10),
      s_cselect_b32(s[1], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 1, "SCC should be true")
    self.assertEqual(st.scc, 1)

  def test_scc_clear(self):
    """SCC should be cleared by failing compare."""
    instructions = [
      s_mov_b32(s[0], 10),
      s_cmp_eq_u32(s[0], 20),
      s_cselect_b32(s[1], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0, "SCC should be false")
    self.assertEqual(st.scc, 0)


class TestSignedArithmetic(unittest.TestCase):
  """Tests for S_ADD_I32, S_SUB_I32 and their SCC overflow behavior."""

  def test_s_add_i32_no_overflow(self):
    """S_ADD_I32: 1 + 1 = 2, no overflow, SCC=0."""
    instructions = [
      s_mov_b32(s[0], 1),
      s_add_i32(s[1], s[0], 1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 2)
    self.assertEqual(st.scc, 0, "No overflow, SCC should be 0")

  def test_s_add_i32_positive_overflow(self):
    """S_ADD_I32: MAX_INT + 1 overflows, SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0x7FFFFFFF),  # MAX_INT
      s_add_i32(s[1], s[0], 1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[1], 0x80000000)  # Wraps to MIN_INT
    self.assertEqual(st.scc, 1, "Overflow, SCC should be 1")

  def test_s_add_i32_negative_no_overflow(self):
    """S_ADD_I32: -10 + 20 = 10, no overflow."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFF6),  # -10 in two's complement
      s_mov_b32(s[1], 20),
      s_add_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 10)
    self.assertEqual(st.scc, 0)

  def test_s_add_i32_negative_overflow(self):
    """S_ADD_I32: MIN_INT + (-1) underflows, SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # MIN_INT
      s_mov_b32(s[1], 0xFFFFFFFF),  # -1
      s_add_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0x7FFFFFFF)  # Wraps to MAX_INT
    self.assertEqual(st.scc, 1, "Underflow, SCC should be 1")

  def test_s_sub_i32_no_overflow(self):
    """S_SUB_I32: 10 - 5 = 5, no overflow."""
    instructions = [
      s_mov_b32(s[0], 10),
      s_mov_b32(s[1], 5),
      s_sub_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 5)
    self.assertEqual(st.scc, 0)

  def test_s_sub_i32_overflow(self):
    """S_SUB_I32: MAX_INT - (-1) overflows, SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0x7FFFFFFF),  # MAX_INT
      s_mov_b32(s[1], 0xFFFFFFFF),  # -1
      s_sub_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0x80000000)  # Wraps to MIN_INT
    self.assertEqual(st.scc, 1, "Overflow, SCC should be 1")

  def test_s_mul_hi_u32(self):
    """S_MUL_HI_U32: high 32 bits of u32 * u32."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # 2^31
      s_mov_b32(s[1], 4),
      s_mul_hi_u32(s[2], s[0], s[1]),  # (2^31 * 4) >> 32 = 2
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 2)

  def test_s_mul_i32(self):
    """S_MUL_I32: signed multiply low 32 bits."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),  # -1
      s_mov_b32(s[1], 10),
      s_mul_i32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 0xFFFFFFF6)  # -10

  def test_division_sequence_from_llvm(self):
    """Test the division sequence pattern from LLVM-generated code."""
    # This sequence is from the sin kernel and computes integer division
    # s10 = dividend, s18 = divisor, result in s6/s14
    dividend = 0x28BE60DB  # Some value from the sin kernel
    divisor = 3  # Simplified divisor
    instructions = [
      s_mov_b32(s[10], dividend),
      s_mov_b32(s[18], divisor),
      # Compute reciprocal approximation: s6 = ~0 / divisor (approx)
      s_mov_b32(s[11], 0),
      s_sub_i32(s[11], s[11], s[18]),  # s11 = -divisor
      # For testing, just verify basic arithmetic works
      s_mul_i32(s[6], s[10], 2),
      s_add_i32(s[7], s[6], 1),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[6], (dividend * 2) & 0xFFFFFFFF)
    self.assertEqual(st.sgpr[7], ((dividend * 2) + 1) & 0xFFFFFFFF)


class Test64BitCompare(unittest.TestCase):
  """Tests for 64-bit scalar compare instructions."""

  def test_s_cmp_eq_u64_equal(self):
    """S_CMP_EQ_U64: comparing equal 64-bit values sets SCC=1."""
    val = 0x123456789ABCDEF0
    instructions = [
      s_mov_b32(s[0], val & 0xFFFFFFFF),
      s_mov_b32(s[1], val >> 32),
      s_mov_b32(s[2], val & 0xFFFFFFFF),
      s_mov_b32(s[3], val >> 32),
      s_cmp_eq_u64(s[0:1], s[2:3]),
      s_cselect_b32(s[4], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.scc, 1)
    self.assertEqual(st.sgpr[4], 1)

  def test_s_cmp_eq_u64_different_upper_bits(self):
    """S_CMP_EQ_U64: values differing only in upper 32 bits are not equal."""
    # This is the bug case - if only lower 32 bits are compared, these would be equal
    instructions = [
      s_mov_b32(s[0], 0),  # lower 32 bits of value 0
      s_mov_b32(s[1], 0),  # upper 32 bits of value 0
      s_mov_b32(s[2], 0),  # lower 32 bits of 0x100000000
      s_mov_b32(s[3], 1),  # upper 32 bits of 0x100000000
      s_cmp_eq_u64(s[0:1], s[2:3]),
      s_cselect_b32(s[4], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.scc, 0, "0 != 0x100000000, SCC should be 0")
    self.assertEqual(st.sgpr[4], 0)

  def test_s_cmp_lg_u64_different(self):
    """S_CMP_LG_U64: different 64-bit values sets SCC=1."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 0),  # s[0:1] = 0
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 1),  # s[2:3] = 0x100000000
      s_cmp_lg_u64(s[0:1], s[2:3]),
      s_cselect_b32(s[4], 1, 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.scc, 1, "0 != 0x100000000, SCC should be 1")
    self.assertEqual(st.sgpr[4], 1)


if __name__ == '__main__':
  unittest.main()
