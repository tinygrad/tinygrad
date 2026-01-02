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
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 0),
      s_mov_b32(s[SrcEnum.VCC_HI - 128], 1),
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
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 1),
      s_mov_b32(s[SrcEnum.VCC_HI - 128], 0),
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
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 1),
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

  def test_64bit_literal_positive_encoding(self):
    """64-bit instruction encodes large positive literals correctly."""
    large_val = 0x12345678
    inst = v_add_f64(v[2], v[0], large_val)
    self.assertIsNotNone(inst._literal, "Literal should be set")
    actual_lit = (inst._literal >> 32) & 0xffffffff
    self.assertEqual(actual_lit, large_val, f"Literal should be {large_val:#x}, got {actual_lit:#x}")


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


if __name__ == '__main__':
  unittest.main()
