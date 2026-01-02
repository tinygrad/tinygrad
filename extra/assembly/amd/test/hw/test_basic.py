#!/usr/bin/env python3
"""Tests for basic ALU instructions: add, mul, mov, shifts, and/or/xor."""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f

class TestBasicArithmetic(unittest.TestCase):
  """Basic arithmetic instruction tests."""

  def test_v_add_f32(self):
    """V_ADD_F32 adds two floats."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_add_f32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, places=5)

  def test_v_mul_f32(self):
    """V_MUL_F32 multiplies two floats."""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 8.0, places=5)

  def test_v_mov_b32(self):
    """V_MOV_B32 moves a value."""
    st = run_program([
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 42)

  def test_s_add_u32(self):
    """S_ADD_U32 adds two scalar values."""
    st = run_program([
      s_mov_b32(s[0], 100),
      s_mov_b32(s[1], 200),
      s_add_u32(s[2], s[0], s[1]),
    ], n_lanes=1)
    self.assertEqual(st.sgpr[2], 300)

  def test_s_add_u32_carry(self):
    """S_ADD_U32 sets SCC on overflow."""
    st = run_program([
      s_mov_b32(s[0], 64),
      s_not_b32(s[0], s[0]),
      s_mov_b32(s[1], 64),
      s_add_u32(s[2], s[0], s[1]),
      s_mov_b32(s[3], 1),
      s_add_u32(s[4], s[2], s[3]),
    ], n_lanes=1)
    self.assertEqual(st.sgpr[4], 0)
    self.assertEqual(st.scc, 1)

  def test_s_sub_u32(self):
    """S_SUB_U32 subtracts two scalar values."""
    st = run_program([
      s_mov_b32(s[0], 100),
      s_mov_b32(s[1], 30),
      s_sub_u32(s[2], s[0], s[1]),
    ], n_lanes=1)
    self.assertEqual(st.sgpr[2], 70)

  def test_v_add_nc_u32(self):
    """V_ADD_NC_U32 adds without carry."""
    st = run_program([
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 200),
      v_add_nc_u32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 300)


class TestBitManipulation(unittest.TestCase):
  """Tests for bit manipulation instructions."""

  def test_v_and_b32(self):
    """V_AND_B32 bitwise and."""
    st = run_program([
      s_mov_b32(s[0], 0xff),
      s_mov_b32(s[1], 0x0f),
      v_mov_b32_e32(v[0], s[0]),
      v_and_b32_e32(v[1], s[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x0f)

  def test_v_or_b32(self):
    """V_OR_B32 bitwise or."""
    st = run_program([
      s_mov_b32(s[0], 0xf0),
      s_mov_b32(s[1], 0x0f),
      v_mov_b32_e32(v[0], s[0]),
      v_or_b32_e32(v[1], s[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff)

  def test_v_xor_b32(self):
    """V_XOR_B32 bitwise xor."""
    st = run_program([
      s_mov_b32(s[0], 0xff),
      s_mov_b32(s[1], 0xf0),
      v_mov_b32_e32(v[0], s[0]),
      v_xor_b32_e32(v[1], s[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x0f)

  def test_v_lshrrev_b32(self):
    """V_LSHRREV_B32 logical shift right."""
    st = run_program([
      s_mov_b32(s[0], 0xff00),
      v_mov_b32_e32(v[0], s[0]),
      v_lshrrev_b32_e32(v[1], 8, v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff)

  def test_v_lshlrev_b32(self):
    """V_LSHLREV_B32 logical shift left."""
    st = run_program([
      s_mov_b32(s[0], 0xff),
      v_mov_b32_e32(v[0], s[0]),
      v_lshlrev_b32_e32(v[1], 8, v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff00)

  def test_v_ashrrev_i32(self):
    """V_ASHRREV_I32 arithmetic shift right preserves sign."""
    st = run_program([
      s_mov_b32(s[0], 0x80000000),  # -2147483648
      v_mov_b32_e32(v[0], s[0]),
      v_ashrrev_i32_e32(v[1], 4, v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xf8000000)

  def test_s_not_b32(self):
    """S_NOT_B32 bitwise not."""
    st = run_program([
      s_mov_b32(s[0], 0x0f0f0f0f),
      s_not_b32(s[1], s[0]),
    ], n_lanes=1)
    self.assertEqual(st.sgpr[1], 0xf0f0f0f0)

  def test_v_alignbit_b32(self):
    """V_ALIGNBIT_B32 extracts bits from concatenated sources."""
    st = run_program([
      s_mov_b32(s[0], 0x12),
      s_mov_b32(s[1], 0x34),
      s_mov_b32(s[2], 4),
      v_mov_b32_e32(v[0], s[2]),
      v_alignbit_b32(v[1], s[0], s[1], v[0]),
    ], n_lanes=1)
    expected = ((0x12 << 32) | 0x34) >> 4
    self.assertEqual(st.vgpr[0][1], expected & 0xffffffff)


class TestMultiLane(unittest.TestCase):
  """Tests for multi-lane execution."""

  def test_v_mov_all_lanes(self):
    """V_MOV_B32 sets all lanes to the same value."""
    st = run_program([
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 42)

  def test_v_cmp_sets_vcc_bits(self):
    """V_CMP_EQ sets VCC bits based on per-lane comparison."""
    st = run_program([
      s_mov_b32(s[0], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_eq_u32_e32(v[0], v[1]),
    ], n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should match")

  def test_v_cmp_ne(self):
    """V_CMP_NE tests for inequality."""
    st = run_program([
      s_mov_b32(s[0], 5),
      s_mov_b32(s[1], 10),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_ne_u32_e32(v[0], v[1]),
    ], n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should differ")


class TestComparison(unittest.TestCase):
  """Tests for comparison instructions."""

  def test_v_cmp_gt_u32(self):
    """V_CMP_GT_U32 tests for greater than (unsigned)."""
    st = run_program([
      v_mov_b32_e32(v[0], 10),
      v_mov_b32_e32(v[1], 5),
      v_cmp_gt_u32_e32(v[0], v[1]),
    ], n_lanes=1)
    self.assertEqual(st.vcc & 1, 1)

  def test_v_cmp_lt_u32(self):
    """V_CMP_LT_U32 tests for less than (unsigned)."""
    st = run_program([
      v_mov_b32_e32(v[0], 5),
      v_mov_b32_e32(v[1], 10),
      v_cmp_lt_u32_e32(v[0], v[1]),
    ], n_lanes=1)
    self.assertEqual(st.vcc & 1, 1)

  def test_v_cmp_ge_i32(self):
    """V_CMP_GE_I32 tests for greater or equal (signed)."""
    st = run_program([
      v_mov_b32_e32(v[0], 5),
      v_mov_b32_e32(v[1], 5),
      v_cmp_ge_i32_e32(v[0], v[1]),
    ], n_lanes=1)
    self.assertEqual(st.vcc & 1, 1)


class TestMin(unittest.TestCase):
  """Tests for min/max instructions."""

  def test_v_min_f32(self):
    """V_MIN_F32 returns minimum of two floats."""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_min_f32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 2.0, places=5)

  def test_v_max_f32(self):
    """V_MAX_F32 returns maximum of two floats."""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_max_f32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 4.0, places=5)

  def test_v_min_u32(self):
    """V_MIN_U32 returns minimum of two unsigned integers."""
    st = run_program([
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 50),
      v_min_u32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 50)

  def test_v_max_u32(self):
    """V_MAX_U32 returns maximum of two unsigned integers."""
    st = run_program([
      v_mov_b32_e32(v[0], 100),
      v_mov_b32_e32(v[1], 50),
      v_max_u32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 100)


if __name__ == '__main__':
  unittest.main()
