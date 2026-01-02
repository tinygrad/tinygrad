#!/usr/bin/env python3
import math, unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f


class TestMad64(unittest.TestCase):
  """Tests for V_MAD_U64_U32 - critical for OCML Payne-Hanek sin reduction."""

  def test_v_mad_u64_u32_simple(self):
    """V_MAD_U64_U32: D = S0 * S1 + S2 (64-bit result)."""
    # 3 * 4 + 5 = 17
    instructions = [
      s_mov_b32(s[0], 3),
      s_mov_b32(s[1], 4),
      v_mov_b32_e32(v[2], 5),  # S2 lo
      v_mov_b32_e32(v[3], 0),  # S2 hi
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),  # result in v[4:5]
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    self.assertEqual(result, 17)

  def test_v_mad_u64_u32_large_mult(self):
    """V_MAD_U64_U32 with large values that overflow 32 bits."""
    # 0x80000000 * 2 + 0 = 0x100000000
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      s_mov_b32(s[1], 2),
      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[3], 0),
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    self.assertEqual(result, 0x100000000)

  def test_v_mad_u64_u32_with_add(self):
    """V_MAD_U64_U32 with 64-bit addend."""
    # 1000 * 1000 + 0x100000000 = 1000000 + 0x100000000 = 0x1000F4240
    instructions = [
      s_mov_b32(s[0], 1000),
      s_mov_b32(s[1], 1000),
      v_mov_b32_e32(v[2], 0),  # S2 lo
      v_mov_b32_e32(v[3], 1),  # S2 hi = 0x100000000
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    expected = 1000 * 1000 + 0x100000000
    self.assertEqual(result, expected)

  def test_v_mad_u64_u32_max_values(self):
    """V_MAD_U64_U32 with max u32 values."""
    # 0xFFFFFFFF * 0xFFFFFFFF + 0 = 0xFFFFFFFE00000001
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0xFFFFFFFF),
      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[3], 0),
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    expected = 0xFFFFFFFF * 0xFFFFFFFF
    self.assertEqual(result, expected)


class TestClz(unittest.TestCase):
  """Tests for V_CLZ_I32_U32 - count leading zeros, used in Payne-Hanek."""

  def test_v_clz_i32_u32_zero(self):
    """V_CLZ_I32_U32 of 0 returns -1 (all bits are 0)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # -1 as unsigned 32-bit
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_clz_i32_u32_one(self):
    """V_CLZ_I32_U32 of 1 returns 31 (31 leading zeros)."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)

  def test_v_clz_i32_u32_msb_set(self):
    """V_CLZ_I32_U32 of 0x80000000 returns 0 (no leading zeros)."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_clz_i32_u32_half(self):
    """V_CLZ_I32_U32 of 0x8000 (bit 15) returns 16."""
    instructions = [
      s_mov_b32(s[0], 0x8000),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 16)

  def test_v_clz_i32_u32_all_ones(self):
    """V_CLZ_I32_U32 of 0xFFFFFFFF returns 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)


class TestCtz(unittest.TestCase):
  """Tests for V_CTZ_I32_B32 - count trailing zeros."""

  def test_v_ctz_i32_b32_zero(self):
    """V_CTZ_I32_B32 of 0 returns -1 (all bits are 0)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_ctz_i32_b32_one(self):
    """V_CTZ_I32_B32 of 1 returns 0 (no trailing zeros)."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_ctz_i32_b32_msb_set(self):
    """V_CTZ_I32_B32 of 0x80000000 returns 31."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)

  def test_v_ctz_i32_b32_half(self):
    """V_CTZ_I32_B32 of 0x8000 (bit 15) returns 15."""
    instructions = [
      s_mov_b32(s[0], 0x8000),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15)

  def test_v_ctz_i32_b32_all_ones(self):
    """V_CTZ_I32_B32 of 0xFFFFFFFF returns 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)


class TestDivision(unittest.TestCase):
  """Tests for division instructions - V_RCP, V_DIV_SCALE, V_DIV_FMAS, V_DIV_FIXUP."""

  def test_v_rcp_f32_normal(self):
    """V_RCP_F32 of 2.0 returns 0.5."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_rcp_f32_inf(self):
    """V_RCP_F32 of +inf returns 0."""
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_rcp_f32_neg_inf(self):
    """V_RCP_F32 of -inf returns -0."""
    instructions = [
      s_mov_b32(s[0], 0xff800000),  # -inf
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertEqual(result, 0.0)
    # Check it's negative zero
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_rcp_f32_zero(self):
    """V_RCP_F32 of 0 returns +inf."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))

  def test_v_div_fixup_f32_normal(self):
    """V_DIV_FIXUP_F32 normal division 1.0/2.0."""
    # S0 = approximation (from rcp * scale), S1 = denominator, S2 = numerator
    instructions = [
      s_mov_b32(s[0], f2i(0.5)),   # approximation
      s_mov_b32(s[1], f2i(2.0)),   # denominator
      s_mov_b32(s[2], f2i(1.0)),   # numerator
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_div_fixup_f32_one_div_inf(self):
    """V_DIV_FIXUP_F32: 1.0 / +inf = 0."""
    # For x/inf: S0=approx(~0), S1=inf, S2=x
    instructions = [
      s_mov_b32(s[0], 0),           # approximation (rcp of inf = 0)
      s_mov_b32(s[1], 0x7f800000),  # denominator = +inf
      s_mov_b32(s[2], f2i(1.0)),    # numerator = 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_div_fixup_f32_one_div_neg_inf(self):
    """V_DIV_FIXUP_F32: 1.0 / -inf = -0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # approximation (rcp of -inf = -0)
      s_mov_b32(s[1], 0xff800000),  # denominator = -inf
      s_mov_b32(s[2], f2i(1.0)),    # numerator = 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x80000000)  # -0.0

  def test_v_div_fixup_f32_inf_div_inf(self):
    """V_DIV_FIXUP_F32: inf / inf = NaN."""
    instructions = [
      s_mov_b32(s[0], 0),           # approximation
      s_mov_b32(s[1], 0x7f800000),  # denominator = +inf
      s_mov_b32(s[2], 0x7f800000),  # numerator = +inf
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_div_fixup_f32_zero_div_zero(self):
    """V_DIV_FIXUP_F32: 0 / 0 = NaN."""
    instructions = [
      s_mov_b32(s[0], 0),  # approximation
      s_mov_b32(s[1], 0),  # denominator = 0
      s_mov_b32(s[2], 0),  # numerator = 0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_div_fixup_f32_x_div_zero(self):
    """V_DIV_FIXUP_F32: 1.0 / 0 = +inf."""
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # approximation (rcp of 0 = inf)
      s_mov_b32(s[1], 0),           # denominator = 0
      s_mov_b32(s[2], f2i(1.0)),    # numerator = 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertTrue(math.isinf(result) and result > 0)

  def test_v_div_fixup_f32_neg_x_div_zero(self):
    """V_DIV_FIXUP_F32: -1.0 / 0 = -inf."""
    instructions = [
      s_mov_b32(s[0], 0xff800000),  # approximation (rcp of 0 = inf, with sign)
      s_mov_b32(s[1], 0),           # denominator = 0
      s_mov_b32(s[2], f2i(-1.0)),   # numerator = -1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertTrue(math.isinf(result) and result < 0)


if __name__ == "__main__":
  unittest.main()
