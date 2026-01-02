#!/usr/bin/env python3
"""Tests for type conversion instructions."""
import unittest, struct
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import RawImm
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f, f2i64, i642f

class TestF32Conversions(unittest.TestCase):
  """Tests for f32 conversion instructions."""

  def test_v_cvt_i32_f32_positive(self):
    """V_CVT_I32_F32 converts float to signed int."""
    st = run_program([
      s_mov_b32(s[0], f2i(42.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 42)

  def test_v_cvt_i32_f32_negative(self):
    """V_CVT_I32_F32 converts negative float to signed int."""
    st = run_program([
      s_mov_b32(s[0], f2i(-42.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1] & 0xffffffff, (-42) & 0xffffffff)

  def test_v_cvt_f32_i32(self):
    """V_CVT_F32_I32 converts signed int to float."""
    st = run_program([
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_i32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 42.0, places=5)

  def test_v_cvt_f32_u32(self):
    """V_CVT_F32_U32 converts unsigned int to float."""
    st = run_program([
      s_mov_b32(s[0], 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_u32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4294967296.0, places=-5)

  def test_v_cvt_u32_f32(self):
    """V_CVT_U32_F32 converts float to unsigned int."""
    st = run_program([
      s_mov_b32(s[0], f2i(100.9)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_u32_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 100)


class TestF16Conversions(unittest.TestCase):
  """Tests for f16 conversion and packing instructions."""

  def test_v_cvt_f16_f32_basic(self):
    """V_CVT_F16_F32 converts f32 to f16 in low 16 bits."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    lo_bits = st.vgpr[0][1] & 0xffff
    self.assertEqual(lo_bits, 0x3c00)

  def test_v_cvt_f16_f32_negative(self):
    """V_CVT_F16_F32 converts negative f32 to f16."""
    st = run_program([
      v_mov_b32_e32(v[0], -2.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    lo_bits = st.vgpr[0][1] & 0xffff
    self.assertEqual(lo_bits, 0xc000)

  def test_v_cvt_f16_f32_preserves_high_bits(self):
    """V_CVT_F16_F32 preserves high 16 bits of destination."""
    st = run_program([
      s_mov_b32(s[0], 0xdead0000),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[0], 1.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    hi_bits = (st.vgpr[0][1] >> 16) & 0xffff
    lo_bits = st.vgpr[0][1] & 0xffff
    self.assertEqual(lo_bits, 0x3c00)
    self.assertEqual(hi_bits, 0xdead)

  def test_v_cvt_f32_f16(self):
    """V_CVT_F32_F16 converts f16 to f32."""
    st = run_program([
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_f16_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=5)

  def test_v_pack_b32_f16(self):
    """V_PACK_B32_F16 packs two f16 values into one 32-bit register."""
    from extra.assembly.amd.pcode import _f16
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[2], -2.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
      v_cvt_f16_f32_e32(v[3], v[2]),
      v_pack_b32_f16(v[4], v[1], v[3]),
    ], n_lanes=1)
    lo_bits = st.vgpr[0][4] & 0xffff
    hi_bits = (st.vgpr[0][4] >> 16) & 0xffff
    self.assertEqual(lo_bits, 0x3c00)
    self.assertEqual(hi_bits, 0xc000)


class TestF64Conversions(unittest.TestCase):
  """Tests for 64-bit float operations and conversions."""

  def test_v_add_f64_inline_constant(self):
    """V_ADD_F64 with inline constant."""
    one_f64 = f2i64(1.0)
    st = run_program([
      s_mov_b32(s[0], one_f64 & 0xffffffff),
      s_mov_b32(s[1], one_f64 >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f64(v[2:4], v[0:2], SrcEnum.POS_ONE),
    ], n_lanes=1)
    result = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    self.assertAlmostEqual(result, 2.0, places=5)

  def test_v_cvt_i32_f64_writes_32bit(self):
    """V_CVT_I32_F64 should only write 32 bits, not 64."""
    val_bits = f2i64(-1.0)
    st = run_program([
      s_mov_b32(s[0], val_bits & 0xffffffff),
      s_mov_b32(s[1], val_bits >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[3], s[2]),
      v_cvt_i32_f64_e32(v[2], v[0:2]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xffffffff)
    self.assertEqual(st.vgpr[0][3], 0xDEADBEEF)

  def test_v_frexp_mant_f64_range(self):
    """V_FREXP_MANT_F64 should return mantissa in [0.5, 1.0) range."""
    two_f64 = f2i64(2.0)
    st = run_program([
      s_mov_b32(s[0], two_f64 & 0xffffffff),
      s_mov_b32(s[1], two_f64 >> 32),
      v_frexp_mant_f64_e32(v[0:2], s[0:2]),
      v_frexp_exp_i32_f64_e32(v[2], s[0:2]),
    ], n_lanes=1)
    mant = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    exp = st.vgpr[0][2]
    if exp >= 0x80000000: exp -= 0x100000000
    self.assertAlmostEqual(mant, 0.5, places=10)
    self.assertEqual(exp, 2)

  def test_f64_to_i64_full_sequence(self):
    """Full f64->i64 conversion sequence."""
    val = f2i64(-41.0)
    lit = 0xC1F00000
    st = run_program([
      s_mov_b32(s[0], val & 0xffffffff),
      s_mov_b32(s[1], (val >> 32) & 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_trunc_f64_e32(v[0:2], v[0:2]),
      v_ldexp_f64(v[2:4], v[0:2], 0xFFFFFFE0),
      v_floor_f64_e32(v[2:4], v[2:4]),
      VOP3(VOP3Op.V_FMA_F64, vdst=v[0], src0=RawImm(255), src1=v[2], src2=v[0], literal=lit),
      v_cvt_u32_f64_e32(v[4], v[0:2]),
      v_cvt_i32_f64_e32(v[5], v[2:4]),
    ], n_lanes=1)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, -41)


class TestClzCtz(unittest.TestCase):
  """Tests for count leading/trailing zeros."""

  def test_v_clz_i32_u32_zero(self):
    """V_CLZ_I32_U32 of 0 returns -1."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      v_clz_i32_u32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_clz_i32_u32_one(self):
    """V_CLZ_I32_U32 of 1 returns 31."""
    st = run_program([
      v_mov_b32_e32(v[0], 1),
      v_clz_i32_u32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)

  def test_v_clz_i32_u32_msb_set(self):
    """V_CLZ_I32_U32 of 0x80000000 returns 0."""
    st = run_program([
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_ctz_i32_b32_zero(self):
    """V_CTZ_I32_B32 of 0 returns -1."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_ctz_i32_b32_one(self):
    """V_CTZ_I32_B32 of 1 returns 0."""
    st = run_program([
      v_mov_b32_e32(v[0], 1),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_ctz_i32_b32_msb_set(self):
    """V_CTZ_I32_B32 of 0x80000000 returns 31."""
    st = run_program([
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)


if __name__ == '__main__':
  unittest.main()
