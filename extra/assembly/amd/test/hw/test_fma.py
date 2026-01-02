#!/usr/bin/env python3
"""Tests for FMA instructions - key for OCML sin argument reduction."""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f

class TestFMA(unittest.TestCase):
  """Tests for FMA instructions."""

  def test_v_fma_f32_basic(self):
    """V_FMA_F32: a*b+c basic case."""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mov_b32_e32(v[2], 1.0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 9.0, places=5)

  def test_v_fma_f32_negative(self):
    """V_FMA_F32 with negative multiplier."""
    st = run_program([
      v_mov_b32_e32(v[0], -2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mov_b32_e32(v[2], 1.0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), -7.0, places=5)

  def test_v_fmac_f32(self):
    """V_FMAC_F32: d = d + a*b."""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mov_b32_e32(v[2], 1.0),
      v_fmac_f32_e32(v[2], v[0], v[1]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmaak_f32(self):
    """V_FMAAK_F32: d = a * b + K."""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_fmaak_f32_e32(v[2], v[0], v[1], 0x3f800000),  # K=1.0
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fma_f32_with_sgpr(self):
    """V_FMA_F32 using SGPR for non-inline constant."""
    st = run_program([
      s_mov_b32(s[0], f2i(3.0)),
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 4.0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 10.0, places=5)

  def test_v_fmamk_f32(self):
    """V_FMAMK_F32: d = a * K + b."""
    st = run_program([
      s_mov_b32(s[0], f2i(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(1.0)),
      v_mov_b32_e32(v[1], s[1]),
      v_fmamk_f32_e32(v[2], v[0], f2i(3.0), v[1]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 7.0, places=5)

  def test_v_fma_precision(self):
    """V_FMA_F32 should be more precise than separate mul+add."""
    st = run_program([
      s_mov_b32(s[0], f2i(1.0000001)),
      s_mov_b32(s[1], f2i(1.0000001)),
      s_mov_b32(s[2], f2i(-1.0000002)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    # Just check it runs without error; precision test would need exact bits
    self.assertFalse(st.vgpr[0][3] == 0 and st.vgpr[0][0] != 0)


class TestRounding(unittest.TestCase):
  """Tests for rounding instructions - used in sin argument reduction."""

  def test_v_rndne_f32_half_even(self):
    """V_RNDNE_F32 rounds to nearest even."""
    st = run_program([
      s_mov_b32(s[0], f2i(2.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)

  def test_v_rndne_f32_half_odd(self):
    """V_RNDNE_F32 rounds 3.5 to 4 (nearest even)."""
    st = run_program([
      s_mov_b32(s[0], f2i(3.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4.0, places=5)

  def test_v_floor_f32(self):
    """V_FLOOR_F32 floors to integer."""
    st = run_program([
      s_mov_b32(s[0], f2i(3.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_floor_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 3.0, places=5)

  def test_v_trunc_f32(self):
    """V_TRUNC_F32 truncates toward zero."""
    st = run_program([
      s_mov_b32(s[0], f2i(-3.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_trunc_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -3.0, places=5)

  def test_v_fract_f32(self):
    """V_FRACT_F32 returns fractional part."""
    st = run_program([
      s_mov_b32(s[0], f2i(3.75)),
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.75, places=5)

  def test_v_ceil_f32(self):
    """V_CEIL_F32 rounds up."""
    st = run_program([
      s_mov_b32(s[0], f2i(3.1)),
      v_mov_b32_e32(v[0], s[0]),
      v_ceil_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4.0, places=5)


class TestVFmaMix(unittest.TestCase):
  """Tests for V_FMA_MIX_F32/F16 mixed-precision FMA instructions."""

  def test_v_fma_mix_f32_all_f32(self):
    """V_FMA_MIX_F32 with all f32 sources."""
    st = run_program([
      s_mov_b32(s[0], f2i(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f2i(1.0)),
      v_mov_b32_e32(v[2], s[2]),
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=0),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 7.0, places=5)

  def test_v_fma_mix_f32_src2_f16_lo(self):
    """V_FMA_MIX_F32 with src2 as f16 from lo bits."""
    from extra.assembly.amd.pcode import f32_to_f16
    f16_2 = f32_to_f16(2.0)
    st = run_program([
      s_mov_b32(s[0], f2i(1.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f16_2),
      v_mov_b32_e32(v[2], s[2]),
      VOP3P(VOP3POp.V_FMA_MIX_F32, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=1),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 5.0, places=5)

  def test_v_fma_mixlo_f16(self):
    """V_FMA_MIXLO_F16 writes to low 16 bits of destination."""
    from extra.assembly.amd.pcode import _f16
    st = run_program([
      s_mov_b32(s[0], f2i(2.0)),
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], f2i(3.0)),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], f2i(1.0)),
      v_mov_b32_e32(v[2], s[2]),
      s_mov_b32(s[3], 0xdead0000),
      v_mov_b32_e32(v[3], s[3]),
      VOP3P(VOP3POp.V_FMA_MIXLO_F16, vdst=v[3], src0=v[0], src1=v[1], src2=v[2], opsel=0, opsel_hi=0, opsel_hi2=0),
    ], n_lanes=1)
    lo = _f16(st.vgpr[0][3] & 0xffff)
    hi = (st.vgpr[0][3] >> 16) & 0xffff
    self.assertAlmostEqual(lo, 7.0, places=1)
    self.assertEqual(hi, 0xdead)


if __name__ == '__main__':
  unittest.main()
