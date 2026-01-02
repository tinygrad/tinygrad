#!/usr/bin/env python3
"""Tests for division-related instructions: V_DIV_SCALE, V_DIV_FMAS, V_DIV_FIXUP, V_RCP."""
import unittest, math
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f, VCC

class TestVDivScale(unittest.TestCase):
  """Tests for V_DIV_SCALE_F32 edge cases.

  V_DIV_SCALE_F32 is used in Newton-Raphson division to handle denormals and near-overflow.
  Scales operands and sets VCC when final result needs unscaling.
  """

  def test_vcc_zero_no_scaling(self):
    """VCC=0 when no scaling needed."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 4.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vcc, 0)

  def test_vcc_zero_multiple_lanes(self):
    """VCC=0 for all lanes when no scaling needed."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 4.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ], n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0)

  def test_preserves_input(self):
    """Outputs S0 when no scaling needed."""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 2.0, places=5)

  def test_zero_denom_gives_nan(self):
    """Zero denominator -> NaN, VCC=1."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ], n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))
    self.assertEqual(st.vcc & 1, 1)

  def test_zero_numer_gives_nan(self):
    """Zero numerator -> NaN, VCC=1."""
    st = run_program([
      v_mov_b32_e32(v[0], 0.0),
      v_mov_b32_e32(v[1], 1.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ], n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))
    self.assertEqual(st.vcc & 1, 1)

  def test_large_exp_diff_scales_denom(self):
    """exp(numer) - exp(denom) >= 96 -> scale denom, VCC=1."""
    max_float = 0x7f7fffff
    st = run_program([
      s_mov_b32(s[0], max_float),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_div_scale_f32(v[2], VCC, v[1], v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(st.vcc & 1, 1)
    expected = 1.0 * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), expected, delta=expected * 1e-6)

  def test_denorm_denom_gives_nan(self):
    """Denormalized denominator -> NaN, VCC=1."""
    denorm = 0x00000001
    st = run_program([
      s_mov_b32(s[0], denorm),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], s[0]),
      v_div_scale_f32(v[2], VCC, v[1], v[1], v[0]),
    ], n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))
    self.assertEqual(st.vcc & 1, 1)

  def test_tiny_numer_scales(self):
    """exponent(numer) <= 23 -> scale by 2^64, VCC=1."""
    smallest_normal = 0x00800000
    st = run_program([
      s_mov_b32(s[0], smallest_normal),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ], n_lanes=1)
    numer_f = i2f(smallest_normal)
    expected = numer_f * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), expected, delta=abs(expected) * 1e-5)
    self.assertEqual(st.vcc & 1, 1)


class TestVDivFmas(unittest.TestCase):
  """Tests for V_DIV_FMAS_F32 edge cases.

  V_DIV_FMAS_F32 performs FMA with optional scaling based on VCC.
  Scale direction depends on S2's exponent.
  """

  def test_no_scale(self):
    """VCC=0 -> normal FMA."""
    st = run_program([
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 0),
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 3.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 7.0, places=5)

  def test_scale_up(self):
    """VCC=1 with S2 >= 2.0 -> scale by 2^+64."""
    st = run_program([
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 1),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 2.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    expected = 3.0 * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), expected, delta=abs(expected) * 1e-6)

  def test_scale_down(self):
    """VCC=1 with S2 < 2.0 -> scale by 2^-64."""
    st = run_program([
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 1),
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 3.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    expected = 7.0 * (2.0 ** -64)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), expected, delta=abs(expected) * 1e-6)

  def test_per_lane_vcc(self):
    """Different VCC per lane with S2 < 2.0."""
    st = run_program([
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 0b0101),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=4)
    scaled = 2.0 * (2.0 ** -64)
    unscaled = 2.0
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), scaled, delta=abs(scaled) * 1e-6)
    self.assertAlmostEqual(i2f(st.vgpr[1][3]), unscaled, places=5)
    self.assertAlmostEqual(i2f(st.vgpr[2][3]), scaled, delta=abs(scaled) * 1e-6)
    self.assertAlmostEqual(i2f(st.vgpr[3][3]), unscaled, places=5)


class TestVDivFixup(unittest.TestCase):
  """Tests for V_DIV_FIXUP_F32 - final step of Newton-Raphson division.

  Args: S0=quotient from NR iteration, S1=denominator, S2=numerator
  """

  def test_normal(self):
    """Normal division passes through quotient."""
    st = run_program([
      v_mov_b32_e32(v[0], 3.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 6.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 3.0, places=5)

  def test_nan_numer(self):
    """NaN numerator -> quiet NaN."""
    nan = 0x7fc00000
    st = run_program([
      s_mov_b32(s[0], nan),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], s[0]),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])))

  def test_zero_div_zero(self):
    """0/0 -> NaN."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_mov_b32_e32(v[2], 0.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])))

  def test_inf_div_inf(self):
    """inf/inf -> NaN."""
    pos_inf = 0x7f800000
    st = run_program([
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], s[0]),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])))

  def test_x_div_zero(self):
    """x/0 -> +inf."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])))
    self.assertGreater(i2f(st.vgpr[0][3]), 0)

  def test_neg_x_div_zero(self):
    """-x/0 -> -inf."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 0.0),
      v_mov_b32_e32(v[2], -1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])))
    self.assertLess(i2f(st.vgpr[0][3]), 0)

  def test_zero_div_x(self):
    """0/x -> 0."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_mov_b32_e32(v[2], 0.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][3]), 0.0)

  def test_x_div_inf(self):
    """x/inf -> 0."""
    pos_inf = 0x7f800000
    st = run_program([
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][3]), 0.0)

  def test_inf_div_x(self):
    """inf/x -> inf."""
    pos_inf = 0x7f800000
    st = run_program([
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], s[0]),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])))

  def test_sign_propagation(self):
    """Sign is XOR of numer and denom signs."""
    st = run_program([
      v_mov_b32_e32(v[0], 3.0),
      v_mov_b32_e32(v[1], -2.0),
      v_mov_b32_e32(v[2], 6.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), -3.0, places=5)

  def test_nan_estimate_overflow(self):
    """NaN estimate returns overflow (inf)."""
    quiet_nan = 0x7fc00000
    st = run_program([
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 1.0),
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ], n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])))
    self.assertEqual(st.vgpr[0][3], 0x7f800000)


class TestVRcp(unittest.TestCase):
  """Tests for V_RCP_F32 reciprocal instruction."""

  def test_normal(self):
    """rcp(2.0) = 0.5"""
    st = run_program([
      v_mov_b32_e32(v[0], 2.0),
      v_rcp_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_inf(self):
    """rcp(+inf) = 0."""
    st = run_program([
      s_mov_b32(s[0], 0x7f800000),
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_neg_inf(self):
    """rcp(-inf) = -0."""
    st = run_program([
      s_mov_b32(s[0], 0xff800000),
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_zero(self):
    """rcp(0) = +inf."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      v_rcp_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))

  def test_one(self):
    """rcp(1.0) = 1.0."""
    st = run_program([
      v_mov_b32_e32(v[0], 1.0),
      v_rcp_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 1.0, places=5)

  def test_negative(self):
    """rcp(-4.0) = -0.25."""
    st = run_program([
      v_mov_b32_e32(v[0], -4.0),
      v_rcp_f32_e32(v[1], v[0]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -0.25, places=5)

  def test_small(self):
    """rcp(0.125) = 8.0."""
    st = run_program([
      v_mov_b32_e32(v[0], 0.5),
      v_rcp_f32_e32(v[1], v[0]),
      v_rcp_f32_e32(v[2], v[1]),
    ], n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 0.5, places=5)


if __name__ == '__main__':
  unittest.main()
