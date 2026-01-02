#!/usr/bin/env python3
"""Tests for F16 operations - VOP2/VOPC hi-half encoding, sin kernel ops, modifiers."""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import RawImm
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f

VCC = SrcEnum.VCC_LO


class TestVOP2_16bit_HiHalf(unittest.TestCase):
  """Regression tests for VOP2 16-bit ops reading from high half of VGPR (v128+ encoding).

  Bug: VOP2 16-bit ops like v_add_f16 with src0 as v128+ should read the HIGH 16 bits
  of the corresponding VGPR (v128 = v0.hi, v129 = v1.hi, etc). The emulator was
  incorrectly reading from VGPR v128+ instead of the high half of v0+.

  Example: v_add_f16 v0, v128, v0 means v0.lo = v0.hi + v0.lo (fold packed result)
  """

  def test_v_add_f16_src0_hi_fold(self):
    """v_add_f16 with src0=v128 (v0.hi) - fold packed f16 values."""
    instructions = [
      # v0 = packed f16: high=2.0 (0x4000), low=1.0 (0x3c00)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v_add_f16 v1, v128, v0 means: v1.lo = v0.hi + v0.lo = 2.0 + 1.0 = 3.0
      # v128 in src0 means "read high 16 bits of v0"
      v_add_f16_e32(v[1], v[0].h, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0x4200, f"Expected 3.0 (0x4200), got 0x{result:04x}")

  def test_v_add_f16_src0_hi_different_reg(self):
    """v_add_f16 with src0=v129 (v1.hi) reads high half of v1."""
    instructions = [
      s_mov_b32(s[0], 0x44004200),  # v1: high=4.0, low=3.0
      v_mov_b32_e32(v[1], s[0]),
      s_mov_b32(s[1], 0x3c00),      # v0: low=1.0
      v_mov_b32_e32(v[0], s[1]),
      # v_add_f16 v2, v129, v0 means: v2.lo = v1.hi + v0.lo = 4.0 + 1.0 = 5.0
      v_add_f16_e32(v[2], v[1].h, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4500, f"Expected 5.0 (0x4500), got 0x{result:04x}")

  def test_v_mul_f16_src0_hi(self):
    """v_mul_f16 with src0 from high half."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # v0: high=2.0, low=1.0
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x4200),      # v1: low=3.0
      v_mov_b32_e32(v[1], s[1]),
      # v_mul_f16 v2, v128, v1 means: v2.lo = v0.hi * v1.lo = 2.0 * 3.0 = 6.0
      v_mul_f16_e32(v[2], v[0].h, v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4600, f"Expected 6.0 (0x4600), got 0x{result:04x}")


class TestVOPC_16bit_HiHalf(unittest.TestCase):
  """Regression tests for VOPC 16-bit ops reading from high half of VGPR (v128+ encoding)."""

  def test_v_cmp_lt_f16_vsrc1_hi(self):
    """v_cmp_lt_f16 comparing low half with high half of same register."""
    instructions = [
      # v0: high=2.0 (0x4000), low=1.0 (0x3c00)
      s_mov_b32(s[0], 0x40003c00),
      v_mov_b32_e32(v[0], s[0]),
      # v_cmp_lt_f16 vcc, v0, v128 means: vcc = (v0.lo < v0.hi) = (1.0 < 2.0) = true
      v_cmp_lt_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Expected vcc=1 (1.0 < 2.0)")

  def test_v_cmp_gt_f16_vsrc1_hi(self):
    """v_cmp_gt_f16 with vsrc1 from high half."""
    instructions = [
      # v0: high=1.0 (0x3c00), low=2.0 (0x4000)
      s_mov_b32(s[0], 0x3c004000),
      v_mov_b32_e32(v[0], s[0]),
      # v_cmp_gt_f16 vcc, v0, v128 means: vcc = (v0.lo > v0.hi) = (2.0 > 1.0) = true
      v_cmp_gt_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Expected vcc=1 (2.0 > 1.0)")

  def test_v_cmp_nge_f16_inf_self(self):
    """v_cmp_nge_f16 comparing -inf with itself (unordered less than).
    Regression test: -inf < -inf should be false (IEEE 754).
    """
    instructions = [
      # v0: both halves = -inf (0xFC00)
      s_mov_b32(s[0], 0xFC00FC00),
      v_mov_b32_e32(v[0], s[0]),
      # v_cmp_nge_f16 is "not greater or equal" - -inf nge -inf should be false
      v_cmp_nge_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Expected vcc=0 (-inf >= -inf)")


class TestF16SinKernelOps(unittest.TestCase):
  """Tests for F16 instructions used in the sin kernel."""

  def test_v_cvt_i16_f16_zero(self):
    """v_cvt_i16_f16: Convert f16 0.0 to i16 0."""
    instructions = [
      s_mov_b32(s[0], 0x00000000),  # f16 0.0 in low bits
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xFFFF
    self.assertEqual(result, 0, f"Expected 0, got {result}")

  def test_v_cvt_i16_f16_one(self):
    """v_cvt_i16_f16: Convert f16 1.0 (0x3c00) to i16 1."""
    instructions = [
      s_mov_b32(s[0], 0x00003c00),  # f16 1.0 in low bits
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i16_f16_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xFFFF
    self.assertEqual(result, 1, f"Expected 1, got {result}")

  def test_v_mul_f16_basic(self):
    """v_mul_f16: 2.0 * 3.0 = 6.0."""
    instructions = [
      s_mov_b32(s[0], 0x00004000),  # f16 2.0 in low bits
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x00004200),  # f16 3.0 in low bits
      v_mov_b32_e32(v[1], s[1]),
      v_mul_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xFFFF
    self.assertEqual(result, 0x4600, f"Expected 0x4600 (6.0), got 0x{result:04x}")

  def test_v_fmac_f16_basic(self):
    """v_fmac_f16: dst = src0 * src1 + dst = 2.0 * 3.0 + 1.0 = 7.0."""
    instructions = [
      s_mov_b32(s[0], 0x00004000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x00004200),  # f16 3.0
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], 0x00003c00),  # f16 1.0 (accumulator)
      v_mov_b32_e32(v[2], s[2]),
      v_fmac_f16_e32(v[2], v[0], v[1]),  # v2 = v0 * v1 + v2
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xFFFF
    self.assertEqual(result, 0x4700, f"Expected 0x4700 (7.0), got 0x{result:04x}")

  def test_v_add_f16_basic(self):
    """v_add_f16: 1.0 + 2.0 = 3.0."""
    instructions = [
      s_mov_b32(s[0], 0x00003c00),  # f16 1.0
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x00004000),  # f16 2.0
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xFFFF
    self.assertEqual(result, 0x4200, f"Expected 0x4200 (3.0), got 0x{result:04x}")

  def test_v_mov_b16_to_hi(self):
    """v_mov_b16: Move immediate to high half, preserving low."""
    instructions = [
      s_mov_b32(s[0], 0x0000DEAD),  # initial: lo=0xDEAD, hi=0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b16_e32(v[0].h, 0x3800),  # Move 0.5 to high half
    ]
    st = run_program(instructions, n_lanes=1)
    result_hi = (st.vgpr[0][0] >> 16) & 0xFFFF
    result_lo = st.vgpr[0][0] & 0xFFFF
    self.assertEqual(result_hi, 0x3800, f"Expected hi=0x3800, got 0x{result_hi:04x}")
    self.assertEqual(result_lo, 0xDEAD, f"Expected lo=0xDEAD (preserved), got 0x{result_lo:04x}")


class TestVOP3F16Modifiers(unittest.TestCase):
  """Tests for VOP3 16-bit ops with abs/neg modifiers and inline constants."""

  def test_v_cvt_f32_f16_abs_negative(self):
    """V_CVT_F32_F16 with |abs| on negative value."""
    from extra.assembly.amd.pcode import f32_to_f16
    f16_neg1 = f32_to_f16(-1.0)  # 0xbc00
    instructions = [
      s_mov_b32(s[0], f16_neg1),
      v_mov_b32_e32(v[1], s[0]),
      v_cvt_f32_f16_e64(v[0], abs(v[1])),  # |(-1.0)| = 1.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][0])
    self.assertAlmostEqual(result, 1.0, places=5, msg=f"Expected 1.0, got {result}")

  def test_v_cvt_f32_f16_neg_positive(self):
    """V_CVT_F32_F16 with neg on positive value."""
    from extra.assembly.amd.pcode import f32_to_f16
    f16_2 = f32_to_f16(2.0)  # 0x4000
    instructions = [
      s_mov_b32(s[0], f16_2),
      v_mov_b32_e32(v[1], s[0]),
      v_cvt_f32_f16_e64(v[0], -v[1]),  # -(2.0) = -2.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][0])
    self.assertAlmostEqual(result, -2.0, places=5, msg=f"Expected -2.0, got {result}")

  def test_v_fma_f16_inline_const_1_0(self):
    """V_FMA_F16: a*b + 1.0 should use f16 inline constant."""
    from extra.assembly.amd.pcode import f32_to_f16, _f16
    f16_a = f32_to_f16(0.325928)
    f16_b = f32_to_f16(-0.486572)
    instructions = [
      s_mov_b32(s[0], f16_a),
      v_mov_b32_e32(v[4], s[0]),
      s_mov_b32(s[1], f16_b),
      v_mov_b32_e32(v[6], s[1]),
      v_fma_f16(v[4], v[4], v[6], 1.0),  # 1.0 is inline constant
    ]
    st = run_program(instructions, n_lanes=1)
    result = _f16(st.vgpr[0][4] & 0xffff)
    expected = 0.325928 * (-0.486572) + 1.0
    self.assertAlmostEqual(result, expected, delta=0.01, msg=f"Expected ~{expected:.4f}, got {result}")


class TestVFmaMixSinCase(unittest.TestCase):
  """Tests for the specific V_FMA_MIXLO_F16 case that fails in AMD_LLVM sin(0) kernel."""

  def test_v_fma_mixlo_f16_sin_case(self):
    """V_FMA_MIXLO_F16 case from sin kernel at pc=0x14e."""
    from extra.assembly.amd.pcode import _f16
    instructions = [
      # Set up operands as in the sin kernel
      s_mov_b32(s[0], 0x3f800000),  # f32 1.0
      v_mov_b32_e32(v[3], s[0]),
      s_mov_b32(s[1], 0xaf05a309),  # f32 tiny negative
      s_mov_b32(s[6], s[1]),
      s_mov_b32(s[2], 0xc0490fdb),  # f32 -π
      v_mov_b32_e32(v[5], s[2]),
      # Pre-fill v3 with expected hi bits
      s_mov_b32(s[3], 0x3f800000),  # hi = f32 1.0 encoding
      v_mov_b32_e32(v[3], s[3]),
      # V_FMA_MIXLO_F16: src0=v3, src1=s6, src2=v5, opsel=0, opsel_hi=0, opsel_hi2=0
      VOP3P(VOP3POp.V_FMA_MIXLO_F16, vdst=v[3], src0=v[3], src1=s[6], src2=v[5], opsel=0, opsel_hi=0, opsel_hi2=0),
    ]
    st = run_program(instructions, n_lanes=1)
    lo = _f16(st.vgpr[0][3] & 0xffff)
    # Result should be approximately -π = -3.14...
    self.assertAlmostEqual(lo, -3.14159, delta=0.01, msg=f"Expected ~-π, got {lo}")


if __name__ == "__main__":
  unittest.main()
