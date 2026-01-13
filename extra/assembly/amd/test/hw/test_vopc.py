"""Tests for VOPC instructions - vector compare operations.

Includes: v_cmp_class_f32, v_cmp_class_f16, v_cmp_eq_*, v_cmp_lt_*, v_cmp_gt_*
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

VCC = 106  # SGPR index for VCC_LO

class TestCmpClass(unittest.TestCase):
  """Tests for V_CMP_CLASS_F32 float classification."""

  def test_cmp_class_quiet_nan(self):
    """V_CMP_CLASS_F32 detects quiet NaN."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect quiet NaN")

  def test_cmp_class_signaling_nan(self):
    """V_CMP_CLASS_F32 detects signaling NaN."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect signaling NaN")

  def test_cmp_class_positive_inf(self):
    """V_CMP_CLASS_F32 detects +inf."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      s_mov_b32(s[1], 0b1000000000),  # bit 9 = +inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect +inf")

  def test_cmp_class_negative_inf(self):
    """V_CMP_CLASS_F32 detects -inf."""
    neg_inf = 0xff800000
    instructions = [
      s_mov_b32(s[0], neg_inf),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000100),  # bit 2 = -inf
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect -inf")

  def test_cmp_class_normal_positive(self):
    """V_CMP_CLASS_F32 detects positive normal."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[1], 0b0100000000),  # bit 8 = positive normal
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive normal")

  def test_cmp_class_normal_negative(self):
    """V_CMP_CLASS_F32 detects negative normal."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),
      v_mov_b32_e32(v[1], 0b0000001000),  # bit 3 = negative normal
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative normal")

  def test_cmp_class_quiet_nan_not_signaling(self):
    """Quiet NaN does not match signaling NaN mask."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN only
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Quiet NaN should not match signaling mask")

  def test_cmp_class_signaling_nan_not_quiet(self):
    """Signaling NaN does not match quiet NaN mask."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN only
      v_cmp_class_f32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Signaling NaN should not match quiet mask")

  def test_v_cmp_sets_vcc_bits(self):
    """V_CMP_EQ sets VCC bits based on per-lane comparison."""
    instructions = [
      s_mov_b32(s[0], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_eq_u32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should match")


class TestCmpClassF16(unittest.TestCase):
  """Tests for V_CMP_CLASS_F16 float classification.

  Class bit mapping:
    bit 0 = signaling NaN
    bit 1 = quiet NaN
    bit 2 = -infinity
    bit 3 = -normal
    bit 4 = -denormal
    bit 5 = -zero
    bit 6 = +zero
    bit 7 = +denormal
    bit 8 = +normal
    bit 9 = +infinity
  """

  def test_cmp_class_f16_positive_zero(self):
    """V_CMP_CLASS_F16: +zero matches bit 6."""
    instructions = [
      v_mov_b32_e32(v[0], 0x0000),  # f16 +0.0
      v_mov_b32_e32(v[1], 0x40),     # bit 6 = +zero
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive zero")

  def test_cmp_class_f16_negative_zero(self):
    """V_CMP_CLASS_F16: -zero matches bit 5."""
    instructions = [
      s_mov_b32(s[0], 0x8000),       # f16 -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x20),     # bit 5 = -zero
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative zero")

  def test_cmp_class_f16_positive_normal(self):
    """V_CMP_CLASS_F16: +1.0 (normal) matches bit 8."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),       # f16 +1.0
      s_mov_b32(s[1], 0x100),        # bit 8 = +normal
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive normal")

  def test_cmp_class_f16_negative_normal(self):
    """V_CMP_CLASS_F16: -1.0 (normal) matches bit 3."""
    instructions = [
      s_mov_b32(s[0], 0xbc00),       # f16 -1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x08),     # bit 3 = -normal
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative normal")

  def test_cmp_class_f16_positive_infinity(self):
    """V_CMP_CLASS_F16: +inf matches bit 9."""
    instructions = [
      s_mov_b32(s[0], 0x7c00),       # f16 +inf
      s_mov_b32(s[1], 0x200),        # bit 9 = +inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive infinity")

  def test_cmp_class_f16_negative_infinity(self):
    """V_CMP_CLASS_F16: -inf matches bit 2."""
    instructions = [
      s_mov_b32(s[0], 0xfc00),       # f16 -inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x04),     # bit 2 = -inf
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative infinity")

  def test_cmp_class_f16_quiet_nan(self):
    """V_CMP_CLASS_F16: quiet NaN matches bit 1."""
    instructions = [
      s_mov_b32(s[0], 0x7e00),       # f16 quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x02),     # bit 1 = quiet NaN
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect quiet NaN")

  def test_cmp_class_f16_signaling_nan(self):
    """V_CMP_CLASS_F16: signaling NaN matches bit 0."""
    instructions = [
      s_mov_b32(s[0], 0x7c01),       # f16 signaling NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x01),     # bit 0 = signaling NaN
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect signaling NaN")

  def test_cmp_class_f16_positive_denormal(self):
    """V_CMP_CLASS_F16: positive denormal matches bit 7."""
    instructions = [
      v_mov_b32_e32(v[0], 1),        # f16 +denormal (0x0001)
      v_mov_b32_e32(v[1], 0x80),     # bit 7 = +denormal
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive denormal")

  def test_cmp_class_f16_negative_denormal(self):
    """V_CMP_CLASS_F16: negative denormal matches bit 4."""
    instructions = [
      s_mov_b32(s[0], 0x8001),       # f16 -denormal
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x10),     # bit 4 = -denormal
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative denormal")

  def test_cmp_class_f16_combined_mask_zeros(self):
    """V_CMP_CLASS_F16: mask 0x60 covers both +zero and -zero."""
    instructions = [
      v_mov_b32_e32(v[0], 0),         # f16 +0.0
      v_mov_b32_e32(v[1], 0x60),      # bits 5 and 6 (+-zero)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with mask 0x60")

  def test_cmp_class_f16_combined_mask_1f8(self):
    """V_CMP_CLASS_F16: mask 0x1f8 covers -normal,-denorm,-zero,+zero,+denorm,+normal.

    This is the exact mask used in the f16 sin kernel at PC=46.
    """
    instructions = [
      v_mov_b32_e32(v[0], 0),         # f16 +0.0
      s_mov_b32(s[0], 0x1f8),
      v_mov_b32_e32(v[1], s[0]),      # mask 0x1f8
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with mask 0x1f8")

  def test_cmp_class_f16_vop3_encoding(self):
    """V_CMP_CLASS_F16 in VOP3 encoding (v_cmp_class_f16_e64)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),         # f16 +0.0
      s_mov_b32(s[0], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with VOP3 encoding")

  def test_cmp_class_f16_vop3_normal_positive(self):
    """V_CMP_CLASS_F16 VOP3 encoding with +1.0 (normal)."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),        # f16 +1.0
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +1.0 (normal) with mask 0x1f8")

  def test_cmp_class_f16_vop3_nan_fails_mask(self):
    """V_CMP_CLASS_F16 VOP3: NaN should NOT match mask 0x1f8 (no NaN bits set)."""
    instructions = [
      s_mov_b32(s[0], 0x7e00),        # f16 quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "VCC should be 0 for NaN with mask 0x1f8 (no NaN bits)")

  def test_cmp_class_f16_vop3_inf_fails_mask(self):
    """V_CMP_CLASS_F16 VOP3: +inf should NOT match mask 0x1f8 (no inf bits set)."""
    instructions = [
      s_mov_b32(s[0], 0x7c00),        # f16 +inf
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),         # class mask
      v_cmp_class_f16_e64(VCC_LO, v[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "VCC should be 0 for +inf with mask 0x1f8 (no inf bits)")


class TestCmpInt(unittest.TestCase):
  """Tests for integer comparison operations."""

  def test_v_cmp_eq_u32(self):
    """V_CMP_EQ_U32 sets VCC bits based on per-lane comparison."""
    instructions = [
      s_mov_b32(s[0], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_eq_u32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should match")

  def test_cmp_eq_u16_opsel_lo_lo(self):
    """V_CMP_EQ_U16 comparing lo halves."""
    instructions = [
      s_mov_b32(s[0], 0x12340005),  # lo=5, hi=0x1234
      s_mov_b32(s[1], 0xABCD0005),  # lo=5, hi=0xABCD
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_eq_u16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Lo halves should be equal")

  def test_cmp_eq_u16_opsel_hi_hi(self):
    """V_CMP_EQ_U16 comparing hi halves with VOP3 opsel."""
    instructions = [
      s_mov_b32(s[2], 0x00051234),  # hi=5, lo=0x1234
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x0005ABCD),  # hi=5, lo=0xABCD
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_eq_u16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "Hi halves should be equal: 5==5")

  def test_cmp_eq_u16_opsel_hi_hi_equal(self):
    """V_CMP_EQ_U16 VOP3 with opsel=3 compares hi halves (equal case)."""
    instructions = [
      s_mov_b32(s[2], 0x12340005),  # lo=5, hi=0x1234
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x12340009),  # lo=9, hi=0x1234
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_eq_u16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "hi==hi should be true: 0x1234==0x1234")

  def test_cmp_gt_u16_opsel_hi(self):
    """V_CMP_GT_U16 VOP3 with opsel=3 compares hi halves."""
    instructions = [
      s_mov_b32(s[2], 0x99990005),  # lo=5, hi=0x9999
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x12340005),  # lo=5, hi=0x1234
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_gt_u16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "hi>hi should be true: 0x9999>0x1234")


class TestCmpFloat(unittest.TestCase):
  """Tests for float comparison operations."""

  def test_v_cmp_lt_f16_vsrc1_hi(self):
    """V_CMP_LT_F16 with both operands from high half using VOP3 opsel."""
    instructions = [
      s_mov_b32(s[2], 0x3c000000),  # hi=1.0 (f16), lo=0
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x40000000),  # hi=2.0 (f16), lo=0
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_lt_f16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "1.0 < 2.0 should be true")

  def test_v_cmp_gt_f16_vsrc1_hi(self):
    """V_CMP_GT_F16 with both operands from high half using VOP3 opsel."""
    instructions = [
      s_mov_b32(s[2], 0x40000000),  # hi=2.0 (f16), lo=0
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x3c000000),  # hi=1.0 (f16), lo=0
      v_mov_b32_e32(v[1], s[2]),
      v_cmp_gt_f16_e64(vdst=s[0], src0=v[0], src1=v[1], opsel=3),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[0] & 1, 1, "2.0 > 1.0 should be true")

  def test_v_cmp_eq_f16_vsrc1_hi_equal(self):
    """v_cmp_eq_f16 with equal low and high halves."""
    instructions = [
      s_mov_b32(s[0], 0x42004200),  # hi=3.0 (0x4200), lo=3.0 (0x4200)
      v_mov_b32_e32(v[0], s[0]),
      v_cmp_eq_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Expected vcc=1 (3.0 == 3.0)")

  def test_v_cmp_neq_f16_vsrc1_hi(self):
    """v_cmp_neq_f16 with different low and high halves."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # hi=2.0 (0x4000), lo=1.0 (0x3c00)
      v_mov_b32_e32(v[0], s[0]),
      v_cmp_lg_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Expected vcc=1 (1.0 != 2.0)")

  def test_v_cmp_nge_f16_inf_self(self):
    """v_cmp_nge_f16 comparing -inf with itself (unordered less than).

    Regression test: -inf < -inf should be false (IEEE 754).
    """
    instructions = [
      s_mov_b32(s[0], 0xFC00FC00),  # both halves = -inf (0xFC00)
      v_mov_b32_e32(v[0], s[0]),
      v_cmp_nge_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Expected vcc=0 (-inf >= -inf)")

  def test_v_cmp_f16_multilane(self):
    """v_cmp_lt_f16 with vsrc1=v128 across multiple lanes."""
    instructions = [
      # Lane 0: v0 = 0x40003c00 (hi=2.0, lo=1.0) -> 1.0 < 2.0 = true
      # Lane 1: v0 = 0x3c004000 (hi=1.0, lo=2.0) -> 2.0 < 1.0 = false
      v_mov_b32_e32(v[0], 0x40003c00),  # default
      v_cmp_eq_u32_e32(1, v[255]),  # vcc = (lane == 1)
      v_cndmask_b32_e64(v[0], v[0], 0x3c004000, SrcEnum.VCC_LO),
      v_cmp_lt_f16_e32(v[0], v[0].h),
    ]
    st = run_program(instructions, n_lanes=2)
    self.assertEqual(st.vcc & 1, 1, "Lane 0: expected vcc=1 (1.0 < 2.0)")
    self.assertEqual((st.vcc >> 1) & 1, 0, "Lane 1: expected vcc=0 (2.0 < 1.0)")


class TestVCCBehavior(unittest.TestCase):
  """Tests for VCC condition code behavior."""

  def test_vcc_all_lanes_true(self):
    """VCC should have all bits set when all lanes compare true."""
    instructions = [
      v_mov_b32_e32(v[0], 5),
      v_mov_b32_e32(v[1], 5),
      v_cmp_eq_u32_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=32)
    self.assertEqual(st.vcc, 0xFFFFFFFF, "All 32 lanes should be true")

  def test_vcc_lane_dependent(self):
    """VCC should differ per lane based on lane_id comparison."""
    instructions = [
      v_mov_b32_e32(v[0], 16),
      v_cmp_lt_u32_e32(v[255], v[0]),  # lanes 0-15 are < 16
    ]
    st = run_program(instructions, n_lanes=32)
    self.assertEqual(st.vcc & 0xFFFF, 0xFFFF, "Lanes 0-15 should be true")
    self.assertEqual(st.vcc >> 16, 0x0000, "Lanes 16-31 should be false")


if __name__ == '__main__':
  unittest.main()
