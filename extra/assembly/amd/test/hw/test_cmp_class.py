#!/usr/bin/env python3
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import RawImm
from extra.assembly.amd.test.hw.helpers import run_program

VCC = SrcEnum.VCC_LO

class TestVCmpClass(unittest.TestCase):
  """Tests for V_CMP_CLASS_F32 float classification."""

  def test_cmp_class_quiet_nan(self):
    """V_CMP_CLASS_F32 detects quiet NaN."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN (mask in VGPR for VOPC)
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect quiet NaN")

  def test_cmp_class_signaling_nan(self):
    """V_CMP_CLASS_F32 detects signaling NaN."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect signaling NaN")

  def test_cmp_class_quiet_nan_not_signaling(self):
    """Quiet NaN does not match signaling NaN mask."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN only
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Quiet NaN should not match signaling mask")

  def test_cmp_class_signaling_nan_not_quiet(self):
    """Signaling NaN does not match quiet NaN mask."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN only
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Signaling NaN should not match quiet mask")

  def test_cmp_class_positive_inf(self):
    """V_CMP_CLASS_F32 detects +inf."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),  # large int encodes as literal
      s_mov_b32(s[1], 0b1000000000),  # bit 9 = +inf (512 is outside inline range)
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], s[1]),  # mask in VGPR
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect +inf")

  def test_cmp_class_negative_inf(self):
    """V_CMP_CLASS_F32 detects -inf."""
    neg_inf = 0xff800000
    instructions = [
      s_mov_b32(s[0], neg_inf),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000100),  # bit 2 = -inf
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect -inf")

  def test_cmp_class_normal_positive(self):
    """V_CMP_CLASS_F32 detects positive normal."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),  # inline constant - value to classify
      s_mov_b32(s[1], 0b0100000000),  # bit 8 = positive normal (256 is outside inline range)
      v_mov_b32_e32(v[1], s[1]),  # mask in VGPR
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive normal")

  def test_cmp_class_normal_negative(self):
    """V_CMP_CLASS_F32 detects negative normal."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),  # inline constant - value to classify
      v_mov_b32_e32(v[1], 0b0000001000),  # bit 3 = negative normal
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative normal")


class TestVCmpClassF16(unittest.TestCase):
  """Tests for V_CMP_CLASS_F16 - critical for f16 sin/cos classification.

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

  This is crucial for the f16 sin kernel which uses v_cmp_class_f16 to detect
  special values like +-0, +-inf, NaN and select appropriate outputs.
  """

  def test_cmp_class_f16_positive_zero(self):
    """V_CMP_CLASS_F16: +zero should match bit 6."""
    # f16 +0.0 = 0x0000
    instructions = [
      v_mov_b32_e32(v[0], 0),        # f16 +0.0 in low 16 bits
      v_mov_b32_e32(v[1], 0x40),     # bit 6 only (+zero)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with mask 0x40")

  def test_cmp_class_f16_negative_zero(self):
    """V_CMP_CLASS_F16: -zero should match bit 5."""
    # f16 -0.0 = 0x8000
    instructions = [
      s_mov_b32(s[0], 0x8000),       # f16 -0.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x20),     # bit 5 only (-zero)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for -zero with mask 0x20")

  def test_cmp_class_f16_positive_normal(self):
    """V_CMP_CLASS_F16: +1.0 (normal) should match bit 8."""
    # f16 1.0 = 0x3c00
    instructions = [
      s_mov_b32(s[0], 0x3c00),       # f16 +1.0
      s_mov_b32(s[1], 0x100),        # bit 8 (+normal)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +1.0 with mask 0x100 (+normal)")

  def test_cmp_class_f16_negative_normal(self):
    """V_CMP_CLASS_F16: -1.0 (normal) should match bit 3."""
    # f16 -1.0 = 0xbc00
    instructions = [
      s_mov_b32(s[0], 0xbc00),       # f16 -1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x08),     # bit 3 (-normal)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for -1.0 with mask 0x08 (-normal)")

  def test_cmp_class_f16_positive_infinity(self):
    """V_CMP_CLASS_F16: +inf should match bit 9."""
    # f16 +inf = 0x7c00
    instructions = [
      s_mov_b32(s[0], 0x7c00),       # f16 +inf
      s_mov_b32(s[1], 0x200),        # bit 9 (+inf)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +inf with mask 0x200")

  def test_cmp_class_f16_negative_infinity(self):
    """V_CMP_CLASS_F16: -inf should match bit 2."""
    # f16 -inf = 0xfc00
    instructions = [
      s_mov_b32(s[0], 0xfc00),       # f16 -inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x04),     # bit 2 (-inf)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for -inf with mask 0x04")

  def test_cmp_class_f16_quiet_nan(self):
    """V_CMP_CLASS_F16: quiet NaN should match bit 1."""
    # f16 quiet NaN = 0x7e00 (exponent all 1s, mantissa MSB set)
    instructions = [
      s_mov_b32(s[0], 0x7e00),       # f16 quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x02),     # bit 1 (quiet NaN)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for quiet NaN with mask 0x02")

  def test_cmp_class_f16_signaling_nan(self):
    """V_CMP_CLASS_F16: signaling NaN should match bit 0."""
    # f16 signaling NaN = 0x7c01 (exponent all 1s, mantissa MSB clear, other mantissa bits set)
    instructions = [
      s_mov_b32(s[0], 0x7c01),       # f16 signaling NaN
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x01),     # bit 0 (signaling NaN)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for signaling NaN with mask 0x01")

  def test_cmp_class_f16_positive_denormal(self):
    """V_CMP_CLASS_F16: positive denormal should match bit 7."""
    # f16 smallest positive denormal = 0x0001
    instructions = [
      v_mov_b32_e32(v[0], 1),        # f16 +denormal (0x0001)
      v_mov_b32_e32(v[1], 0x80),     # bit 7 (+denormal)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +denormal with mask 0x80")

  def test_cmp_class_f16_negative_denormal(self):
    """V_CMP_CLASS_F16: negative denormal should match bit 4."""
    # f16 smallest negative denormal = 0x8001
    instructions = [
      s_mov_b32(s[0], 0x8001),       # f16 -denormal
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0x10),     # bit 4 (-denormal)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for -denormal with mask 0x10")

  def test_cmp_class_f16_combined_mask_zeros(self):
    """V_CMP_CLASS_F16: mask 0x60 covers both +zero and -zero."""
    # Test with +0.0
    instructions = [
      v_mov_b32_e32(v[0], 0),        # f16 +0.0
      v_mov_b32_e32(v[1], 0x60),     # bits 5 and 6 (+-zero)
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with mask 0x60")

  def test_cmp_class_f16_combined_mask_1f8(self):
    """V_CMP_CLASS_F16: mask 0x1f8 covers -normal,-denorm,-zero,+zero,+denorm,+normal.

    This is the exact mask used in the f16 sin kernel at PC=46:
      v_cmp_class_f16_e64 vcc_lo, v1, 0x1f8

    The kernel uses this to detect if the input is a "normal" finite value
    (not NaN, not infinity). If the check fails (vcc=0), it selects NaN output.
    """
    # Test with +0.0 - should match via bit 6
    instructions = [
      v_mov_b32_e32(v[0], 0),           # f16 +0.0
      s_mov_b32(s[0], 0x1f8),
      v_mov_b32_e32(v[1], s[0]),        # mask 0x1f8
      v_cmp_class_f16_e32(v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with mask 0x1f8")

  def test_cmp_class_f16_vop3_encoding(self):
    """V_CMP_CLASS_F16 in VOP3 encoding (v_cmp_class_f16_e64).

    This tests the exact instruction encoding used in the f16 sin kernel.
    VOP3 encoding allows the result to go to any SGPR pair, not just VCC.
    """
    # v_cmp_class_f16_e64 vcc_lo, v0, 0x1f8
    # Use SGPR to hold the mask since literals require special handling
    instructions = [
      v_mov_b32_e32(v[0], 0),           # f16 +0.0
      s_mov_b32(s[0], 0x1f8),           # class mask
      VOP3(VOP3Op.V_CMP_CLASS_F16, vdst=RawImm(VCC), src0=v[0], src1=s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +zero with VOP3 encoding")

  def test_cmp_class_f16_vop3_normal_positive(self):
    """V_CMP_CLASS_F16 VOP3 encoding with +1.0 (normal)."""
    # f16 1.0 = 0x3c00, should match bit 8 (+normal) in mask 0x1f8
    instructions = [
      s_mov_b32(s[0], 0x3c00),          # f16 +1.0
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),           # class mask
      VOP3(VOP3Op.V_CMP_CLASS_F16, vdst=RawImm(VCC), src0=v[0], src1=s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for +1.0 (normal) with mask 0x1f8")

  def test_cmp_class_f16_vop3_nan_fails_mask(self):
    """V_CMP_CLASS_F16 VOP3: NaN should NOT match mask 0x1f8 (no NaN bits set)."""
    # f16 quiet NaN = 0x7e00, should NOT match mask 0x1f8 (bits 3-8 only)
    instructions = [
      s_mov_b32(s[0], 0x7e00),          # f16 quiet NaN
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),           # class mask
      VOP3(VOP3Op.V_CMP_CLASS_F16, vdst=RawImm(VCC), src0=v[0], src1=s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "VCC should be 0 for NaN with mask 0x1f8 (no NaN bits)")

  def test_cmp_class_f16_vop3_inf_fails_mask(self):
    """V_CMP_CLASS_F16 VOP3: +inf should NOT match mask 0x1f8 (no inf bits set)."""
    # f16 +inf = 0x7c00, should NOT match mask 0x1f8 (bits 3-8 only)
    instructions = [
      s_mov_b32(s[0], 0x7c00),          # f16 +inf
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x1f8),           # class mask
      VOP3(VOP3Op.V_CMP_CLASS_F16, vdst=RawImm(VCC), src0=v[0], src1=s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "VCC should be 0 for +inf with mask 0x1f8 (no inf bits)")


if __name__ == "__main__":
  unittest.main()
