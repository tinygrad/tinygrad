"""Tests for VOP2 instructions - two operand vector operations.

Includes: v_add_f32, v_mul_f32, v_and_b32, v_or_b32, v_xor_b32,
          v_lshrrev_b32, v_lshlrev_b32, v_fmac_f32, v_fmaak_f32, v_fmamk_f32,
          v_add_nc_u32, v_cndmask_b32, v_add_f16, v_mul_f16
"""
import unittest
from extra.assembly.amd.test.hw.helpers import *

class TestBasicArithmetic(unittest.TestCase):
  """Tests for basic arithmetic VOP2 instructions."""

  def test_v_add_f32(self):
    """V_ADD_F32 adds two floats."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, places=5)

  def test_v_mul_f32(self):
    """V_MUL_F32 multiplies two floats."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 8.0, places=5)

  def test_v_fmac_f32(self):
    """V_FMAC_F32: d = d + a*b using inline constants."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_mov_b32_e32(v[2], 1.0),
      v_fmac_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmaak_f32(self):
    """V_FMAAK_F32: d = a * b + K using inline constants."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 4.0),
      v_fmaak_f32_e32(v[2], v[0], v[1], literal=0x3f800000),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmamk_f32_basic(self):
    """V_FMAMK_F32: d = a * K + b."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_mov_b32_e32(v[1], 1.0),
      v_fmamk_f32_e32(v[2], v[0], v[1], literal=0x40800000),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmamk_f32_small_constant(self):
    """V_FMAMK_F32 with small constant."""
    instructions = [
      v_mov_b32_e32(v[0], 4.0),
      v_mov_b32_e32(v[1], 1.0),
      v_fmamk_f32_e32(v[2], v[0], v[1], literal=f2i(0.5)),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, places=5)


class TestBitManipulation(unittest.TestCase):
  """Tests for bit manipulation VOP2 instructions."""

  def test_v_and_b32(self):
    """V_AND_B32 bitwise and."""
    instructions = [
      s_mov_b32(s[0], 0xff),
      s_mov_b32(s[1], 0x0f),
      v_mov_b32_e32(v[0], s[0]),
      v_and_b32_e32(v[1], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x0f)

  def test_v_and_b32_quadrant(self):
    """V_AND_B32 for quadrant extraction (n & 3)."""
    instructions = [
      s_mov_b32(s[0], 15915),
      v_mov_b32_e32(v[0], s[0]),
      v_and_b32_e32(v[1], 3, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15915 & 3)

  def test_v_lshrrev_b32(self):
    """V_LSHRREV_B32 logical shift right."""
    instructions = [
      s_mov_b32(s[0], 0xff00),
      v_mov_b32_e32(v[0], s[0]),
      v_lshrrev_b32_e32(v[1], 8, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff)

  def test_v_lshlrev_b32(self):
    """V_LSHLREV_B32 logical shift left."""
    instructions = [
      s_mov_b32(s[0], 0xff),
      v_mov_b32_e32(v[0], s[0]),
      v_lshlrev_b32_e32(v[1], 8, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff00)

  def test_v_xor_b32(self):
    """V_XOR_B32 bitwise xor (used in sin for sign)."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      s_mov_b32(s[1], f2i(1.0)),
      v_mov_b32_e32(v[0], s[1]),
      v_xor_b32_e32(v[1], s[0], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -1.0, places=5)

  def test_v_xor_b32_sign_flip(self):
    """V_XOR_B32 for sign flip pattern."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], -2.0),
      v_xor_b32_e32(v[1], s[0], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)


class TestSpecialValues(unittest.TestCase):
  """Tests for special float values - inf, nan, zero handling."""

  def test_v_mul_f32_zero_times_inf(self):
    """V_MUL_F32: 0 * inf = NaN."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0x7f800000),
      v_mov_b32_e32(v[1], s[0]),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_add_f32_inf_minus_inf(self):
    """V_ADD_F32: inf + (-inf) = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),
      s_mov_b32(s[1], 0xff800000),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))


class TestF16Ops(unittest.TestCase):
  """Tests for 16-bit VOP2 operations."""

  def test_v_add_f16_basic(self):
    """V_ADD_F16 adds two f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4200, f"Expected 0x4200 (f16 3.0), got 0x{result:04x}")

  def test_v_add_f16_negative(self):
    """V_ADD_F16 with negative values."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0xc000),  # f16 -2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0xbc00, f"Expected 0xbc00 (f16 -1.0), got 0x{result:04x}")

  def test_v_mul_f16_basic(self):
    """V_MUL_F16 multiplies two f16 values."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      s_mov_b32(s[1], 0x4200),  # f16 3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mul_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4600, f"Expected 0x4600 (f16 6.0), got 0x{result:04x}")

  def test_v_mul_f16_by_zero(self):
    """V_MUL_F16 by zero."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], 0),
      v_mul_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x0000, f"Expected 0x0000 (f16 0.0), got 0x{result:04x}")

  def test_v_fmac_f16_basic(self):
    """V_FMAC_F16: d = d + a*b."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      s_mov_b32(s[1], 0x4200),  # f16 3.0
      s_mov_b32(s[2], 0x3c00),  # f16 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_fmac_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    # 2.0 * 3.0 + 1.0 = 7.0, f16 7.0 = 0x4700
    self.assertEqual(result, 0x4700, f"Expected 0x4700 (f16 7.0), got 0x{result:04x}")

  def test_v_fmaak_f16_basic(self):
    """V_FMAAK_F16: d = a * b + K."""
    instructions = [
      s_mov_b32(s[0], 0x4000),  # f16 2.0
      s_mov_b32(s[1], 0x4200),  # f16 3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_fmaak_f16_e32(v[2], v[0], v[1], literal=0x3c00),  # + f16 1.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    # 2.0 * 3.0 + 1.0 = 7.0, f16 7.0 = 0x4700
    self.assertEqual(result, 0x4700, f"Expected 0x4700 (f16 7.0), got 0x{result:04x}")


class TestHiHalfOps(unittest.TestCase):
  """Tests for VOP2 16-bit operations with hi-half operands."""

  def test_v_add_f16_src0_hi_fold(self):
    """V_ADD_F16 with src0 hi-half fold (same register, different halves)."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # lo=f16(1.0), hi=f16(2.0)
      v_mov_b32_e32(v[0], s[0]),
      VOP3(VOP3Op.V_ADD_F16, vdst=v[1], src0=v[0], src1=v[0], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0x4200, f"Expected f16(3.0)=0x4200, got 0x{result:04x}")

  def test_v_add_f16_src0_hi_different_reg(self):
    """V_ADD_F16 with src0 hi-half from different register."""
    instructions = [
      s_mov_b32(s[0], 0x40000000),  # hi=f16(2.0), lo=0
      s_mov_b32(s[1], 0x00003c00),  # hi=0, lo=f16(1.0)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      VOP3(VOP3Op.V_ADD_F16, vdst=v[2], src0=v[0], src1=v[1], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4200, f"Expected f16(3.0)=0x4200, got 0x{result:04x}")

  def test_v_mul_f16_src0_hi(self):
    """V_MUL_F16 with src0 from high half."""
    instructions = [
      s_mov_b32(s[0], 0x40000000),  # hi=f16(2.0), lo=0
      s_mov_b32(s[1], 0x00004200),  # hi=0, lo=f16(3.0)
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      VOP3(VOP3Op.V_MUL_F16, vdst=v[2], src0=v[0], src1=v[1], opsel=0b0001),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4600, f"Expected f16(6.0)=0x4600, got 0x{result:04x}")

  def test_v_mul_f16_hi_half(self):
    """V_MUL_F16 reading from high half."""
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # lo=1.0, hi=2.0
      v_mov_b32_e32(v[0], s[0]),
      VOP3(VOP3Op.V_MUL_F16, vdst=v[1], src0=v[0], src1=v[0], opsel=0b0011),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1] & 0xffff
    self.assertEqual(result, 0x4400, f"Expected f16(4.0)=0x4400, got 0x{result:04x}")

  def test_v_fma_f16_hi_dest(self):
    """V_FMA_F16 writing to high half with opsel.

    Uses V_FMA_F16 (not V_FMAC_F16) because it has explicit src2 operand
    which makes opsel handling clearer.
    """
    instructions = [
      s_mov_b32(s[0], 0x3c000000),  # hi=f16(1.0), lo=0
      s_mov_b32(s[1], 0x4000),      # f16(2.0) in lo
      s_mov_b32(s[2], 0x4200),      # f16(3.0) in lo
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      # V_FMA_F16: dst = src0 * src1 + src2
      # opsel=0b1100: bit2=src2 hi, bit3=dst hi
      # So: v[0].hi = v[1].lo * v[2].lo + v[0].hi = 2.0 * 3.0 + 1.0 = 7.0
      VOP3(VOP3Op.V_FMA_F16, vdst=v[0], src0=v[1], src1=v[2], src2=v[0], opsel=0b1100),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][0] >> 16) & 0xffff
    # 2.0 * 3.0 + 1.0 = 7.0, f16 7.0 = 0x4700
    self.assertEqual(hi, 0x4700, f"Expected f16(7.0)=0x4700 in hi, got 0x{hi:04x}")

  def test_v_add_f16_multilane(self):
    """V_ADD_F16 with multiple lanes."""
    instructions = [
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f16_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      result = st.vgpr[lane][2] & 0xffff
      self.assertEqual(result, 0x4200, f"Lane {lane}: expected 0x4200, got 0x{result:04x}")


class TestCndmask(unittest.TestCase):
  """Tests for V_CNDMASK_B32 and V_CNDMASK_B16."""

  def test_v_cndmask_b16_select_src0(self):
    """V_CNDMASK_B16 selects src0 when VCC bit is 0."""
    instructions = [
      s_mov_b32(VCC_LO, 0),  # VCC = 0
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cndmask_b16(v[2], v[0], v[1], VCC),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x3c00, f"Expected src0=0x3c00, got 0x{result:04x}")

  def test_v_cndmask_b16_select_src1(self):
    """V_CNDMASK_B16 selects src1 when VCC bit is 1."""
    instructions = [
      s_mov_b32(VCC_LO, 1),  # VCC = 1
      s_mov_b32(s[0], 0x3c00),  # f16 1.0
      s_mov_b32(s[1], 0x4000),  # f16 2.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cndmask_b16(v[2], v[0], v[1], VCC),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2] & 0xffff
    self.assertEqual(result, 0x4000, f"Expected src1=0x4000, got 0x{result:04x}")

  def test_v_cndmask_b16_write_hi(self):
    """V_CNDMASK_B16 can write to high 16 bits with opsel."""
    instructions = [
      s_mov_b32(s[0], 0x3c003800),  # src0: hi=1.0, lo=0.5
      v_mov_b32_e32(v[0], s[0]),
      s_mov_b32(s[1], 0x4000c000),  # src1: hi=2.0, lo=-2.0
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], 0xDEAD0000),  # v2 initial: hi=0xDEAD, lo=0
      v_mov_b32_e32(v[2], s[2]),
      s_mov_b32(VCC_LO, 0),  # vcc = 0, select src0
      # opsel=0b1011: bit0=src0 hi, bit1=src1 hi, bit3=dst hi
      VOP3(VOP3Op.V_CNDMASK_B16, vdst=v[2], src0=v[0], src1=v[1], src2=SrcEnum.VCC_LO, opsel=0b1011),
    ]
    st = run_program(instructions, n_lanes=1)
    hi = (st.vgpr[0][2] >> 16) & 0xffff
    lo = st.vgpr[0][2] & 0xffff
    # vcc=0 selects src0.h = 1.0 = 0x3c00, writes to hi
    self.assertEqual(hi, 0x3c00, f"Expected hi=0x3c00 (1.0), got 0x{hi:04x}")
    self.assertEqual(lo, 0x0000, f"Expected lo preserved as 0, got 0x{lo:04x}")


class TestSpecialFloatValues(unittest.TestCase):
  """Tests for special float value handling in VOP2 instructions."""

  def test_neg_zero_add(self):
    """-0.0 + 0.0 = +0.0 (IEEE 754)."""
    neg_zero = 0x80000000
    instructions = [
      s_mov_b32(s[0], neg_zero),
      v_mov_b32_e32(v[0], s[0]),
      v_add_f32_e32(v[1], 0.0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x00000000, "Should be +0.0")

  def test_neg_zero_mul(self):
    """-0.0 * -1.0 = +0.0."""
    neg_zero = 0x80000000
    instructions = [
      s_mov_b32(s[0], neg_zero),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], -1.0, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x00000000, "Should be +0.0")

  def test_inf_minus_inf(self):
    """+inf - inf = NaN."""
    import math
    pos_inf = 0x7f800000
    neg_inf = 0xff800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      s_mov_b32(s[1], neg_inf),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_sub_f32_e32(v[2], v[0], v[1]),  # inf - (-inf) = inf
      v_add_f32_e32(v[3], v[0], v[1]),  # inf + (-inf) = NaN
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], pos_inf, "inf - (-inf) = inf")
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])), "inf + (-inf) = NaN")

  def test_denormal_f32_mul_ftz(self):
    """Denormal * normal - RDNA3 flushes denormals to zero (FTZ mode)."""
    smallest_denorm = 0x00000001  # Smallest positive denormal
    instructions = [
      s_mov_b32(s[0], smallest_denorm),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], 2.0, v[0]),  # Denormal input gets flushed to 0
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x00000000)


if __name__ == '__main__':
  unittest.main()
