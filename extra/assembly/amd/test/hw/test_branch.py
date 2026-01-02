#!/usr/bin/env python3
"""Tests for branch instructions, 64-bit literals, and VOP3 VOPC 16-bit."""
import struct, unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import RawImm
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f, f2i64, i642f


class Test64BitLiterals(unittest.TestCase):
  """Regression tests for 64-bit instruction literal encoding.
  Tests verify that Inst.to_bytes() correctly encodes 64-bit literals."""

  def test_64bit_literal_negative_encoding(self):
    """Verify 64-bit instruction encodes negative literals correctly.
    Regression test: -33 should encode as 0xffffffdf in the literal field,
    NOT as 0xffffffff (which would happen with incorrect sign extension)."""
    neg_val = -33
    expected_lit = neg_val & 0xffffffff  # 0xffffffdf
    inst = v_add_f64(v[2], v[0], neg_val)
    # Check the literal is stored correctly (in high 32 bits for 64-bit ops)
    self.assertIsNotNone(inst._literal, "Literal should be set")
    # Literal is stored as (lit32 << 32) for 64-bit ops
    actual_lit = (inst._literal >> 32) & 0xffffffff
    self.assertEqual(actual_lit, expected_lit, f"Literal should be {expected_lit:#x}, got {actual_lit:#x}")
    # Also verify the encoded bytes
    code = inst.to_bytes()
    # Literal is last 4 bytes
    lit_bytes = code[-4:]
    lit_val = int.from_bytes(lit_bytes, 'little')
    self.assertEqual(lit_val, expected_lit, f"Encoded literal should be {expected_lit:#x}, got {lit_val:#x}")

  def test_64bit_literal_positive_encoding(self):
    """Verify 64-bit instruction encodes large positive literals correctly."""
    large_val = 0x12345678
    inst = v_add_f64(v[2], v[0], large_val)
    self.assertIsNotNone(inst._literal, "Literal should be set")
    actual_lit = (inst._literal >> 32) & 0xffffffff
    self.assertEqual(actual_lit, large_val, f"Literal should be {large_val:#x}, got {actual_lit:#x}")
    # Verify encoded bytes
    code = inst.to_bytes()
    lit_bytes = code[-4:]
    lit_val = int.from_bytes(lit_bytes, 'little')
    self.assertEqual(lit_val, large_val, f"Encoded literal should be {large_val:#x}, got {lit_val:#x}")


class TestWave32VCCBranch(unittest.TestCase):
  """Regression tests for wave32 VCC branch behavior.
  In wave32 mode, S_CBRANCH_VCCNZ/VCCZ should only check VCC_LO (lower 32 bits),
  ignoring VCC_HI. Bug: emulator was checking full 64-bit VCC, causing incorrect
  branches when VCC_LO=0 but VCC_HI!=0."""

  def test_cbranch_vccnz_ignores_vcc_hi(self):
    """S_CBRANCH_VCCNZ should NOT branch when VCC_LO=0, even if VCC_HI!=0."""
    instructions = [
      # Set VCC_HI to non-zero (simulating stale bits from previous ops)
      s_mov_b32(s[SrcEnum.VCC_HI - 128], 0x80000000),  # VCC_HI = 0x80000000
      # Set VCC_LO to zero (the condition we're testing)
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 0),  # VCC_LO = 0
      # Now S_CBRANCH_VCCNZ should NOT branch since VCC_LO is 0
      v_mov_b32_e32(v[0], 0),
      s_cbranch_vccnz(2),  # Skip next instruction if VCC != 0
      v_mov_b32_e32(v[0], 1),  # This should execute
      s_nop(0),  # Jump target
    ]
    st = run_program(instructions, n_lanes=1)
    # v0 should be 1 because VCC_LO=0 means no branch
    self.assertEqual(st.vgpr[0][0], 1, "Should NOT branch when VCC_LO=0 (VCC_HI ignored in wave32)")

  def test_cbranch_vccz_ignores_vcc_hi(self):
    """S_CBRANCH_VCCZ should branch when VCC_LO=0, regardless of VCC_HI."""
    instructions = [
      # Set VCC_HI to non-zero (simulating stale bits)
      s_mov_b32(s[SrcEnum.VCC_HI - 128], 0x80000000),  # VCC_HI = 0x80000000
      # Set VCC_LO to zero
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 0),  # VCC_LO = 0
      # S_CBRANCH_VCCZ should branch since VCC_LO is 0
      v_mov_b32_e32(v[0], 0),
      s_cbranch_vccz(2),  # Skip next instruction if VCC == 0
      v_mov_b32_e32(v[0], 1),  # This should NOT execute
      s_nop(0),  # Jump target
    ]
    st = run_program(instructions, n_lanes=1)
    # v0 should be 0 because VCC_LO=0 means branch is taken
    self.assertEqual(st.vgpr[0][0], 0, "Should branch when VCC_LO=0 (VCC_HI ignored in wave32)")

  def test_cbranch_vccnz_branches_on_vcc_lo(self):
    """S_CBRANCH_VCCNZ should branch when VCC_LO!=0."""
    instructions = [
      # Set VCC_LO to non-zero
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 1),  # VCC_LO = 1
      s_mov_b32(s[SrcEnum.VCC_HI - 128], 0),  # VCC_HI = 0
      v_mov_b32_e32(v[0], 0),
      s_cbranch_vccnz(2),  # Skip next instruction if VCC != 0
      v_mov_b32_e32(v[0], 1),  # This should NOT execute
      s_nop(0),  # Jump target
    ]
    st = run_program(instructions, n_lanes=1)
    # v0 should be 0 because VCC_LO=1 means branch is taken
    self.assertEqual(st.vgpr[0][0], 0, "Should branch when VCC_LO!=0")


class TestVOP3VOPC16Bit(unittest.TestCase):
  """Regression tests for VOP3-encoded VOPC 16-bit comparison instructions.
  When VOPC comparisons are encoded in VOP3 format, they use opsel bits to select
  which 16-bit half of each source to compare.
  Bug: Emulator was ignoring opsel and using VGPR bit 7 encoding instead."""

  def test_cmp_eq_u16_opsel_lo_lo(self):
    """V_CMP_EQ_U16 VOP3 with opsel=0 compares lo halves."""
    # v0 = 0x12340005 (lo=5, hi=0x1234)
    # v1 = 0x56780005 (lo=5, hi=0x5678)
    # opsel=0: compare lo halves -> 5 == 5 -> true
    instructions = [
      s_mov_b32(s[2], 0x12340005),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x56780005),
      v_mov_b32_e32(v[1], s[2]),
      VOP3(VOP3Op.V_CMP_EQ_U16, vdst=v[0], src0=v[0], src1=v[1], opsel=0),  # dst=s0
    ]
    st = run_program(instructions, n_lanes=1)
    # s0 should have bit 0 set (comparison true for lane 0)
    self.assertEqual(st.sgpr[0] & 1, 1, "lo==lo should be true: 5==5")

  def test_cmp_eq_u16_opsel_hi_hi(self):
    """V_CMP_EQ_U16 VOP3 with opsel=3 compares hi halves."""
    # v0 = 0x12340005 (lo=5, hi=0x1234)
    # v1 = 0x56780005 (lo=5, hi=0x5678)
    # opsel=3 (bits 0 and 1 set): compare hi halves -> 0x1234 != 0x5678 -> false
    instructions = [
      s_mov_b32(s[2], 0x12340005),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x56780005),
      v_mov_b32_e32(v[1], s[2]),
      VOP3(VOP3Op.V_CMP_EQ_U16, vdst=v[0], src0=v[0], src1=v[1], opsel=3),  # dst=s0, hi vs hi
    ]
    st = run_program(instructions, n_lanes=1)
    # s0 should have bit 0 clear (comparison false for lane 0)
    self.assertEqual(st.sgpr[0] & 1, 0, "hi==hi should be false: 0x1234!=0x5678")

  def test_cmp_eq_u16_opsel_hi_hi_equal(self):
    """V_CMP_EQ_U16 VOP3 with opsel=3 compares hi halves (equal case)."""
    # v0 = 0x12340005 (lo=5, hi=0x1234)
    # v1 = 0x12340009 (lo=9, hi=0x1234)
    # opsel=3: compare hi halves -> 0x1234 == 0x1234 -> true
    instructions = [
      s_mov_b32(s[2], 0x12340005),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x12340009),
      v_mov_b32_e32(v[1], s[2]),
      VOP3(VOP3Op.V_CMP_EQ_U16, vdst=v[0], src0=v[0], src1=v[1], opsel=3),  # dst=s0, hi vs hi
    ]
    st = run_program(instructions, n_lanes=1)
    # s0 should have bit 0 set (comparison true for lane 0)
    self.assertEqual(st.sgpr[0] & 1, 1, "hi==hi should be true: 0x1234==0x1234")


class Test64BitLiteralSources(unittest.TestCase):
  """Regression tests for 64-bit instruction literal source handling.

  For f64 operations, a 32-bit literal in the instruction stream represents the
  HIGH 32 bits of the 64-bit value (low 32 bits are implicitly 0).
  """

  def test_v_fma_f64_literal_neg_2pow32(self):
    """V_FMA_F64 with literal encoding of -2^32."""
    # v[0:1] = -41.0 (trunc), v[2:3] = -1.0 (floor of -41/2^32)
    # FMA: result = (-2^32) * (-1.0) + (-41.0) = 4294967296 - 41 = 4294967255.0
    val_41 = f2i64(-41.0)
    val_m1 = f2i64(-1.0)
    lit = 0xC1F00000  # high 32 bits of f64 -2^32
    instructions = [
      s_mov_b32(s[0], val_41 & 0xffffffff),
      s_mov_b32(s[1], (val_41 >> 32) & 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      s_mov_b32(s[2], val_m1 & 0xffffffff),
      s_mov_b32(s[3], (val_m1 >> 32) & 0xffffffff),
      v_mov_b32_e32(v[2], s[2]),
      v_mov_b32_e32(v[3], s[3]),
      # V_FMA_F64 v[4:5], literal, v[2:3], v[0:1]
      VOP3(VOP3Op.V_FMA_F64, vdst=v[4], src0=RawImm(255), src1=v[2], src2=v[0], literal=lit),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    expected = 4294967255.0  # 2^32 - 41
    self.assertAlmostEqual(result, expected, places=0, msg=f"Expected {expected}, got {result}")

  def test_f64_to_i64_full_sequence(self):
    """Full f64->i64 conversion sequence with negative value."""
    val = f2i64(-41.0)
    lit = 0xC1F00000  # high 32 bits of f64 -2^32
    instructions = [
      s_mov_b32(s[0], val & 0xffffffff),
      s_mov_b32(s[1], (val >> 32) & 0xffffffff),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_trunc_f64_e32(v[0:2], v[0:2]),
      v_ldexp_f64(v[2:4], v[0:2], 0xFFFFFFE0),  # -32
      v_floor_f64_e32(v[2:4], v[2:4]),
      VOP3(VOP3Op.V_FMA_F64, vdst=v[0], src0=RawImm(255), src1=v[2], src2=v[0], literal=lit),
      v_cvt_u32_f64_e32(v[4], v[0:2]),
      v_cvt_i32_f64_e32(v[5], v[2:4]),
    ]
    st = run_program(instructions, n_lanes=1)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, -41, f"Expected -41, got {result} (lo=0x{lo:08x}, hi=0x{hi:08x})")


if __name__ == "__main__":
  unittest.main()
