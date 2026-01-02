#!/usr/bin/env python3
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f


class TestPackInstructions(unittest.TestCase):
  """Tests for pack instructions."""

  def test_v_pack_b32_f16(self):
    """V_PACK_B32_F16 packs two f16 values into one 32-bit register."""
    instructions = []
    # f16 1.0 = 0x3c00, f16 2.0 = 0x4000
    instructions.append(s_mov_b32(s[0], 0x3c00))  # f16 1.0
    instructions.append(s_mov_b32(s[1], 0x4000))  # f16 2.0
    instructions.append(v_mov_b32_e32(v[0], s[0]))
    instructions.append(v_mov_b32_e32(v[1], s[1]))
    # Pack: v[2] = (v[1].f16 << 16) | v[0].f16
    instructions.append(v_pack_b32_f16(v[2], v[0], v[1]))

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4000 (2.0), lo=0x3c00 (1.0) -> 0x40003c00
    self.assertEqual(result, 0x40003c00, f"Expected 0x40003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_with_cvt(self):
    """V_PACK_B32_F16 after V_CVT_F16_F32 conversions."""
    instructions = []
    # f32 1.0 = 0x3f800000
    instructions.append(s_mov_b32(s[0], 0x3f800000))
    instructions.append(v_mov_b32_e32(v[0], s[0]))  # f32 1.0
    instructions.append(v_mov_b32_e32(v[1], s[0]))  # f32 1.0
    # Convert to f16
    instructions.append(v_cvt_f16_f32_e32(v[2], v[0]))  # v[2].f16 = 1.0
    instructions.append(v_cvt_f16_f32_e32(v[3], v[1]))  # v[3].f16 = 1.0
    # Pack
    instructions.append(v_pack_b32_f16(v[4], v[2], v[3]))

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][4]
    # Expected: 0x3c003c00 (two f16 1.0 values)
    self.assertEqual(result, 0x3c003c00, f"Expected 0x3c003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_packed_sources(self):
    """V_PACK_B32_F16 with sources that have packed f16 pairs (both hi and lo used).
    This mimics what happens in matmul kernels where VGPRs contain packed f16 data.
    """
    instructions = []
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # V_PACK_B32_F16 with default opsel=0 reads low halves from each source
    # Result should be: hi=v1.lo=0x4200 (3.0), lo=v0.lo=0x3c00 (1.0) -> 0x42003c00
    instructions.append(s_mov_b32(s[0], 0x40003c00))  # packed: hi=2.0, lo=1.0
    instructions.append(s_mov_b32(s[1], 0x44004200))  # packed: hi=4.0, lo=3.0
    instructions.append(v_mov_b32_e32(v[0], s[0]))
    instructions.append(v_mov_b32_e32(v[1], s[1]))
    instructions.append(v_pack_b32_f16(v[2], v[0], v[1]))

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4200 (3.0), lo=0x3c00 (1.0) -> 0x42003c00
    self.assertEqual(result, 0x42003c00, f"Expected 0x42003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_hi_hi(self):
    """V_PACK_B32_F16 with opsel=0b0011 to read high halves from both sources.
    This is used when extracting the high f16 values from packed registers.
    """
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # With opsel=0b0011: read hi from v0 (0x4000=2.0) and hi from v1 (0x4400=4.0)
    # Result should be: hi=v1.hi=0x4400 (4.0), lo=v0.hi=0x4000 (2.0) -> 0x44004000
    inst = v_pack_b32_f16(v[2], v[0], v[1])
    inst._values['opsel'] = 0b0011  # opsel[0]=1 for src0 hi, opsel[1]=1 for src1 hi

    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # packed: hi=2.0, lo=1.0
      s_mov_b32(s[1], 0x44004200),  # packed: hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      inst,
    ]

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4400 (4.0), lo=0x4000 (2.0) -> 0x44004000
    self.assertEqual(result, 0x44004000, f"Expected 0x44004000, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_lo_hi(self):
    """V_PACK_B32_F16 with opsel=0b0010 to read lo from src0, hi from src1."""
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # With opsel=0b0010: read lo from v0 (0x3c00=1.0), hi from v1 (0x4400=4.0)
    # Result should be: hi=v1.hi=0x4400 (4.0), lo=v0.lo=0x3c00 (1.0) -> 0x44003c00
    inst = v_pack_b32_f16(v[2], v[0], v[1])
    inst._values['opsel'] = 0b0010  # opsel[0]=0 for src0 lo, opsel[1]=1 for src1 hi

    instructions = [
      s_mov_b32(s[0], 0x40003c00),
      s_mov_b32(s[1], 0x44004200),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      inst,
    ]

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4400 (4.0), lo=0x3c00 (1.0) -> 0x44003c00
    self.assertEqual(result, 0x44003c00, f"Expected 0x44003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_hi_lo(self):
    """V_PACK_B32_F16 with opsel=0b0001 to read hi from src0, lo from src1."""
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # With opsel=0b0001: read hi from v0 (0x4000=2.0), lo from v1 (0x4200=3.0)
    # Result should be: hi=v1.lo=0x4200 (3.0), lo=v0.hi=0x4000 (2.0) -> 0x42004000
    inst = v_pack_b32_f16(v[2], v[0], v[1])
    inst._values['opsel'] = 0b0001  # opsel[0]=1 for src0 hi, opsel[1]=0 for src1 lo

    instructions = [
      s_mov_b32(s[0], 0x40003c00),
      s_mov_b32(s[1], 0x44004200),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      inst,
    ]

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4200 (3.0), lo=0x4000 (2.0) -> 0x42004000
    self.assertEqual(result, 0x42004000, f"Expected 0x42004000, got 0x{result:08x}")


class TestWMMA(unittest.TestCase):
  """Tests for WMMA (Wave Matrix Multiply-Accumulate) instructions."""

  def test_v_wmma_f32_16x16x16_f16_basic(self):
    """V_WMMA_F32_16X16X16_F16 basic test - verify emulator matches hardware."""
    # WMMA does D = A @ B + C where A,B are 16x16 f16, C,D are 16x16 f32
    # Use: A=v[16:23], B=v[24:31], C=D=v[0:7] (output in captured range v[0:15])
    instructions = []

    # f16 1.0 = 0x3c00, packed pair = 0x3c003c00
    instructions.append(s_mov_b32(s[0], 0x3c003c00))

    # Set A (v16-v23) and B (v24-v31) to all 1.0s
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))

    # Set C (v0-v7) to all 0s (will also be output D)
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))

    # Execute WMMA: v[0:7] = A @ B + C
    instructions.append(v_wmma_f32_16x16x16_f16(v[0], v[16], v[24], v[0]))

    # Just run and compare - USE_HW=1 will verify emulator matches hardware
    st = run_program(instructions, n_lanes=32)

    # Verify at least some output is non-zero (actual values depend on WMMA layout)
    # Output should be 16.0 (16 x 1.0 x 1.0) for each element
    any_nonzero = any(st.vgpr[lane][0] != 0 for lane in range(32))
    self.assertTrue(any_nonzero, "WMMA should produce non-zero output")

  def test_v_wmma_f32_16x16x16_f16_all_ones(self):
    """V_WMMA_F32_16X16X16_F16 with all ones should produce 16.0 for each output element.
    This verifies the matrix multiply is computing the correct sum.
    """
    instructions = []

    # f16 1.0 = 0x3c00, packed pair = 0x3c003c00
    instructions.append(s_mov_b32(s[0], 0x3c003c00))

    # Set A (v16-v23) and B (v24-v31) to all 1.0s
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))

    # Set C (v0-v7) to all 0s (will also be output D)
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))

    # Execute WMMA: v[0:7] = A @ B + C
    instructions.append(v_wmma_f32_16x16x16_f16(v[0], v[16], v[24], v[0]))

    st = run_program(instructions, n_lanes=32)

    # All output elements should be 16.0 (sum of 16 * 1.0 * 1.0)
    expected = f2i(16.0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 0x{expected:08x} (16.0), got 0x{result:08x} ({i2f(result)})")

  def test_v_wmma_f32_16x16x16_f16_with_accumulator(self):
    """V_WMMA_F32_16X16X16_F16 with non-zero accumulator.
    Verifies that C matrix is properly added to the product.
    """
    instructions = []

    # f16 1.0 = 0x3c00, packed pair = 0x3c003c00
    instructions.append(s_mov_b32(s[0], 0x3c003c00))
    # f32 5.0 = 0x40a00000
    instructions.append(s_mov_b32(s[1], f2i(5.0)))

    # Set A (v16-v23) and B (v24-v31) to all 1.0s
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))

    # Set C (v0-v7) to all 5.0s
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], s[1]))

    # Execute WMMA: v[0:7] = A @ B + C = 16.0 + 5.0 = 21.0
    instructions.append(v_wmma_f32_16x16x16_f16(v[0], v[16], v[24], v[0]))

    st = run_program(instructions, n_lanes=32)

    # All output elements should be 21.0 (16.0 + 5.0)
    expected = f2i(21.0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 0x{expected:08x} (21.0), got 0x{result:08x} ({i2f(result)})")


class TestVOP3P(unittest.TestCase):
  """Tests for VOP3P packed 16-bit operations."""

  def test_v_pk_add_f16_basic(self):
    """V_PK_ADD_F16 adds two packed f16 values."""
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (1.0, 2.0), v1 = packed (3.0, 4.0)
    # Result should be packed (4.0, 6.0)
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # packed f16: hi=2.0, lo=1.0
      s_mov_b32(s[1], 0x44004200),  # packed f16: hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_add_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: lo=1.0+3.0=4.0 (0x4400), hi=2.0+4.0=6.0 (0x4600) -> 0x46004400
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 4.0, places=2, msg=f"lo: expected 4.0, got {lo}")
    self.assertAlmostEqual(hi, 6.0, places=2, msg=f"hi: expected 6.0, got {hi}")

  def test_v_pk_add_f16_with_inline_constant(self):
    """V_PK_ADD_F16 with inline constant POS_ONE (1.0).
    Inline constants for VOP3P are f16 values in the low 16 bits only.
    The opsel_hi bits (default=0b11) select lo half for hi result, so both halves use the constant.
    """
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (1.0, 1.0), add POS_ONE
    # With default opsel_hi=0b11: both lo and hi results use lo half of src1 (the constant)
    # But opsel_hi=1 means src1 hi comes from lo half - wait, let me check the actual encoding
    # Default opsel_hi=3 means: bit0=1 (src0 hi from hi), bit1=1 (src1 hi from hi)
    # Since inline constant has 0 in hi half, hi result = v0.hi + 0 = 1.0
    instructions = [
      s_mov_b32(s[0], 0x3c003c00),  # packed f16: hi=1.0, lo=1.0
      v_mov_b32_e32(v[0], s[0]),
      v_pk_add_f16(v[1], v[0], SrcEnum.POS_ONE),  # Add inline constant 1.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    # lo = 1.0 + 1.0 = 2.0, hi = 1.0 + 0.0 = 1.0 (inline const hi half is 0)
    self.assertAlmostEqual(lo, 2.0, places=2, msg=f"lo: expected 2.0, got {lo} (result=0x{result:08x})")
    self.assertAlmostEqual(hi, 1.0, places=2, msg=f"hi: expected 1.0, got {hi} (result=0x{result:08x})")

  def test_v_pk_mul_f16_basic(self):
    """V_PK_MUL_F16 multiplies two packed f16 values."""
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (2.0, 3.0), v1 = packed (4.0, 5.0)
    # Result should be packed (8.0, 15.0)
    instructions = [
      s_mov_b32(s[0], 0x42004000),  # packed f16: hi=3.0, lo=2.0
      s_mov_b32(s[1], 0x45004400),  # packed f16: hi=5.0, lo=4.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_mul_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 8.0, places=1, msg=f"lo: expected 8.0, got {lo}")
    self.assertAlmostEqual(hi, 15.0, places=1, msg=f"hi: expected 15.0, got {hi}")

  def test_v_pk_mul_f16_with_inline_constant(self):
    """V_PK_MUL_F16 with inline constant POS_TWO (2.0).
    Inline constant has value only in low 16 bits, hi is 0.
    """
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (3.0, 4.0), multiply by POS_TWO
    # lo = 3.0 * 2.0 = 6.0, hi = 4.0 * 0.0 = 0.0 (inline const hi is 0)
    instructions = [
      s_mov_b32(s[0], 0x44004200),  # packed f16: hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_pk_mul_f16(v[1], v[0], SrcEnum.POS_TWO),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 6.0, places=1, msg=f"lo: expected 6.0, got {lo}")
    self.assertAlmostEqual(hi, 0.0, places=1, msg=f"hi: expected 0.0, got {hi}")

  def test_v_pk_fma_f16_basic(self):
    """V_PK_FMA_F16: D = A * B + C for packed f16."""
    from extra.assembly.amd.pcode import _f16
    # A = packed (2.0, 3.0), B = packed (4.0, 5.0), C = packed (1.0, 1.0)
    # Result should be packed (2*4+1=9.0, 3*5+1=16.0)
    instructions = [
      s_mov_b32(s[0], 0x42004000),  # A: hi=3.0, lo=2.0
      s_mov_b32(s[1], 0x45004400),  # B: hi=5.0, lo=4.0
      s_mov_b32(s[2], 0x3c003c00),  # C: hi=1.0, lo=1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_pk_fma_f16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 9.0, places=1, msg=f"lo: expected 9.0, got {lo}")
    self.assertAlmostEqual(hi, 16.0, places=0, msg=f"hi: expected 16.0, got {hi}")


if __name__ == "__main__":
  unittest.main()
