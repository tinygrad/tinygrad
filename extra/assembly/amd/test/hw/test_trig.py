#!/usr/bin/env python3
import math, unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program, f2i, i2f, f2i64, i642f


class TestTrigonometry(unittest.TestCase):
  """Tests for trigonometric instructions."""

  def test_v_sin_f32_small(self):
    """V_SIN_F32 computes sin for small values."""
    # sin(1.0) ≈ 0.8414709848
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = math.sin(1.0 * 2 * math.pi)  # V_SIN_F32 expects input in cycles (0-1 = 0-2π)
    self.assertAlmostEqual(result, expected, places=4)

  def test_v_sin_f32_quarter(self):
    """V_SIN_F32 at 0.25 cycles = sin(π/2) = 1.0."""
    instructions = [
      s_mov_b32(s[0], f2i(0.25)),  # 0.25 is not an inline constant, use f2i
      v_mov_b32_e32(v[0], s[0]),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 1.0, places=4)

  def test_v_sin_f32_large(self):
    """V_SIN_F32 for large input value (132000.0)."""
    # This is the failing case: sin(132000.0) should be ≈ 0.294
    # V_SIN_F32 input is in cycles, so we need frac(132000.0) * 2π
    instructions = [
      s_mov_b32(s[0], f2i(132000.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    # frac(132000.0) = 0, so sin(0) = 0... but actually V_SIN_F32 does its own frac internally
    # The expected value is sin(frac(132000.0) * 2π) where frac is done in the instruction
    # For 132000.0, the hardware computes frac(132000.0) ≈ 0.046875 (due to precision)
    # sin(0.046875 * 2π) ≈ 0.294
    expected = math.sin(132000.0 * 2 * math.pi)
    # Allow some tolerance due to precision differences
    self.assertAlmostEqual(result, expected, places=2, msg=f"sin(132000) got {result}, expected ~{expected}")


class TestOCMLSinSequence(unittest.TestCase):
  """Test the specific instruction sequence used in OCML sin."""

  def test_sin_reduction_step1_mul(self):
    """First step: v12 = |x| * (1/2pi)."""
    one_over_2pi = 1.0 / (2.0 * math.pi)  # 0x3e22f983 in hex
    x = 100000.0
    instructions = [
      s_mov_b32(s[0], f2i(x)),
      s_mov_b32(s[1], f2i(one_over_2pi)),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = x * one_over_2pi
    self.assertAlmostEqual(result, expected, places=0)

  def test_sin_reduction_step2_round(self):
    """Second step: round to nearest integer."""
    one_over_2pi = 1.0 / (2.0 * math.pi)
    x = 100000.0
    val = x * one_over_2pi  # ~15915.49
    instructions = [
      s_mov_b32(s[0], f2i(val)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = round(val)
    self.assertAlmostEqual(result, expected, places=0)

  def test_sin_reduction_step3_fma(self):
    """Third step: x - n * (pi/2) via FMA."""
    # This is where precision matters - the FMA does: |x| + (-pi/2) * n
    neg_half_pi = -math.pi / 2.0  # 0xbfc90fda
    x = 100000.0
    n = 15915.0
    instructions = [
      s_mov_b32(s[0], f2i(neg_half_pi)),
      s_mov_b32(s[1], f2i(n)),
      s_mov_b32(s[2], f2i(x)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_fma_f32(v[3], v[0], v[1], v[2]),  # x + (-pi/2) * n
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    expected = x + neg_half_pi * n
    # Allow some tolerance due to float precision
    self.assertAlmostEqual(result, expected, places=2)

  def test_sin_1e5_full_reduction(self):
    """Full reduction sequence for sin(1e5)."""
    x = 100000.0
    one_over_2pi = 1.0 / (2.0 * math.pi)
    neg_half_pi = -math.pi / 2.0

    instructions = [
      # Load constants
      s_mov_b32(s[0], f2i(x)),
      s_mov_b32(s[1], f2i(one_over_2pi)),
      s_mov_b32(s[2], f2i(neg_half_pi)),
      # Step 1: v1 = x * (1/2pi)
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], s[1], v[0]),
      # Step 2: v2 = round(v1)
      v_rndne_f32_e32(v[2], v[1]),
      # Step 3: v3 = x + (-pi/2) * round_val (FMA)
      v_fma_f32(v[3], s[2], v[2], v[0]),
      # Step 4: convert to int for quadrant
      v_cvt_i32_f32_e32(v[4], v[2]),
      # Step 5: quadrant = n & 3
      v_and_b32_e32(v[5], 3, v[4]),
    ]
    st = run_program(instructions, n_lanes=1)

    # Check intermediate values
    mul_result = i2f(st.vgpr[0][1])
    round_result = i2f(st.vgpr[0][2])
    reduced = i2f(st.vgpr[0][3])
    quadrant = st.vgpr[0][5]

    # Verify results match expected
    expected_mul = x * one_over_2pi
    expected_round = round(expected_mul)
    expected_reduced = x + neg_half_pi * expected_round
    expected_quadrant = int(expected_round) & 3

    self.assertAlmostEqual(mul_result, expected_mul, places=0, msg=f"mul: got {mul_result}, expected {expected_mul}")
    self.assertAlmostEqual(round_result, expected_round, places=0, msg=f"round: got {round_result}, expected {expected_round}")
    self.assertEqual(quadrant, expected_quadrant, f"quadrant: got {quadrant}, expected {expected_quadrant}")


class TestVTrigPreopF64(unittest.TestCase):
  """Tests for V_TRIG_PREOP_F64 instruction.

  V_TRIG_PREOP_F64 extracts chunks of 2/PI for Payne-Hanek trig range reduction.
  For input S0 (f64) and index S1 (0, 1, or 2), it returns a portion of 2/PI
  scaled appropriately for computing |S0| * (2/PI) in extended precision.

  The three chunks (index 0, 1, 2) when summed should equal 2/PI.
  """

  def test_trig_preop_f64_index0(self):
    """V_TRIG_PREOP_F64 index=0: primary chunk of 2/PI."""
    two_over_pi = 2.0 / math.pi
    instructions = [
      # S0 = 1.0 (f64), S1 = 0 (index)
      s_mov_b32(s[0], 0x00000000),  # low bits of 1.0
      s_mov_b32(s[1], 0x3ff00000),  # high bits of 1.0
      v_trig_preop_f64(v[0], abs(s[0]), 0),  # index 0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    # For x=1.0, index=0 should give the main part of 2/PI
    self.assertAlmostEqual(result, two_over_pi, places=10, msg=f"Expected ~{two_over_pi}, got {result}")

  def test_trig_preop_f64_index1(self):
    """V_TRIG_PREOP_F64 index=1: secondary chunk (extended precision bits)."""
    instructions = [
      s_mov_b32(s[0], 0x00000000),  # low bits of 1.0
      s_mov_b32(s[1], 0x3ff00000),  # high bits of 1.0
      v_trig_preop_f64(v[0], abs(s[0]), 1),  # index 1
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    # Index 1 gives the next 53 bits, should be very small (~1e-16)
    self.assertLess(abs(result), 1e-15, msg=f"Expected tiny value, got {result}")
    self.assertGreater(abs(result), 0, msg="Expected non-zero value")

  def test_trig_preop_f64_index2(self):
    """V_TRIG_PREOP_F64 index=2: tertiary chunk (more extended precision bits)."""
    instructions = [
      s_mov_b32(s[0], 0x00000000),  # low bits of 1.0
      s_mov_b32(s[1], 0x3ff00000),  # high bits of 1.0
      v_trig_preop_f64(v[0], abs(s[0]), 2),  # index 2
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    # Index 2 gives the next 53 bits after index 1, should be tiny (~1e-32)
    self.assertLess(abs(result), 1e-30, msg=f"Expected very tiny value, got {result}")

  def test_trig_preop_f64_sum_equals_two_over_pi(self):
    """V_TRIG_PREOP_F64: sum of chunks 0,1,2 should equal 2/PI."""
    two_over_pi = 2.0 / math.pi
    instructions = [
      s_mov_b32(s[0], 0x00000000),  # low bits of 1.0
      s_mov_b32(s[1], 0x3ff00000),  # high bits of 1.0
      v_trig_preop_f64(v[0], abs(s[0]), 0),  # index 0 -> v[0:1]
      v_trig_preop_f64(v[2], abs(s[0]), 1),  # index 1 -> v[2:3]
      v_trig_preop_f64(v[4], abs(s[0]), 2),  # index 2 -> v[4:5]
    ]
    st = run_program(instructions, n_lanes=1)
    p0 = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    p1 = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    p2 = i642f(st.vgpr[0][4] | (st.vgpr[0][5] << 32))
    total = p0 + p1 + p2
    self.assertAlmostEqual(total, two_over_pi, places=14, msg=f"Expected {two_over_pi}, got {total} (p0={p0}, p1={p1}, p2={p2})")

  def test_trig_preop_f64_large_input(self):
    """V_TRIG_PREOP_F64 with larger input should adjust shift based on exponent."""
    # For x=2.0, exponent(2.0)=1024 which is <= 1077, so no adjustment
    # But let's test with x=2^60 where exponent > 1077
    large_val = 2.0 ** 60  # exponent = 1083 > 1077
    large_bits = f2i64(large_val)
    instructions = [
      s_mov_b32(s[0], large_bits & 0xffffffff),
      s_mov_b32(s[1], (large_bits >> 32) & 0xffffffff),
      v_trig_preop_f64(v[0], abs(s[0]), 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][0] | (st.vgpr[0][1] << 32))
    # Result should still be a valid float (not NaN or inf)
    self.assertFalse(math.isnan(result), "Result should not be NaN")
    self.assertFalse(math.isinf(result), "Result should not be inf")


if __name__ == "__main__":
  unittest.main()
