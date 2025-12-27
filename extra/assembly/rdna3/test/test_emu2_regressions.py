"""Regression tests for emu2.py (RDNA3 Python emulator).

These tests verify fixes for specific bugs found in emu2.py.
Run with: MOCKGPU=1 PYTHON_REMU=1 python -m pytest test/testextra/test_emu2_regressions.py -v
"""
import unittest
import numpy as np
from tinygrad.helpers import getenv

@unittest.skipUnless(getenv("MOCKGPU") and getenv("PYTHON_REMU"), 'Testing PYTHON_REMU emulator')
class TestEmu2Regressions(unittest.TestCase):

  # === Bug: VOP2 carry instructions writing to vdst instead of VCC ===
  # VOP2 V_ADD_CO_CI_U32 was writing carry output to vdst (VGPR) instead of VCC_LO (SGPR 106).
  # This corrupted buffer addresses stored in SGPRs, causing loads to return zeros.
  def test_vop2_carry_to_vcc(self):
    """VOP2 V_ADD_CO_CI_U32 carry output must go to VCC, not vdst."""
    from tinygrad import Tensor
    # This test exercises the carry chain in address calculations
    np.random.seed(0)
    x = np.random.randn(45, 65).astype(np.float32)
    t = Tensor(x)
    result = t.abs().numpy()
    np.testing.assert_allclose(result, np.abs(x), rtol=1e-5)

  # === Bug: src1=0 treated as "no source" instead of SGPR[0] ===
  # The code used `if src1 else 0` which incorrectly treated src1=0 (SGPR[0]) as falsy.
  # This caused multiplications to use 0 instead of the actual SGPR[0] value.
  def test_src1_zero_is_valid_sgpr(self):
    """src1=0 means SGPR[0], not 'no source'."""
    from tinygrad import Tensor
    # Simple multiplication - both operands come from SGPRs
    x = Tensor([-1.])
    y = Tensor([-1.])
    result = (x * y).numpy()
    np.testing.assert_allclose(result, [1.], rtol=1e-5)

  # === Bug: VOP3 V_CNDMASK_B32 using VCC instead of src2 as mask ===
  # VOP3-encoded V_CNDMASK_B32 uses src2 as the mask register, not VCC.
  # The pseudocode says VCC, but VOP3 encoding overrides this with src2.
  def test_vop3_cndmask_uses_src2_mask(self):
    """VOP3 V_CNDMASK_B32 must use src2 as mask, not VCC."""
    from tinygrad import Tensor
    # abs uses cndmask with per-element condition results
    x = Tensor([-1., 0., 1.])
    result = x.abs().numpy()
    np.testing.assert_allclose(result, [1., 0., 1.], rtol=1e-5)

  # === Bug: VOP3-encoded VOPC using inst.sdst instead of inst.vdst ===
  # VOP3-encoded VOPC comparison instructions store result in inst.vdst (not sdst).
  # The vdst field can be any SGPR, not just VCC_LO.
  def test_vop3_vopc_uses_vdst(self):
    """VOP3-encoded VOPC must write to inst.vdst, not VCC_LO."""
    from tinygrad import Tensor
    # This uses multiple comparisons that write to different SGPRs
    x = Tensor([-1., 0., 1.])
    result = x.abs().numpy()
    np.testing.assert_allclose(result, [1., 0., 1.], rtol=1e-5)

  # === Combined test for abs operation ===
  def test_abs_comprehensive(self):
    """Test abs operation which exercises multiple fixed bugs."""
    from tinygrad import Tensor
    # Various edge cases
    test_cases = [
      ([-1.], [1.]),
      ([0.], [0.]),
      ([1.], [1.]),
      ([-1., 0., 1.], [1., 0., 1.]),
      ([-3.14, 2.71], [3.14, 2.71]),
    ]
    for inp, expected in test_cases:
      result = Tensor(inp).abs().numpy()
      np.testing.assert_allclose(result, expected, rtol=1e-5, err_msg=f"abs({inp}) failed")

  # === Test multiplication with various operand combinations ===
  def test_mul_various_sources(self):
    """Test multiplication with different source register combinations."""
    from tinygrad import Tensor
    test_cases = [
      ([-1.], [-1.], [1.]),   # neg * neg
      ([1.], [1.], [1.]),     # pos * pos
      ([2.], [3.], [6.]),     # simple
      ([-2.], [3.], [-6.]),   # neg * pos
      ([0.], [5.], [0.]),     # zero
    ]
    for a, b, expected in test_cases:
      result = (Tensor(a) * Tensor(b)).numpy()
      np.testing.assert_allclose(result, expected, rtol=1e-5, err_msg=f"mul({a}, {b}) failed")

  # === Test larger array operations ===
  def test_large_array_abs(self):
    """Test abs on larger arrays to exercise different lane patterns."""
    from tinygrad import Tensor
    np.random.seed(42)
    x = np.random.randn(100, 100).astype(np.float32)
    result = Tensor(x).abs().numpy()
    np.testing.assert_allclose(result, np.abs(x), rtol=1e-5)

  def test_large_array_mul(self):
    """Test multiplication on larger arrays."""
    from tinygrad import Tensor
    np.random.seed(42)
    a = np.random.randn(100, 100).astype(np.float32)
    b = np.random.randn(100, 100).astype(np.float32)
    result = (Tensor(a) * Tensor(b)).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  # === Bug: VOP3SD carry-in operations using VCC instead of src2 register ===
  # VOP3SD V_ADD_CO_CI_U32 uses src2 as the carry input register, not VCC.
  # The pseudocode says VCC but VOP3SD encoding uses src2 for the carry source.
  def test_vop3sd_carry_in_uses_src2(self):
    """VOP3SD V_ADD_CO_CI_U32 must use src2 for carry input, not VCC."""
    from tinygrad import Tensor
    # Multi-axis reduction triggers 64-bit address arithmetic with carry chains
    x = np.ones((3, 4, 5, 6)).astype(np.float32)
    result = Tensor(x).all(axis=(1, 2)).numpy()
    expected = x.all(axis=(1, 2))
    np.testing.assert_array_equal(result, expected)

  # === Bug: sqrt/log2 of negative numbers causing exceptions ===
  # GPU sqrt/log2 return NaN for invalid inputs, not exceptions.
  def test_sqrt_negative_returns_nan(self):
    """sqrt of negative should return NaN, not throw exception."""
    from tinygrad import Tensor
    x = Tensor([-1., 0., 1., 4.])
    result = x.sqrt().numpy()
    # -1 -> NaN, 0 -> 0, 1 -> 1, 4 -> 2
    assert np.isnan(result[0]), f"sqrt(-1) should be NaN, got {result[0]}"
    np.testing.assert_allclose(result[1:], [0., 1., 2.], rtol=1e-5)

  # === Bug: S0.i24 and S1.i24 not parsed (24-bit signed integer) ===
  def test_i24_signed_integer(self):
    """24-bit signed integer fields must be parsed correctly."""
    from tinygrad import Tensor
    # Operations that use 24-bit multiplies
    x = Tensor([1, 2, 3]).cast('int32')
    y = Tensor([4, 5, 6]).cast('int32')
    result = (x * y).numpy()
    np.testing.assert_array_equal(result, [4, 10, 18])

  # === Bug: SOPC ssrc1 not being read ===
  # SOPC comparison instructions (S_CMP_*) have two source operands (ssrc0, ssrc1),
  # but exec_scalar only read ssrc1 for SOP2 instructions, not SOPC.
  # This caused comparisons to always compare against 0, breaking argmax/argmin.
  def test_sopc_reads_ssrc1(self):
    """SOPC S_CMP_* must read both ssrc0 and ssrc1."""
    from tinygrad import Tensor
    # argmax uses scalar comparisons to find the maximum index
    x = Tensor([1., 2., 3.])
    result = x.argmax().numpy()
    assert result == 2, f"argmax([1,2,3]) should be 2, got {result}"
    # Also test argmin
    y = Tensor([3., 1., 2.])
    result = y.argmin().numpy()
    assert result == 1, f"argmin([3,1,2]) should be 1, got {result}"

  # === Bug: V_READFIRSTLANE_B32 writing to VGPR instead of SGPR ===
  # V_READFIRSTLANE_B32 is encoded as VOP1 but writes to SGPR (not VGPR).
  # The vdst field is reinterpreted as SDST for this instruction.
  # This broke bfloat16 conversion which uses V_READFIRSTLANE to broadcast
  # the converted float value from VGPR to SGPR for scalar processing.
  def test_readfirstlane_writes_to_sgpr(self):
    """V_READFIRSTLANE_B32 must write to SGPR, not VGPR."""
    from tinygrad import Tensor, dtypes
    # Integer to bf16 conversion uses V_READFIRSTLANE to move float result to scalar
    t = Tensor([10000, -1, -1000, -10000, 20]).cast(dtypes.bfloat16)
    t.realize()
    back = t.cast(dtypes.float32).numpy()
    expected = [9984., -1., -1000., -9984., 20.]
    np.testing.assert_allclose(back, expected, rtol=1e-5)

  # === Bug: V_MAD_U64_U32 reading src2 as 32-bit instead of 64-bit ===
  # V_MAD_U64_U32 expects S2.u64 (64-bit), but the emulator only read 32 bits.
  # When src2 was an SGPR pair, only the low 32 bits were used, truncating addresses.
  # This caused stores to write to wrong addresses (high 32 bits were 0).
  def test_mad64_reads_64bit_src2(self):
    """V_MAD_U64_U32 must read src2 as 64-bit from consecutive SGPRs."""
    from tinygrad import Tensor
    # Matmul backward uses V_MAD_U64_U32 to compute output buffer addresses
    # The base address is a 64-bit value in consecutive SGPRs
    x = Tensor([[1., 2., 3.]], requires_grad=True)
    W = Tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], requires_grad=True)
    out = x.dot(W).sum()
    out.backward()
    # If addresses were truncated, W.grad would be all zeros
    expected = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    np.testing.assert_allclose(W.grad.numpy(), expected, rtol=1e-5)

  # === Bug: Broadcast comparison loading same value for all workgroups ===
  # When comparing tensors with broadcast (e.g., (3,4,5) >= (5,)), the scalar load
  # for the broadcast array was using the same address for all workgroups because:
  # 1. Workgroup IDs were being placed in wrong SGPRs (needed to read COMPUTE_PGM_RSRC2)
  # 2. S_ASHR_I64 was reading only 32-bit source instead of 64-bit
  def test_broadcast_comparison_uses_workgroup_id(self):
    """Broadcast comparison must load different values per workgroup."""
    from tinygrad import Tensor
    np.random.seed(0)
    a = np.random.uniform(-2, 2, (3, 4, 5)).astype(np.float32)
    b = np.random.uniform(-2, 2, (5,)).astype(np.float32)
    ta, tb = Tensor(a), Tensor(b)
    # All comparison operators should work correctly with broadcast
    np.testing.assert_array_equal((ta >= tb).numpy(), a >= b)
    np.testing.assert_array_equal((ta <= tb).numpy(), a <= b)
    np.testing.assert_array_equal((ta > tb).numpy(), a > b)
    np.testing.assert_array_equal((ta < tb).numpy(), a < b)

if __name__ == '__main__':
  unittest.main()
