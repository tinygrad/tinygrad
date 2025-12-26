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

if __name__ == '__main__':
  unittest.main()
