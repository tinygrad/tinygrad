#!/usr/bin/env python3
"""Test pseudocode interpreter and compiler."""
import unittest, math
from extra.assembly.rdna3.test.pseudocode import get_pseudocode, PseudocodeInterpreter, _i32, _f32, _sext, _compile_pseudocode, _f16, _i16
from extra.assembly.rdna3.autogen import VOP1Op, VOP2Op, VOP3Op, VOP3POp



class TestPseudocodeCompiler(unittest.TestCase):
  """Test that compiled pseudocode produces same results as interpreter."""

  def _run_both(self, pseudocode: str, s0: int, s1: int, s2: int = 0, d0: int = 0):
    """Run pseudocode through both interpreter and compiler, compare results."""
    interp = PseudocodeInterpreter()
    interp_result = interp.execute(pseudocode, s0, s1, s2, d0=d0)

    compiled_fn = _compile_pseudocode(pseudocode)
    compiled_result = compiled_fn(s0, s1, s2, d0, 0, 0, 0, 0xffffffff, 0, None, {})

    self.assertEqual(interp_result['d0'], compiled_result['d0'],
                     f"d0 mismatch for pseudocode: {pseudocode[:50]}...")
    return compiled_result

  def test_simple_add_f32(self):
    """Test D0.f32 = S0.f32 + S1.f32"""
    result = self._run_both('D0.f32 = S0.f32 + S1.f32', _i32(1.5), _i32(2.5))
    self.assertAlmostEqual(_f32(result['d0']), 4.0, places=5)

  def test_simple_mul_f32(self):
    """Test D0.f32 = S0.f32 * S1.f32"""
    result = self._run_both('D0.f32 = S0.f32 * S1.f32', _i32(2.0), _i32(3.0))
    self.assertAlmostEqual(_f32(result['d0']), 6.0, places=5)

  def test_simple_sub_f32(self):
    """Test D0.f32 = S0.f32 - S1.f32"""
    result = self._run_both('D0.f32 = S0.f32 - S1.f32', _i32(5.0), _i32(3.0))
    self.assertAlmostEqual(_f32(result['d0']), 2.0, places=5)

  def test_simple_add_u32(self):
    """Test D0.u32 = S0.u32 + S1.u32"""
    result = self._run_both('D0.u32 = S0.u32 + S1.u32', 100, 200)
    self.assertEqual(result['d0'], 300)

  def test_simple_and_b32(self):
    """Test D0.b32 = S0.b32 & S1.b32"""
    result = self._run_both('D0.b32 = S0.b32 & S1.b32', 0xff00ff00, 0x00ff00ff)
    self.assertEqual(result['d0'], 0)

  def test_simple_or_b32(self):
    """Test D0.b32 = S0.b32 | S1.b32"""
    result = self._run_both('D0.b32 = S0.b32 | S1.b32', 0xff00ff00, 0x00ff00ff)
    self.assertEqual(result['d0'], 0xffffffff)

  def test_simple_mov(self):
    """Test D0.b32 = S0.b32"""
    result = self._run_both('D0.b32 = S0.b32', 0xdeadbeef, 0)
    self.assertEqual(result['d0'], 0xdeadbeef)

  def test_if_else(self):
    """Test if/else control flow compilation."""
    pseudocode = """if S0.u32 > S1.u32 then
D0.u32 = S0.u32
else
D0.u32 = S1.u32
endif"""
    # S0 > S1 case
    result = self._run_both(pseudocode, 100, 50)
    self.assertEqual(result['d0'], 100)
    # S0 <= S1 case
    result = self._run_both(pseudocode, 50, 100)
    self.assertEqual(result['d0'], 100)

  def test_fma(self):
    """Test fma function compilation."""
    pseudocode = 'D0.f32 = fma(S0.f32, S1.f32, S2.f32)'
    result = self._run_both(pseudocode, _i32(2.0), _i32(3.0), _i32(1.0))
    self.assertAlmostEqual(_f32(result['d0']), 7.0, places=5)  # 2*3+1=7

  def test_compiled_functions_cached(self):
    """Test that compiled functions are cached."""
    from extra.assembly.rdna3.test.pseudocode import _COMPILED_CACHE
    pseudocode = 'D0.f32 = S0.f32 + S1.f32'
    fn1 = _compile_pseudocode(pseudocode)
    fn2 = _compile_pseudocode(pseudocode)
    self.assertIs(fn1, fn2)  # Same function object

  def test_real_vop2_add_f32(self):
    """Test real V_ADD_F32 pseudocode compilation."""
    pc = get_pseudocode()
    vop2_pc = pc.get(VOP2Op, {})
    if VOP2Op.V_ADD_F32 not in vop2_pc:
      self.skipTest("V_ADD_F32 pseudocode not available")

    pseudocode = vop2_pc[VOP2Op.V_ADD_F32]
    result = self._run_both(pseudocode, _i32(1.5), _i32(2.5))
    self.assertAlmostEqual(_f32(result['d0']), 4.0, places=5)

  def test_real_vop2_mul_f32(self):
    """Test real V_MUL_F32 pseudocode compilation."""
    pc = get_pseudocode()
    vop2_pc = pc.get(VOP2Op, {})
    if VOP2Op.V_MUL_F32 not in vop2_pc:
      self.skipTest("V_MUL_F32 pseudocode not available")

    pseudocode = vop2_pc[VOP2Op.V_MUL_F32]
    result = self._run_both(pseudocode, _i32(2.0), _i32(3.0))
    self.assertAlmostEqual(_f32(result['d0']), 6.0, places=5)

class TestVOP3PCompiler(unittest.TestCase):
  """Test VOP3P packed operation compilation."""

  def _pack_f16(self, lo: float, hi: float) -> int:
    """Pack two f16 values into a u32."""
    lo_bits = _i16(lo)
    hi_bits = _i16(hi)
    return (hi_bits << 16) | lo_bits

  def _unpack_f16(self, packed: int) -> tuple[float, float]:
    """Unpack a u32 into two f16 values."""
    lo_bits = packed & 0xffff
    hi_bits = (packed >> 16) & 0xffff
    return _f16(lo_bits), _f16(hi_bits)

  def test_pk_add_f16(self):
    """Test V_PK_ADD_F16 pseudocode."""
    pc = get_pseudocode()
    vop3p_pc = pc.get(VOP3POp, {})
    if VOP3POp.V_PK_ADD_F16 not in vop3p_pc:
      self.skipTest("V_PK_ADD_F16 pseudocode not available")

    pseudocode = vop3p_pc[VOP3POp.V_PK_ADD_F16]
    s0 = self._pack_f16(1.0, 2.0)
    s1 = self._pack_f16(3.0, 4.0)

    interp = PseudocodeInterpreter()
    result = interp.execute(pseudocode, s0, s1)

    lo, hi = self._unpack_f16(result['d0'])
    self.assertAlmostEqual(lo, 4.0, places=2)  # 1+3
    self.assertAlmostEqual(hi, 6.0, places=2)  # 2+4

  def test_pk_mul_f16(self):
    """Test V_PK_MUL_F16 pseudocode."""
    pc = get_pseudocode()
    vop3p_pc = pc.get(VOP3POp, {})
    if VOP3POp.V_PK_MUL_F16 not in vop3p_pc:
      self.skipTest("V_PK_MUL_F16 pseudocode not available")

    pseudocode = vop3p_pc[VOP3POp.V_PK_MUL_F16]
    s0 = self._pack_f16(2.0, 3.0)
    s1 = self._pack_f16(4.0, 5.0)

    interp = PseudocodeInterpreter()
    result = interp.execute(pseudocode, s0, s1)

    lo, hi = self._unpack_f16(result['d0'])
    self.assertAlmostEqual(lo, 8.0, places=2)   # 2*4
    self.assertAlmostEqual(hi, 15.0, places=2)  # 3*5

if __name__ == "__main__":
  unittest.main()
