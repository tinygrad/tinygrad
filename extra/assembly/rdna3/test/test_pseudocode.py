#!/usr/bin/env python3
"""Test pseudocode interpreter against hand-coded ALU implementations."""
import unittest
from extra.assembly.rdna3.pseudocode import get_pseudocode, PseudocodeInterpreter, _i32, _f32, _sext
from extra.assembly.rdna3.alu import SALU, VALU, SOP1_BASE, SOP2_BASE, SOPC_BASE, SOPK_BASE, VOP1_BASE, VOP2_BASE
from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, VOP1Op, VOP2Op, VOP3Op

class TestPseudocodeInterpreter(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.instructions = get_pseudocode()
    cls.interp = PseudocodeInterpreter()

  def _test_sop2(self, name: str, s0: int, s1: int, scc: int = 0):
    """Test a SOP2 instruction against hand-coded implementation."""
    if name not in self.instructions:
      self.skipTest(f"{name} not in parsed instructions")
    info = self.instructions[name]
    opcode = info['opcode']
    key = SOP2_BASE + opcode

    if key not in SALU:
      self.skipTest(f"{name} not in SALU")

    # Hand-coded result
    expected_result, expected_scc = SALU[key](s0, s1, scc)

    # Pseudocode result (now returns dict)
    result = self.interp.execute(info['pseudocode'], s0, s1, scc=scc)
    pc_result, pc_scc = result['d0'], result['scc']

    self.assertEqual(pc_result, expected_result, f"{name}({s0:#x}, {s1:#x}) result mismatch")
    self.assertEqual(pc_scc, expected_scc, f"{name}({s0:#x}, {s1:#x}) scc mismatch")

  def _test_sop1(self, name: str, s0: int, scc: int = 0):
    """Test a SOP1 instruction against hand-coded implementation."""
    if name not in self.instructions:
      self.skipTest(f"{name} not in parsed instructions")
    info = self.instructions[name]
    opcode = info['opcode']
    key = SOP1_BASE + opcode

    if key not in SALU:
      self.skipTest(f"{name} not in SALU")

    # Hand-coded result
    expected_result, expected_scc = SALU[key](s0, 0, scc)

    # Pseudocode result (now returns dict)
    result = self.interp.execute(info['pseudocode'], s0, 0, scc=scc)
    pc_result, pc_scc = result['d0'], result['scc']

    self.assertEqual(pc_result, expected_result, f"{name}({s0:#x}) result mismatch")
    self.assertEqual(pc_scc, expected_scc, f"{name}({s0:#x}) scc mismatch")

  # SOP2 tests
  def test_s_add_u32(self):
    self._test_sop2('S_ADD_U32', 100, 50)
    self._test_sop2('S_ADD_U32', 0xffffffff, 2)  # overflow case
    self._test_sop2('S_ADD_U32', 0, 0)

  def test_s_sub_u32(self):
    self._test_sop2('S_SUB_U32', 100, 50)
    self._test_sop2('S_SUB_U32', 50, 100)  # borrow case
    self._test_sop2('S_SUB_U32', 0, 1)

  def test_s_and_b32(self):
    self._test_sop2('S_AND_B32', 0xff00ff00, 0x00ff00ff)
    self._test_sop2('S_AND_B32', 0xffffffff, 0xffffffff)
    self._test_sop2('S_AND_B32', 0, 0)

  def test_s_or_b32(self):
    self._test_sop2('S_OR_B32', 0xff00ff00, 0x00ff00ff)
    self._test_sop2('S_OR_B32', 0, 0)

  def test_s_xor_b32(self):
    self._test_sop2('S_XOR_B32', 0xff00ff00, 0x00ff00ff)
    self._test_sop2('S_XOR_B32', 0xffffffff, 0xffffffff)

  def test_s_lshl_b32(self):
    self._test_sop2('S_LSHL_B32', 1, 4)
    self._test_sop2('S_LSHL_B32', 0xffffffff, 16)
    self._test_sop2('S_LSHL_B32', 1, 31)

  def test_s_lshr_b32(self):
    self._test_sop2('S_LSHR_B32', 0x80000000, 4)
    self._test_sop2('S_LSHR_B32', 0xffffffff, 16)

  def test_s_ashr_i32(self):
    self._test_sop2('S_ASHR_I32', 0x80000000, 4)  # negative
    self._test_sop2('S_ASHR_I32', 0x7fffffff, 4)  # positive

  def test_s_mul_i32(self):
    self._test_sop2('S_MUL_I32', 100, 50)
    self._test_sop2('S_MUL_I32', 0xffffffff, 2)  # -1 * 2

  def test_s_min_i32(self):
    self._test_sop2('S_MIN_I32', 100, 50)
    self._test_sop2('S_MIN_I32', 0xffffffff, 1)  # -1 vs 1

  def test_s_max_i32(self):
    self._test_sop2('S_MAX_I32', 100, 50)
    self._test_sop2('S_MAX_I32', 0xffffffff, 1)  # -1 vs 1

  def test_s_min_u32(self):
    self._test_sop2('S_MIN_U32', 100, 50)
    self._test_sop2('S_MIN_U32', 0xffffffff, 1)

  def test_s_max_u32(self):
    self._test_sop2('S_MAX_U32', 100, 50)
    self._test_sop2('S_MAX_U32', 0xffffffff, 1)

  def test_s_cselect_b32(self):
    self._test_sop2('S_CSELECT_B32', 100, 50, scc=1)
    self._test_sop2('S_CSELECT_B32', 100, 50, scc=0)

  # SOP1 tests
  def test_s_mov_b32(self):
    self._test_sop1('S_MOV_B32', 42)
    self._test_sop1('S_MOV_B32', 0xdeadbeef)

  def test_s_not_b32(self):
    self._test_sop1('S_NOT_B32', 0)
    self._test_sop1('S_NOT_B32', 0xffffffff)

  def test_s_brev_b32(self):
    self._test_sop1('S_BREV_B32', 0x80000000)
    self._test_sop1('S_BREV_B32', 0x00000001)

  def test_s_abs_i32(self):
    self._test_sop1('S_ABS_I32', 0xffffffff)  # -1
    self._test_sop1('S_ABS_I32', 100)

  def test_s_cvt_f32_i32(self):
    self._test_sop1('S_CVT_F32_I32', 100)
    self._test_sop1('S_CVT_F32_I32', 0xffffffff)  # -1

  def test_s_cvt_f32_u32(self):
    self._test_sop1('S_CVT_F32_U32', 100)
    self._test_sop1('S_CVT_F32_U32', 0xffffffff)

class TestVectorOps(unittest.TestCase):
  """Test vector operations."""
  @classmethod
  def setUpClass(cls):
    cls.instructions = get_pseudocode()
    cls.interp = PseudocodeInterpreter()

  def test_v_add_f32(self):
    if 'V_ADD_F32' not in self.instructions:
      self.skipTest("V_ADD_F32 not parsed")
    pc = self.instructions['V_ADD_F32']['pseudocode']

    # Test basic addition
    s0, s1 = _i32(1.5), _i32(2.5)
    result = self.interp.execute(pc, s0, s1)
    self.assertAlmostEqual(_f32(result['d0']), 4.0, places=5)

    # Test with negatives
    s0, s1 = _i32(-1.0), _i32(3.0)
    result = self.interp.execute(pc, s0, s1)
    self.assertAlmostEqual(_f32(result['d0']), 2.0, places=5)

  def test_v_sub_f32(self):
    if 'V_SUB_F32' not in self.instructions:
      self.skipTest("V_SUB_F32 not parsed")
    pc = self.instructions['V_SUB_F32']['pseudocode']

    s0, s1 = _i32(5.0), _i32(3.0)
    result = self.interp.execute(pc, s0, s1)
    self.assertAlmostEqual(_f32(result['d0']), 2.0, places=5)

  def test_v_mul_f32(self):
    if 'V_MUL_F32' not in self.instructions:
      self.skipTest("V_MUL_F32 not parsed")
    pc = self.instructions['V_MUL_F32']['pseudocode']

    s0, s1 = _i32(2.0), _i32(3.0)
    result = self.interp.execute(pc, s0, s1)
    self.assertAlmostEqual(_f32(result['d0']), 6.0, places=5)

  def test_v_mov_b32(self):
    if 'V_MOV_B32' not in self.instructions:
      self.skipTest("V_MOV_B32 not parsed")
    pc = self.instructions['V_MOV_B32']['pseudocode']

    result = self.interp.execute(pc, 0xdeadbeef, 0)
    self.assertEqual(result['d0'], 0xdeadbeef)

if __name__ == "__main__":
  unittest.main()
