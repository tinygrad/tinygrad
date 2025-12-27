#!/usr/bin/env python3
"""Regression tests for compiled pseudocode functions vs interpreter."""
import unittest, struct, math
from extra.assembly.rdna3.autogen.pseudocode_functions import get_compiled_functions
from extra.assembly.rdna3.test.pseudocode import PseudocodeInterpreter, get_pseudocode
from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, VOP1Op, VOP2Op, VOP3Op

def _f32(i): return struct.unpack('<f', struct.pack('<I', i & 0xffffffff))[0]
def _f16(i): return struct.unpack('<e', struct.pack('<H', i & 0xffff))[0]
def _f64(i): return struct.unpack('<d', struct.pack('<Q', i & 0xffffffffffffffff))[0]
def to_f32_bits(f): return struct.unpack('<I', struct.pack('<f', f))[0]
def to_f16_bits(f): return struct.unpack('<H', struct.pack('<e', float(f)))[0]
def to_f64_bits(f): return struct.unpack('<Q', struct.pack('<d', f))[0]

class TestCompiledPseudocode(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.compiled = get_compiled_functions()

  def _run(self, cls, op, s0=0, s1=0, s2=0, d0=0, scc=0, vcc=0, lane=0, exec_mask=0xffffffff):
    fn = self.compiled[cls][op]
    return fn(s0, s1, s2, d0, scc, vcc, lane, exec_mask, 0, None, {})

  # ═══════════════════════════════════════════════════════════════════════════════
  # SOP1 tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_s_mov_b32(self):
    r = self._run(SOP1Op, SOP1Op.S_MOV_B32, s0=0x12345678)
    self.assertEqual(r['d0'], 0x12345678)

  def test_s_mov_b64(self):
    r = self._run(SOP1Op, SOP1Op.S_MOV_B64, s0=0x123456789abcdef0)
    self.assertEqual(r['d0'], 0x123456789abcdef0)
    self.assertTrue(r.get('d0_64'))

  def test_s_not_b32(self):
    r = self._run(SOP1Op, SOP1Op.S_NOT_B32, s0=0x0)
    self.assertEqual(r['d0'], 0xffffffff)
    self.assertEqual(r['scc'], 1)
    r = self._run(SOP1Op, SOP1Op.S_NOT_B32, s0=0xffffffff)
    self.assertEqual(r['d0'], 0x0)
    self.assertEqual(r['scc'], 0)

  def test_s_abs_i32(self):
    # abs(1) = 1
    r = self._run(SOP1Op, SOP1Op.S_ABS_I32, s0=1)
    self.assertEqual(r['d0'], 1)
    self.assertEqual(r['scc'], 1)  # != 0
    # abs(-1) = 1
    r = self._run(SOP1Op, SOP1Op.S_ABS_I32, s0=0xffffffff)
    self.assertEqual(r['d0'], 1)
    self.assertEqual(r['scc'], 1)
    # abs(0) = 0
    r = self._run(SOP1Op, SOP1Op.S_ABS_I32, s0=0)
    self.assertEqual(r['d0'], 0)
    self.assertEqual(r['scc'], 0)

  # ═══════════════════════════════════════════════════════════════════════════════
  # SOP2 tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_s_add_u32(self):
    r = self._run(SOP2Op, SOP2Op.S_ADD_U32, s0=1, s1=2)
    self.assertEqual(r['d0'], 3)

  def test_s_sub_u32(self):
    r = self._run(SOP2Op, SOP2Op.S_SUB_U32, s0=5, s1=3)
    self.assertEqual(r['d0'], 2)

  def test_s_min_i32(self):
    r = self._run(SOP2Op, SOP2Op.S_MIN_I32, s0=0, s1=1)
    self.assertEqual(r['d0'], 0)
    r = self._run(SOP2Op, SOP2Op.S_MIN_I32, s0=0xffffffff, s1=1)  # -1 vs 1
    self.assertEqual(r['d0'], 0xffffffff)  # -1 is smaller

  def test_s_max_i32(self):
    r = self._run(SOP2Op, SOP2Op.S_MAX_I32, s0=0, s1=1)
    self.assertEqual(r['d0'], 1)
    r = self._run(SOP2Op, SOP2Op.S_MAX_I32, s0=0xffffffff, s1=1)  # -1 vs 1
    self.assertEqual(r['d0'], 1)

  def test_s_and_b32(self):
    r = self._run(SOP2Op, SOP2Op.S_AND_B32, s0=0xff00ff00, s1=0x0f0f0f0f)
    self.assertEqual(r['d0'], 0x0f000f00)

  def test_s_or_b32(self):
    r = self._run(SOP2Op, SOP2Op.S_OR_B32, s0=0xff00ff00, s1=0x0f0f0f0f)
    self.assertEqual(r['d0'], 0xff0fff0f)

  def test_s_xor_b32(self):
    r = self._run(SOP2Op, SOP2Op.S_XOR_B32, s0=0xff00ff00, s1=0x0f0f0f0f)
    self.assertEqual(r['d0'], 0xf00ff00f)

  def test_s_and_b64(self):
    r = self._run(SOP2Op, SOP2Op.S_AND_B64, s0=0xff, s1=0x0f)
    self.assertEqual(r['d0'], 0x0f)
    self.assertEqual(r['scc'], 1)  # != 0
    r = self._run(SOP2Op, SOP2Op.S_AND_B64, s0=0xff00, s1=0x00ff)
    self.assertEqual(r['d0'], 0)
    self.assertEqual(r['scc'], 0)

  def test_s_lshl_b32(self):
    r = self._run(SOP2Op, SOP2Op.S_LSHL_B32, s0=1, s1=4)
    self.assertEqual(r['d0'], 16)

  def test_s_lshr_b32(self):
    r = self._run(SOP2Op, SOP2Op.S_LSHR_B32, s0=16, s1=4)
    self.assertEqual(r['d0'], 1)

  def test_s_ashr_i32(self):
    r = self._run(SOP2Op, SOP2Op.S_ASHR_I32, s0=0x80000000, s1=4)  # -2^31 >> 4
    self.assertEqual(r['d0'], 0xf8000000)  # sign extended

  def test_s_mul_i32(self):
    r = self._run(SOP2Op, SOP2Op.S_MUL_I32, s0=6, s1=7)
    self.assertEqual(r['d0'], 42)

  # ═══════════════════════════════════════════════════════════════════════════════
  # SOPC tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_s_cmp_eq_i32(self):
    r = self._run(SOPCOp, SOPCOp.S_CMP_EQ_I32, s0=5, s1=5)
    self.assertEqual(r['scc'], 1)
    r = self._run(SOPCOp, SOPCOp.S_CMP_EQ_I32, s0=5, s1=6)
    self.assertEqual(r['scc'], 0)

  def test_s_cmp_lt_i32(self):
    r = self._run(SOPCOp, SOPCOp.S_CMP_LT_I32, s0=5, s1=6)
    self.assertEqual(r['scc'], 1)
    r = self._run(SOPCOp, SOPCOp.S_CMP_LT_I32, s0=0xffffffff, s1=0)  # -1 < 0
    self.assertEqual(r['scc'], 1)

  # ═══════════════════════════════════════════════════════════════════════════════
  # VOP1 tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_v_mov_b32(self):
    r = self._run(VOP1Op, VOP1Op.V_MOV_B32, s0=0xdeadbeef)
    self.assertEqual(r['d0'], 0xdeadbeef)

  def test_v_cvt_f32_i32(self):
    r = self._run(VOP1Op, VOP1Op.V_CVT_F32_I32, s0=42)
    self.assertAlmostEqual(_f32(r['d0']), 42.0)
    r = self._run(VOP1Op, VOP1Op.V_CVT_F32_I32, s0=0xffffffff)  # -1
    self.assertAlmostEqual(_f32(r['d0']), -1.0)

  def test_v_cvt_f32_u32(self):
    r = self._run(VOP1Op, VOP1Op.V_CVT_F32_U32, s0=42)
    self.assertAlmostEqual(_f32(r['d0']), 42.0)

  def test_v_cvt_i32_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_CVT_I32_F32, s0=to_f32_bits(42.7))
    self.assertEqual(r['d0'], 42)
    r = self._run(VOP1Op, VOP1Op.V_CVT_I32_F32, s0=to_f32_bits(-42.7))
    self.assertEqual(r['d0'] & 0xffffffff, (-42) & 0xffffffff)

  def test_v_cvt_f16_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_CVT_F16_F32, s0=to_f32_bits(1.0))
    self.assertEqual(r['d0'], 0x3c00)  # f16 1.0

  def test_v_cvt_f32_f16(self):
    r = self._run(VOP1Op, VOP1Op.V_CVT_F32_F16, s0=0x3c00)  # f16 1.0
    self.assertAlmostEqual(_f32(r['d0']), 1.0)

  def test_v_sqrt_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_SQRT_F32, s0=to_f32_bits(4.0))
    self.assertAlmostEqual(_f32(r['d0']), 2.0, places=5)
    r = self._run(VOP1Op, VOP1Op.V_SQRT_F32, s0=to_f32_bits(-1.0))
    self.assertTrue(math.isnan(_f32(r['d0'])))

  def test_v_rcp_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_RCP_F32, s0=to_f32_bits(2.0))
    self.assertAlmostEqual(_f32(r['d0']), 0.5, places=5)

  def test_v_rsq_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_RSQ_F32, s0=to_f32_bits(4.0))
    self.assertAlmostEqual(_f32(r['d0']), 0.5, places=5)

  def test_v_trunc_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_TRUNC_F32, s0=to_f32_bits(3.7))
    self.assertAlmostEqual(_f32(r['d0']), 3.0)
    r = self._run(VOP1Op, VOP1Op.V_TRUNC_F32, s0=to_f32_bits(-3.7))
    self.assertAlmostEqual(_f32(r['d0']), -3.0)

  def test_v_ceil_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_CEIL_F32, s0=to_f32_bits(3.2))
    self.assertAlmostEqual(_f32(r['d0']), 4.0)
    r = self._run(VOP1Op, VOP1Op.V_CEIL_F32, s0=to_f32_bits(-3.2))
    self.assertAlmostEqual(_f32(r['d0']), -3.0)

  def test_v_floor_f32(self):
    r = self._run(VOP1Op, VOP1Op.V_FLOOR_F32, s0=to_f32_bits(3.7))
    self.assertAlmostEqual(_f32(r['d0']), 3.0)
    r = self._run(VOP1Op, VOP1Op.V_FLOOR_F32, s0=to_f32_bits(-3.7))
    self.assertAlmostEqual(_f32(r['d0']), -4.0)

  # ═══════════════════════════════════════════════════════════════════════════════
  # VOP2 tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_v_add_f32(self):
    r = self._run(VOP2Op, VOP2Op.V_ADD_F32, s0=to_f32_bits(1.5), s1=to_f32_bits(2.5))
    self.assertAlmostEqual(_f32(r['d0']), 4.0)

  def test_v_sub_f32(self):
    r = self._run(VOP2Op, VOP2Op.V_SUB_F32, s0=to_f32_bits(5.0), s1=to_f32_bits(3.0))
    self.assertAlmostEqual(_f32(r['d0']), 2.0)

  def test_v_mul_f32(self):
    r = self._run(VOP2Op, VOP2Op.V_MUL_F32, s0=to_f32_bits(3.0), s1=to_f32_bits(4.0))
    self.assertAlmostEqual(_f32(r['d0']), 12.0)

  def test_v_and_b32(self):
    r = self._run(VOP2Op, VOP2Op.V_AND_B32, s0=0xff00ff00, s1=0x0f0f0f0f)
    self.assertEqual(r['d0'], 0x0f000f00)

  def test_v_or_b32(self):
    r = self._run(VOP2Op, VOP2Op.V_OR_B32, s0=0xff00ff00, s1=0x0f0f0f0f)
    self.assertEqual(r['d0'], 0xff0fff0f)

  def test_v_xor_b32(self):
    r = self._run(VOP2Op, VOP2Op.V_XOR_B32, s0=0xff00ff00, s1=0x0f0f0f0f)
    self.assertEqual(r['d0'], 0xf00ff00f)

  def test_v_lshlrev_b32(self):
    r = self._run(VOP2Op, VOP2Op.V_LSHLREV_B32, s0=4, s1=1)  # 1 << 4
    self.assertEqual(r['d0'], 16)

  def test_v_lshrrev_b32(self):
    r = self._run(VOP2Op, VOP2Op.V_LSHRREV_B32, s0=4, s1=16)  # 16 >> 4
    self.assertEqual(r['d0'], 1)

  def test_v_ashrrev_i32(self):
    r = self._run(VOP2Op, VOP2Op.V_ASHRREV_I32, s0=4, s1=0x80000000)  # -2^31 >> 4
    self.assertEqual(r['d0'], 0xf8000000)

  def test_v_max_f32(self):
    r = self._run(VOP2Op, VOP2Op.V_MAX_F32, s0=to_f32_bits(3.0), s1=to_f32_bits(5.0))
    self.assertAlmostEqual(_f32(r['d0']), 5.0)

  def test_v_min_f32(self):
    r = self._run(VOP2Op, VOP2Op.V_MIN_F32, s0=to_f32_bits(3.0), s1=to_f32_bits(5.0))
    self.assertAlmostEqual(_f32(r['d0']), 3.0)

  def test_v_mul_i32_i24(self):
    r = self._run(VOP2Op, VOP2Op.V_MUL_I32_I24, s0=100, s1=200)
    self.assertEqual(r['d0'], 20000)

  # ═══════════════════════════════════════════════════════════════════════════════
  # VOP3 tests
  # ═══════════════════════════════════════════════════════════════════════════════

  def test_v_fma_f32(self):
    # a*b + c = 2*3 + 4 = 10
    r = self._run(VOP3Op, VOP3Op.V_FMA_F32, s0=to_f32_bits(2.0), s1=to_f32_bits(3.0), s2=to_f32_bits(4.0))
    self.assertAlmostEqual(_f32(r['d0']), 10.0)

  def test_v_mad_u32_u24(self):
    # a*b + c = 10*20 + 5 = 205
    r = self._run(VOP3Op, VOP3Op.V_MAD_U32_U24, s0=10, s1=20, s2=5)
    self.assertEqual(r['d0'], 205)

  def test_v_lshlrev_b64(self):
    r = self._run(VOP3Op, VOP3Op.V_LSHLREV_B64, s0=4, s1=1)  # 1 << 4
    self.assertEqual(r['d0'], 16)
    self.assertTrue(r.get('d0_64'))

  def test_v_lshrrev_b64(self):
    r = self._run(VOP3Op, VOP3Op.V_LSHRREV_B64, s0=4, s1=0x100)  # 0x100 >> 4
    self.assertEqual(r['d0'], 0x10)

if __name__ == '__main__':
  unittest.main()
