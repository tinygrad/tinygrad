#!/usr/bin/env python3
"""Test that PDF parser correctly extracts format fields."""
import unittest
from extra.assembly.rdna3.autogen import *

class TestPDFParser(unittest.TestCase):
  """Verify format classes have correct fields from PDF parsing."""

  def test_sop2_fields(self):
    """SOP2 should have op, sdst, ssrc0, ssrc1."""
    self.assertIn('op', SOP2._fields)
    self.assertIn('sdst', SOP2._fields)
    self.assertIn('ssrc0', SOP2._fields)
    self.assertIn('ssrc1', SOP2._fields)
    # Check bit positions
    self.assertEqual(SOP2._fields['op'].hi, 29)
    self.assertEqual(SOP2._fields['op'].lo, 23)
    self.assertEqual(SOP2._fields['sdst'].hi, 22)
    self.assertEqual(SOP2._fields['sdst'].lo, 16)

  def test_sop1_fields(self):
    """SOP1 should have op, sdst, ssrc0 with correct bit positions."""
    self.assertIn('op', SOP1._fields)
    self.assertIn('sdst', SOP1._fields)
    self.assertIn('ssrc0', SOP1._fields)
    # SOP1 must NOT have simm16 (that's SOPK)
    self.assertNotIn('simm16', SOP1._fields)
    # Verify bit positions - ssrc0 is bits[7:0], op is bits[15:8]
    self.assertEqual(SOP1._fields['ssrc0'].hi, 7)
    self.assertEqual(SOP1._fields['ssrc0'].lo, 0)
    self.assertEqual(SOP1._fields['op'].hi, 15)
    self.assertEqual(SOP1._fields['op'].lo, 8)
    # SOP1 encoding is 0b101111101 at bits[31:23]
    self.assertEqual(SOP1._encoding[0].hi, 31)
    self.assertEqual(SOP1._encoding[0].lo, 23)
    self.assertEqual(SOP1._encoding[1], 0b101111101)

  def test_vop2_fields(self):
    """VOP2 should have op, vdst, src0, vsrc1."""
    self.assertIn('op', VOP2._fields)
    self.assertIn('vdst', VOP2._fields)
    self.assertIn('src0', VOP2._fields)
    self.assertIn('vsrc1', VOP2._fields)

  def test_vop3_fields(self):
    """VOP3 should have op, vdst, src0, src1, src2 and modifiers."""
    self.assertIn('op', VOP3._fields)
    self.assertIn('vdst', VOP3._fields)
    self.assertIn('src0', VOP3._fields)
    self.assertIn('src1', VOP3._fields)
    self.assertIn('src2', VOP3._fields)
    # VOP3 is 64-bit
    self.assertEqual(VOP3._size(), 8)

  def test_vop3sd_fields(self):
    """VOP3SD should have all fields including src0/src1/src2 from page continuation."""
    # VOP3SD table spans pages 175-176, src fields are on continuation page
    self.assertIn('op', VOP3SD._fields)
    self.assertIn('vdst', VOP3SD._fields)
    self.assertIn('sdst', VOP3SD._fields)
    self.assertIn('src0', VOP3SD._fields)
    self.assertIn('src1', VOP3SD._fields)
    self.assertIn('src2', VOP3SD._fields)
    self.assertIn('omod', VOP3SD._fields)
    self.assertIn('neg', VOP3SD._fields)
    # Verify bit positions for src fields (from second DWORD)
    self.assertEqual(VOP3SD._fields['src0'].hi, 40)
    self.assertEqual(VOP3SD._fields['src0'].lo, 32)
    self.assertEqual(VOP3SD._fields['src1'].hi, 49)
    self.assertEqual(VOP3SD._fields['src1'].lo, 41)
    self.assertEqual(VOP3SD._fields['src2'].hi, 58)
    self.assertEqual(VOP3SD._fields['src2'].lo, 50)
    # VOP3SD is 64-bit
    self.assertEqual(VOP3SD._size(), 8)
    # Should not have duplicate fields
    field_names = [name for name in VOP3SD._fields.keys()]
    self.assertEqual(len(field_names), len(set(field_names)), "VOP3SD has duplicate fields")

  def test_flat_has_vdst(self):
    """FLAT should have vdst field (was missing before fix)."""
    self.assertIn('vdst', FLAT._fields)
    self.assertEqual(FLAT._fields['vdst'].hi, 63)
    self.assertEqual(FLAT._fields['vdst'].lo, 56)

  def test_flat_fields(self):
    """FLAT should have all required fields."""
    for field in ['op', 'vdst', 'addr', 'data', 'saddr', 'offset']:
      self.assertIn(field, FLAT._fields, f"FLAT missing {field}")

  def test_smem_fields(self):
    """SMEM should have sbase, sdata, offset."""
    self.assertIn('sbase', SMEM._fields)
    self.assertIn('sdata', SMEM._fields)
    self.assertIn('offset', SMEM._fields)
    self.assertIn('soffset', SMEM._fields)

  def test_sopp_no_extra_fields(self):
    """SOPP should only have op and simm16 (no sdst from page break merge)."""
    # SOPP should NOT have sbase, sdata, etc from SMEM (page break issue)
    self.assertNotIn('sbase', SOPP._fields)
    self.assertNotIn('sdata', SOPP._fields)
    self.assertIn('op', SOPP._fields)
    self.assertIn('simm16', SOPP._fields)

  def test_encoding_bits(self):
    """Verify encoding bits are correct for all major formats."""
    # SOP2 encoding is 10 at bits[31:30]
    self.assertEqual(SOP2._encoding[0].hi, 31)
    self.assertEqual(SOP2._encoding[0].lo, 30)
    self.assertEqual(SOP2._encoding[1], 0b10)
    # SOPK encoding is 1011 at bits[31:28]
    self.assertEqual(SOPK._encoding[0].hi, 31)
    self.assertEqual(SOPK._encoding[0].lo, 28)
    self.assertEqual(SOPK._encoding[1], 0b1011)
    # SOPP encoding is 101111111 at bits[31:23]
    self.assertEqual(SOPP._encoding[0].hi, 31)
    self.assertEqual(SOPP._encoding[0].lo, 23)
    self.assertEqual(SOPP._encoding[1], 0b101111111)
    # VOP1 encoding is 0111111 at bits[31:25]
    self.assertEqual(VOP1._encoding[0].hi, 31)
    self.assertEqual(VOP1._encoding[0].lo, 25)
    self.assertEqual(VOP1._encoding[1], 0b0111111)
    # VOP2 encoding is 0 at bits[31]
    self.assertEqual(VOP2._encoding[0].hi, 31)
    self.assertEqual(VOP2._encoding[0].lo, 31)
    self.assertEqual(VOP2._encoding[1], 0b0)
    # FLAT encoding is 110111 at bits[31:26]
    self.assertEqual(FLAT._encoding[0].hi, 31)
    self.assertEqual(FLAT._encoding[0].lo, 26)
    self.assertEqual(FLAT._encoding[1], 0b110111)

  def test_opcode_enums_exist(self):
    """Verify opcode enums are generated."""
    self.assertTrue(len(SOP1Op) > 50)
    self.assertTrue(len(SOP2Op) > 50)
    self.assertTrue(len(VOP1Op) > 50)
    self.assertTrue(len(VOP2Op) > 30)
    self.assertTrue(len(VOP3Op) > 200)

if __name__ == "__main__":
  unittest.main()
