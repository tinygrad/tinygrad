import unittest
from extra.assembly.amd.dsl2 import *

class TestRegisters(unittest.TestCase):
  def test_vgpr_single(self):
    self.assertEqual(repr(v[5]), "v[5]")
    self.assertEqual(v[5].idx, 5)
    self.assertEqual(v[5].count, 1)

  def test_sgpr_single(self):
    self.assertEqual(repr(s[10]), "s[10]")
    self.assertEqual(s[10].idx, 10)

  def test_vgpr_range(self):
    self.assertEqual(repr(v[0:4]), "v[0:3]")
    self.assertEqual(v[0:4].idx, 0)
    self.assertEqual(v[0:4].count, 4)

  def test_sgpr_range(self):
    self.assertEqual(repr(s[4:6]), "s[4:5]")
    self.assertEqual(s[4:6].count, 2)

class TestOpField(unittest.TestCase):
  def test_enum_name(self):
    self.assertEqual(VOP1Op.V_MOV_B32_E32.name, "V_MOV_B32_E32")

  def test_enum_value(self):
    self.assertEqual(VOP1Op.V_MOV_B32_E32.value, 1)

  def test_enum_comparison(self):
    self.assertEqual(VOP1Op.V_MOV_B32_E32, VOP1Op.V_MOV_B32_E32)
    self.assertNotEqual(VOP1Op.V_NOP_E32, VOP1Op.V_MOV_B32_E32)

class TestVOP1(unittest.TestCase):
  def test_class_setup(self):
    self.assertIsNotNone(VOP1._encoding)
    self.assertEqual(VOP1._size, 4)
    field_names = [n for n, _ in VOP1._fields]
    self.assertIn('op', field_names)
    self.assertIn('vdst', field_names)
    self.assertIn('src0', field_names)

  def test_encoding_vgpr_vgpr(self):
    i = VOP1(VOP1Op.V_MOV_B32_E32, v[5], v[6])
    raw = i._raw
    # Check each field
    self.assertEqual((raw >> 25) & 0x7f, 0b0111111)  # encoding
    self.assertEqual((raw >> 17) & 0xff, 5)          # vdst
    self.assertEqual((raw >> 9) & 0xff, 1)           # op
    self.assertEqual(raw & 0x1ff, 256 + 6)           # src0 (VGPR encoded)

  def test_encoding_vgpr_sgpr(self):
    i = VOP1(VOP1Op.V_MOV_B32_E32, v[5], s[10])
    raw = i._raw
    self.assertEqual((raw >> 17) & 0xff, 5)   # vdst
    self.assertEqual(raw & 0x1ff, 10)          # src0 (SGPR encoded)

  def test_to_bytes(self):
    i = VOP1(VOP1Op.V_MOV_B32_E32, v[5], v[6])
    b = i.to_bytes()
    self.assertEqual(len(b), 4)
    self.assertEqual(int.from_bytes(b, 'little'), i._raw)

  def test_from_bytes(self):
    i1 = VOP1(VOP1Op.V_MOV_B32_E32, v[5], v[6])
    i2 = VOP1.from_bytes(i1.to_bytes())
    self.assertEqual(i1._raw, i2._raw)

  def test_repr(self):
    i = VOP1(VOP1Op.V_MOV_B32_E32, v[5], v[6])
    self.assertEqual(repr(i), "v_mov_b32_e32(v[5], v[6])")

  def test_repr_sgpr_src(self):
    i = VOP1(VOP1Op.V_MOV_B32_E32, v[5], s[10])
    self.assertEqual(repr(i), "v_mov_b32_e32(v[5], s[10])")

if __name__ == "__main__":
  unittest.main()
