import unittest
from extra.assembly.amd.dsl2 import *

class TestRegisters(unittest.TestCase):
  def test_vgpr_single(self):
    self.assertEqual(repr(v[5]), "v[5]")
    self.assertEqual(v[5].offset, 261)  # 256 + 5
    self.assertEqual(v[5].sz, 1)

  def test_sgpr_single(self):
    self.assertEqual(repr(s[10]), "s[10]")
    self.assertEqual(s[10].offset, 10)

  def test_vgpr_range(self):
    self.assertEqual(repr(v[0:3]), "v[0:3]")
    self.assertEqual(v[0:3].offset, 256)
    self.assertEqual(v[0:3].sz, 4)

  def test_sgpr_range(self):
    self.assertEqual(repr(s[4:5]), "s[4:5]")
    self.assertEqual(s[4:5].sz, 2)

  def test_ttmp_reslice(self):
    # ttmp is src[108:123], so ttmp[0] should be src[108]
    self.assertEqual(ttmp[0].offset, 108)
    self.assertEqual(ttmp[1].offset, 109)
    # ttmp[0:1] is 2 elements (inclusive slicing)
    self.assertEqual(ttmp[0:1].offset, 108)
    self.assertEqual(ttmp[0:1].sz, 2)
    # ttmp[0:1][0] should be src[108]
    self.assertEqual(ttmp[0:1][0].offset, 108)

  def test_special_regs(self):
    self.assertEqual(NULL.offset, 124)
    self.assertEqual(M0.offset, 125)
    self.assertEqual(EXEC_LO.offset, 126)
    self.assertEqual(EXEC_HI.offset, 127)
    # Check repr round-trips
    self.assertEqual(repr(NULL), "NULL")
    self.assertEqual(repr(M0), "M0")
    self.assertEqual(repr(EXEC_LO), "EXEC_LO")
    self.assertEqual(repr(EXEC), "EXEC")

  def test_vcc(self):
    self.assertEqual(VCC.offset, 106)
    self.assertEqual(VCC.sz, 2)
    self.assertEqual(VCC_LO.offset, 106)
    self.assertEqual(VCC_HI.offset, 107)
    # Check repr round-trips
    self.assertEqual(repr(VCC_LO), "VCC_LO")
    self.assertEqual(repr(VCC_HI), "VCC_HI")
    self.assertEqual(repr(VCC), "VCC")

  def test_float_constants(self):
    self.assertEqual(src[240].offset, 240)
    self.assertEqual(repr(src[240]), "0.5")
    self.assertEqual(repr(src[242]), "1.0")
    self.assertEqual(repr(src[243]), "-1.0")

  def test_int_constants(self):
    self.assertEqual(repr(src[128]), "0")
    self.assertEqual(repr(src[129]), "1")
    self.assertEqual(repr(src[192]), "64")
    self.assertEqual(repr(src[193]), "-1")
    self.assertEqual(repr(src[208]), "-16")

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
    self.assertEqual(VOP1._size, 4)
    field_names = [n for n, _ in VOP1._fields]
    self.assertIn('encoding', field_names)
    self.assertIn('op', field_names)
    self.assertIn('vdst', field_names)
    self.assertIn('src0', field_names)

  def test_encoding_vgpr_vgpr(self):
    i = VOP1(VOP1Op.V_MOV_B32_E32, v[5], v[6])
    raw = i._raw
    # Check each field
    self.assertEqual((raw >> 25) & 0x7f, 0b0111111)  # encoding
    self.assertEqual((raw >> 17) & 0xff, 5)          # vdst (just VGPR index)
    self.assertEqual((raw >> 9) & 0xff, 1)           # op
    self.assertEqual(raw & 0x1ff, 262)               # src0 (256 + 6)

  def test_encoding_vgpr_sgpr(self):
    i = VOP1(VOP1Op.V_MOV_B32_E32, v[5], s[10])
    raw = i._raw
    self.assertEqual((raw >> 17) & 0xff, 5)    # vdst (just VGPR index)
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

  def test_kwargs(self):
    i1 = VOP1(VOP1Op.V_MOV_B32_E32, v[5], v[6])
    i2 = VOP1(op=VOP1Op.V_MOV_B32_E32, vdst=v[5], src0=v[6])
    self.assertEqual(i1._raw, i2._raw)

  def test_kwargs_partial(self):
    i1 = VOP1(VOP1Op.V_MOV_B32_E32, v[5], v[6])
    i2 = VOP1(VOP1Op.V_MOV_B32_E32, src0=v[6], vdst=v[5])
    self.assertEqual(i1._raw, i2._raw)

if __name__ == "__main__":
  unittest.main()
