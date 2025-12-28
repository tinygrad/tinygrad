#!/usr/bin/env python3
"""Tests for the RDNA3 pseudocode DSL."""
import unittest
from extra.assembly.rdna3.dsl import Reg, TypedView, SliceProxy, ExecContext, compile_pseudocode, _expr, MASK32, MASK64
from extra.assembly.rdna3.dsl import _f32, _i32, _f16, _i16, f32_to_f16

class TestReg(unittest.TestCase):
  def test_u32_read(self):
    r = Reg(0xDEADBEEF)
    self.assertEqual(int(r.u32), 0xDEADBEEF)

  def test_u32_write(self):
    r = Reg(0)
    r.u32 = 0x12345678
    self.assertEqual(r._val, 0x12345678)

  def test_f32_read(self):
    r = Reg(0x40400000)  # 3.0f
    self.assertAlmostEqual(float(r.f32), 3.0)

  def test_f32_write(self):
    r = Reg(0)
    r.f32 = 3.0
    self.assertEqual(r._val, 0x40400000)

  def test_i32_signed(self):
    r = Reg(0xFFFFFFFF)  # -1 as signed
    self.assertEqual(int(r.i32), -1)

  def test_u64(self):
    r = Reg(0xDEADBEEFCAFEBABE)
    self.assertEqual(int(r.u64), 0xDEADBEEFCAFEBABE)

  def test_f64(self):
    r = Reg(0x4008000000000000)  # 3.0 as f64
    self.assertAlmostEqual(float(r.f64), 3.0)

class TestTypedView(unittest.TestCase):
  def test_bit_slice(self):
    r = Reg(0xDEADBEEF)
    # Slices return SliceProxy which supports .u32, .u16 etc (matching pseudocode like S1.u32[1:0].u32)
    self.assertEqual(r.u32[7:0].u32, 0xEF)
    self.assertEqual(r.u32[15:8].u32, 0xBE)
    self.assertEqual(r.u32[23:16].u32, 0xAD)
    self.assertEqual(r.u32[31:24].u32, 0xDE)
    # Also works with int() for arithmetic
    self.assertEqual(int(r.u32[7:0]), 0xEF)

  def test_single_bit_read(self):
    r = Reg(0b11010101)
    self.assertEqual(r.u32[0], 1)
    self.assertEqual(r.u32[1], 0)
    self.assertEqual(r.u32[2], 1)
    self.assertEqual(r.u32[3], 0)

  def test_single_bit_write(self):
    r = Reg(0)
    r.u32[5] = 1
    r.u32[3] = 1
    self.assertEqual(r._val, 0b00101000)

  def test_nested_bit_access(self):
    # S0.u32[S1.u32[4:0]] - access bit at position from another register
    s0 = Reg(0b11010101)
    s1 = Reg(3)
    bit_pos = s1.u32[4:0]  # SliceProxy, int value = 3
    bit_val = s0.u32[int(bit_pos)]  # bit 3 of s0 = 0
    self.assertEqual(int(bit_pos), 3)
    self.assertEqual(bit_val, 0)

  def test_arithmetic(self):
    r1 = Reg(0x40400000)  # 3.0f
    r2 = Reg(0x40800000)  # 4.0f
    result = r1.f32 + r2.f32
    self.assertAlmostEqual(result, 7.0)

  def test_comparison(self):
    r1 = Reg(5)
    r2 = Reg(3)
    self.assertTrue(r1.u32 > r2.u32)
    self.assertFalse(r1.u32 < r2.u32)
    self.assertTrue(r1.u32 != r2.u32)

class TestSliceProxy(unittest.TestCase):
  def test_slice_read(self):
    r = Reg(0x56781234)
    self.assertEqual(r[15:0].u16, 0x1234)
    self.assertEqual(r[31:16].u16, 0x5678)

  def test_slice_write(self):
    r = Reg(0)
    r[15:0].u16 = 0x1234
    r[31:16].u16 = 0x5678
    self.assertEqual(r._val, 0x56781234)

  def test_slice_f16(self):
    r = Reg(0)
    r[15:0].f16 = 3.0
    self.assertAlmostEqual(_f16(r._val & 0xffff), 3.0, places=2)

class TestCompiler(unittest.TestCase):
  def test_ternary(self):
    result = _expr("a > b ? 1 : 0")
    self.assertIn("if", result)
    self.assertIn("else", result)

  def test_type_prefix_strip(self):
    self.assertEqual(_expr("1'0U"), "0")
    self.assertEqual(_expr("32'1"), "1")
    self.assertEqual(_expr("16'0xFFFF"), "0xFFFF")

  def test_suffix_strip(self):
    self.assertEqual(_expr("0ULL"), "0")
    self.assertEqual(_expr("1LL"), "1")
    self.assertEqual(_expr("5U"), "5")
    self.assertEqual(_expr("3.14F"), "3.14")

  def test_boolean_ops(self):
    self.assertIn("and", _expr("a && b"))
    self.assertIn("or", _expr("a || b"))
    self.assertIn("!=", _expr("a <> b"))

  def test_pack16(self):
    result = _expr("{ a, b }")
    self.assertIn("_pack", result)

  def test_type_cast_strip(self):
    self.assertEqual(_expr("64'U(x)"), "(x)")
    self.assertEqual(_expr("32'I(y)"), "(y)")

class TestExecContext(unittest.TestCase):
  def test_float_add(self):
    ctx = ExecContext(s0=0x40400000, s1=0x40800000)  # 3.0f, 4.0f
    ctx.D0.f32 = ctx.S0.f32 + ctx.S1.f32
    self.assertAlmostEqual(_f32(ctx.D0._val), 7.0)

  def test_float_mul(self):
    ctx = ExecContext(s0=0x40400000, s1=0x40800000)  # 3.0f, 4.0f
    ctx.run("D0.f32 = S0.f32 * S1.f32")
    self.assertAlmostEqual(_f32(ctx.D0._val), 12.0)

  def test_scc_comparison(self):
    ctx = ExecContext(s0=42, s1=42)
    ctx.run("SCC = S0.u32 == S1.u32")
    self.assertEqual(ctx.SCC._val, 1)

  def test_scc_comparison_false(self):
    ctx = ExecContext(s0=42, s1=43)
    ctx.run("SCC = S0.u32 == S1.u32")
    self.assertEqual(ctx.SCC._val, 0)

  def test_ternary(self):
    code = compile_pseudocode("D0.u32 = S0.u32 > S1.u32 ? 1'1U : 1'0U")
    ctx = ExecContext(s0=5, s1=3)
    ctx.run(code)
    self.assertEqual(ctx.D0._val, 1)

  def test_pack(self):
    code = compile_pseudocode("D0 = { S1[15:0].u16, S0[15:0].u16 }")
    ctx = ExecContext(s0=0x1234, s1=0x5678)
    ctx.run(code)
    self.assertEqual(ctx.D0._val, 0x56781234)

  def test_tmp_with_typed_access(self):
    code = compile_pseudocode("""tmp = S0.u32 + S1.u32
D0.u32 = tmp.u32""")
    ctx = ExecContext(s0=100, s1=200)
    ctx.run(code)
    self.assertEqual(ctx.D0._val, 300)

  def test_s_add_u32_pattern(self):
    # Real pseudocode pattern from S_ADD_U32
    code = compile_pseudocode("""tmp = 64'U(S0.u32) + 64'U(S1.u32)
SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U
D0.u32 = tmp.u32""")
    # Test overflow case
    ctx = ExecContext(s0=0xFFFFFFFF, s1=0x00000001)
    ctx.run(code)
    self.assertEqual(ctx.D0._val, 0)  # Wraps to 0
    self.assertEqual(ctx.SCC._val, 1)  # Carry set

  def test_s_add_u32_no_overflow(self):
    code = compile_pseudocode("""tmp = 64'U(S0.u32) + 64'U(S1.u32)
SCC = tmp >= 0x100000000ULL ? 1'1U : 1'0U
D0.u32 = tmp.u32""")
    ctx = ExecContext(s0=100, s1=200)
    ctx.run(code)
    self.assertEqual(ctx.D0._val, 300)
    self.assertEqual(ctx.SCC._val, 0)  # No carry

  def test_vcc_lane_read(self):
    ctx = ExecContext(vcc=0b1010, lane=1)
    # Lane 1 is set
    self.assertEqual(ctx.VCC.u64[1], 1)
    self.assertEqual(ctx.VCC.u64[2], 0)

  def test_vcc_lane_write(self):
    ctx = ExecContext(vcc=0, lane=0)
    ctx.VCC.u64[3] = 1
    ctx.VCC.u64[1] = 1
    self.assertEqual(ctx.VCC._val, 0b1010)

  def test_for_loop(self):
    # CTZ pattern - find first set bit
    code = compile_pseudocode("""tmp = -1
for i in 0 : 31 do
  if S0.u32[i] == 1 then
    tmp = i
D0.i32 = tmp""")
    ctx = ExecContext(s0=0b1000)  # Bit 3 is set
    ctx.run(code)
    self.assertEqual(ctx.D0._val & MASK32, 3)

  def test_result_dict(self):
    ctx = ExecContext(s0=5, s1=3)
    ctx.D0.u32 = 42
    ctx.SCC._val = 1
    result = ctx.result()
    self.assertEqual(result['d0'], 42)
    self.assertEqual(result['scc'], 1)

if __name__ == '__main__':
  unittest.main()
