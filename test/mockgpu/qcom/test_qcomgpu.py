import math, unittest

from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom.qcomgpu import (
  QCOMGPU, _cat0_branch_taken, _cat1_swz, _cat2_signed, _cat6_private_offset, _clz_b, _mul_s24,
  _mull_u, _madsh_m16, _absneg_s, _cov_src, _cat4_fn, _rsqrt, _log2, _exp2, _trunc_f,
  _floor_f, _sign_f, _mad_f16, _fmod)


class TestQCOMGPUDecode(unittest.TestCase):
  def test_cat3_constant_vs_inline_immediate(self):
    gpr = list(range(256))
    consts = [0] * 64
    consts[16] = 0x7f800000

    # The same raw 0x1010 field is context-sensitive in CAT3: sel.b32 uses it
    # as c4.x (constant word 16), while shift SRC1 uses it as immediate 16.
    self.assertEqual(QCOMGPU._cat3_src(0x1010, gpr, consts), 0x7f800000)
    self.assertEqual(QCOMGPU._cat3_src(0x1010, gpr, consts, immediate=True), 16)
    self.assertEqual(QCOMGPU._cat3_src(0x101e, gpr, consts, immediate=True), 30)
    self.assertEqual(QCOMGPU._cat3_src(0x0003, gpr, consts), 3)

  def test_half_float_constant_vs_half_gpr(self):
    gpr, hreg = [0] * 256, [0] * 256
    consts = [0] * 64
    # hc5.x is constant word 20. Floating half instructions interpret a
    # constant-file source as the full float32 value, not as its low fp16 bits.
    consts[20] = 0xc3e00000  # -448.0f
    hreg[3] = 0xa400         # -0.015625h

    self.assertEqual(QCOMGPU._float_src(0x1014, gpr, hreg, consts, full=False), -448.0)
    self.assertEqual(QCOMGPU._float_src(0x0003, gpr, hreg, consts, full=False), -0.015625)
    # 0x2c07 is the encoded special immediate printed by Mesa as
    # h(1/log2(e)); it is a value, not the low half of an fp32 bit pattern.
    self.assertAlmostEqual(QCOMGPU._float_src(0x2c07, gpr, hreg, consts, full=False), 0.6931471805599453)

  def test_mull_and_madsh_reconstruct_low_product(self):
    a = b = 0xfffffa9d  # low word of int64(-1379)
    out = _mull_u(a, b)
    out = _madsh_m16(a, b, out)
    out = _madsh_m16(b, a, out)
    self.assertEqual(out, 0x001d0449)

  def test_mul_s24_signed_operands(self):
    self.assertEqual(_mul_s24(3, 7), 21)
    self.assertEqual(_mul_s24(0xffffff, 9), 0xfffffff7)  # -1 * 9
    self.assertEqual(_mul_s24(0x800000, 2), 0xff000000)  # -2^23 * 2
    self.assertEqual(_mul_s24(0x1000001, 3), 3)          # truncate operands to 24 bits

  def test_cat2_signed_precision(self):
    self.assertEqual(_cat2_signed(0xffff, False), -1)
    self.assertEqual(_cat2_signed(0x8000, False), -32768)
    self.assertEqual(_cat2_signed(0x0000ffff, True), 65535)
    self.assertEqual(_cat2_signed(0xffffffff, True), -1)

  def test_clz_b_precision(self):
    self.assertEqual(_clz_b(0, True), 0xffffffff)
    self.assertEqual(_clz_b(1, True), 31)
    self.assertEqual(_clz_b(0x80000000, True), 0)
    self.assertEqual(_clz_b(0, False), 0xffffffff)
    self.assertEqual(_clz_b(1, False), 15)
    self.assertEqual(_clz_b(0x8000, False), 0)

  def test_cat0_two_predicate_branches(self):
    # Real uint64 modulo lowering: brao !p0.y, !p0.x.  With both predicates
    # true neither negated condition is true, so the branch must fall through.
    brao = 0x00b02020000000c1
    gpr = [0] * 256
    gpr[0xf8], gpr[0xf9] = 1, 1
    self.assertFalse(_cat0_branch_taken(brao, gpr))
    gpr[0xf8] = 0
    self.assertTrue(_cat0_branch_taken(brao, gpr))

    # Same predicate fields, BRAA instead of BRAO: both negated predicates
    # must be true.  BR consumes only COMP1/INV1.
    braa = (brao & ~(7 << 37)) | (2 << 37)
    gpr[0xf8], gpr[0xf9] = 0, 0
    self.assertTrue(_cat0_branch_taken(braa, gpr))
    gpr[0xf8] = 1
    self.assertFalse(_cat0_branch_taken(braa, gpr))
    br = brao & ~(7 << 37)
    gpr[0xf9] = 0
    self.assertTrue(_cat0_branch_taken(br, gpr))
    gpr[0xf9] = 1
    self.assertFalse(_cat0_branch_taken(br, gpr))

  def test_cat1_swz_parallel_pointer_halves(self):
    # Real cumprod-backward lowering uses this exact SWZ to swap the two words
    # of a 64-bit address.  The two writes must use the original source values.
    swz = 0x240cc02b002c2b2c  # swz.u32u32 r10.w, r11.x, r11.x, r10.w
    regs = [0] * 256
    regs[43], regs[44] = 0x89abcdef, 0x00007fff
    _cat1_swz(swz, regs, regs)
    self.assertEqual(regs[43], 0x00007fff)
    self.assertEqual(regs[44], 0x89abcdef)

  def test_cat6_private_offsets(self):
    # Exact Mesa encodings observed in cumprod backward spill/reload code.
    ldp4 = 0xc086002b0180c009   # ldp.u32 r10.w, p[r0.w+4], 1
    stp4 = 0xc146070401800042   # stp.u32 p[r0.w+4], r8.y, 1
    stp28 = 0xc146071c018000ea  # stp.u32 p[r0.w+28], r29.y, 1
    self.assertEqual(_cat6_private_offset(ldp4, store=False), 4)
    self.assertEqual(_cat6_private_offset(stp4, store=True), 4)
    self.assertEqual(_cat6_private_offset(stp28, store=True), 28)

  def test_absneg_s_modifiers(self):
    self.assertEqual(_absneg_s(7, 1), 0xfffffff9)       # neg
    self.assertEqual(_absneg_s(0xfffffff9, 2), 7)       # abs
    self.assertEqual(_absneg_s(0xfffffff9, 3), 0xfffffff9)  # abs then neg

  def test_cat1_u8_signed_widening(self):
    # Mesa lowers signed int8 shifts through ldg.u8 -> cov.u8s16 -> ashr.b.
    # CAT1 therefore sign-extends TYPE_U8 when the destination is signed,
    # while an unsigned destination keeps the ordinary zero-extension.
    self.assertEqual(_cov_src(0xff, mesa.TYPE_U8, mesa.TYPE_S16), -1)
    self.assertEqual(_cov_src(0xdb, mesa.TYPE_U8, mesa.TYPE_S16), -37)
    self.assertEqual(_cov_src(0xff, mesa.TYPE_U8, mesa.TYPE_U16), 255)

  def test_cat4_half_sfu_opcodes(self):
    self.assertIs(_cat4_fn(0x9), _rsqrt)
    self.assertIs(_cat4_fn(0xa), _log2)
    self.assertIs(_cat4_fn(0xb), _exp2)
    with self.assertRaises(NotImplementedError): _cat4_fn(0x8)

  def test_trunc_f_ieee_semantics(self):
    self.assertEqual(_trunc_f(3.75), 3.0)
    self.assertEqual(_trunc_f(-3.75), -3.0)
    self.assertEqual(_trunc_f(math.inf), math.inf)
    self.assertEqual(_trunc_f(-math.inf), -math.inf)
    self.assertTrue(math.isnan(_trunc_f(math.nan)))

  def test_floor_f_ieee_semantics(self):
    self.assertEqual(_floor_f(3.75), 3.0)
    self.assertEqual(_floor_f(-3.25), -4.0)
    self.assertEqual(_floor_f(math.inf), math.inf)
    self.assertEqual(_floor_f(-math.inf), -math.inf)
    self.assertTrue(math.isnan(_floor_f(math.nan)))
    self.assertLess(math.copysign(1.0, _floor_f(-0.0)), 0.0)

  def test_float_absneg_modifiers(self):
    self.assertEqual(_fmod(3.5, 0 << 14), 3.5)
    self.assertEqual(_fmod(3.5, 1 << 14), -3.5)
    self.assertEqual(_fmod(-3.5, 2 << 14), 3.5)
    self.assertEqual(_fmod(3.5, 3 << 14), -3.5)

  def test_sign_f_semantics(self):
    self.assertEqual(_sign_f(-2.0), -1.0)
    self.assertEqual(_sign_f(-0.0), 0.0)
    self.assertEqual(_sign_f(0.0), 0.0)
    self.assertEqual(_sign_f(2.0), 1.0)
    self.assertEqual(_sign_f(math.nan), 1.0)

  def test_mad_f16_modifiers_and_constants(self):
    self.assertEqual(_mad_f16(2.0, 3.0, 4.0), 10.0)
    self.assertEqual(_mad_f16(2.0, 3.0, 4.0, 1), -2.0)
    self.assertEqual(_mad_f16(2.0, 3.0, 4.0, 2), -2.0)
    self.assertEqual(_mad_f16(2.0, 3.0, 4.0, 4), 2.0)
    self.assertEqual(_mad_f16(2.0, 3.0, 4.0, sat=True), 1.0)
    self.assertEqual(_mad_f16(-2.0, 3.0, 1.0, sat=True), 0.0)

    hreg, consts = [0] * 256, [0] * 64
    hreg[3] = 0x3c00       # 1.0h
    consts[16] = 0x40000000  # 2.0f
    self.assertEqual(QCOMGPU._cat3_f16_src(3, hreg, consts), 1.0)
    self.assertEqual(QCOMGPU._cat3_f16_src(0x1010, hreg, consts), 2.0)

  def test_cat1_constant_source_bounds(self):
    # CAT1 immediate-constant mode (mode=1) must default to 0 for an index past
    # the loaded constant buffer, like every other const read, instead of
    # raising IndexError.  In-range reads return the constant word.
    sf = [0] * 256
    consts = [0] * 64
    consts[16] = 0xdeadbeef
    self.assertEqual(QCOMGPU._cat1_src(0x1010, 1, sf, consts, 0), 0xdeadbeef)  # idx 16
    self.assertEqual(QCOMGPU._cat1_src(0x1010, 1, sf, [], 0), 0)               # idx 16, empty consts
    self.assertEqual(QCOMGPU._cat1_src(0x0042, 2, sf, [], 0), 0x42)            # inline immediate
    sf[7] = 0x1234
    self.assertEqual(QCOMGPU._cat1_src(0x0007, 0, sf, [], 7), 0x1234)          # register

  def test_float_immediate_out_of_range(self):
    # A 10-bit special-immediate field with an index outside the defined table
    # is an unrecognized encoding: report it clearly, don't IndexError.
    gpr, hreg = [0] * 256, [0] * 256
    with self.assertRaises(NotImplementedError):
      QCOMGPU._float_src((5 << 11) | 20, gpr, hreg, [], full=False)

  def test_address_range_lifecycle(self):
    gpu = QCOMGPU(0)
    gpu.map_range(0x1000, 0x100)
    gpu._check(0x1040, 0x20)
    gpu.unmap_range(0x1000, 0x100)
    with self.assertRaises(RuntimeError): gpu._check(0x1040, 0x20)


if __name__ == '__main__': unittest.main()
