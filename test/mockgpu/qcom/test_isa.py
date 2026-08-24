import unittest

from test.mockgpu.qcom.corpus import (BACKWARD_BRANCH, CLZ_ZERO, END, FLOAT_LUT_MUL_PI, GLOBAL_LOAD, HALF_SOURCE_MUL,
  LOG2_ZERO, MADSH_MAGIC_DIVIDE, MADSH_REPEAT, MOVS_BROADCAST_A0, MOVS_BROADCAST_IMMEDIATE, MULL_REPEAT,
  PREDICATED_GLOBAL_LOAD, REPEATED_ADD, SFU_EXP2_REPEAT, SHRG_HALF_DEST, SIGNED_BYTE_TO_HALF, SWZ_SWAP, ir3_program)
from test.mockgpu.qcom.decoder import decode_ir3
from test.mockgpu.qcom.executor import execute_ir3


class TestA630IR3ISA(unittest.TestCase):
  def test_log2_zero_returns_negative_infinity(self):
    # log2 r0.z, r0.z
    program = ir3_program(LOG2_ZERO, END)
    regs = {('r', 0, 2): [0]}
    execute_ir3(program, regs)
    self.assertEqual(regs[('r', 0, 2)], [0xff800000])

  def test_movs_broadcast_is_rejected_instead_of_silently_moved(self):
    for instruction in (MOVS_BROADCAST_IMMEDIATE, MOVS_BROADCAST_A0):
      with self.subTest(instruction=instruction.name), \
           self.assertRaisesRegex(ValueError, r'IR3 decode failed at PC 0 .*unsupported IR3 movs broadcast'):
        decode_ir3(ir3_program(instruction))

  def test_shrg_full_sources_are_masked_after_half_destination_write(self):
    # shrg hr0.y, 16, r1.y, r2.w
    program = ir3_program(SHRG_HALF_DEST, END)
    regs = {('r', 1, 1): [0xffffffff], ('r', 2, 3): [0xffff0000]}
    execute_ir3(program, regs)
    self.assertEqual(regs[('hr', 0, 1)], [0xffff])

  def test_clz_zero_returns_all_ones(self):
    # clz.b r2.z, r1.z
    program = ir3_program(CLZ_ZERO, END)
    regs = {('r', 1, 2): [0, 5]}
    execute_ir3(program, regs)
    self.assertEqual(regs[('r', 2, 2)], [0xffffffff, 29])

  def test_compiler_signed_byte_to_half_conversion(self):
    # cov.u8s16 hr0.x, hr0.x; cov.s16f16 hr0.x, hr0.x
    program = ir3_program(SIGNED_BYTE_TO_HALF, END)
    regs = {('hr', 0, 0): [0xfd]}
    execute_ir3(program, regs)
    self.assertEqual(regs[('hr', 0, 0)], [0xc200])  # -3.0h

  def test_compiler_mull_repeat_updates_destination_and_repeated_source(self):
    # (rpt1)mull.u r0.w, r48.y, (r)r0.w
    program = ir3_program(MULL_REPEAT, END)
    regs = {('r', 48, 1): [0, 3], ('r', 0, 3): [4928, 4928], ('r', 1, 0): [672, 672]}
    execute_ir3(program, regs)
    self.assertEqual(regs[('r', 0, 3)], [0, 14784])
    self.assertEqual(regs[('r', 1, 0)], [0, 2016])

  def test_float_lut_immediates_are_decoded_and_executed(self):
    # mul.f r0.x, (2.0), (pi)
    program = ir3_program(FLOAT_LUT_MUL_PI, END)
    mul, end = decode_ir3(program)
    self.assertEqual((mul.name, mul.srcs), ("mul.f", (0x40000000, 0x40490fdb)))
    self.assertEqual(end.name, "end")

    regs = {("r", 0, 0): [0]}
    execute_ir3(program, regs)
    self.assertEqual(regs[("r", 0, 0)], [0x40c90fdb])  # 2 * pi

  def test_repeated_add_advances_both_sources_and_destination(self):
    # (rpt3) add.u r8.x, (r)r0.x, (r)r4.x
    program = ir3_program(REPEATED_ADD, END)
    add, _ = decode_ir3(program)
    self.assertEqual((add.name, add.repeat, add.repeat_srcs), ("add.u", 3, (True, True)))

    regs = {**{("r", 0, component): [10 + component] for component in range(4)},
            **{("r", 4, component): [20 + component] for component in range(4)}}
    execute_ir3(program, regs)
    self.assertEqual([regs[("r", 8, component)][0] for component in range(4)], [30, 32, 34, 36])

  def test_half_source_full_destination_float_mul(self):
    # mul.f r2.x, hr0.x, hr1.x
    program = ir3_program(HALF_SOURCE_MUL, END)
    mul, _ = decode_ir3(program)
    self.assertEqual((mul.dst, mul.srcs, mul.source_half), (("r", 2, 0), (("hr", 0, 0), ("hr", 1, 0)), True))

    regs = {("hr", 0, 0): [0x3e00], ("hr", 1, 0): [0xc080]}  # 1.5 * -2.25
    execute_ir3(program, regs)
    self.assertEqual(regs[("r", 2, 0)], [0xc0580000])  # -3.375f

  def test_signed_backward_branch_loops_until_predicate_is_false(self):
    # sub.u r0.x, r0.x, 1; cmps.u.ne p0.x, r0.x, 0; br p0.x, #-2
    program = ir3_program(BACKWARD_BRANCH, END)
    _, compare, branch, _ = decode_ir3(program)
    self.assertEqual((compare.name, compare.condition), ("cmps.u", 5))  # ne
    self.assertEqual((branch.name, branch.branch_offset, branch.srcs), ("br", -2, (("p", 62, 0),)))

    regs = {("r", 0, 0): [3], ("p", 62, 0): [0]}
    execute_ir3(program, regs)
    self.assertEqual(regs[("r", 0, 0)], [0])
    self.assertEqual(regs[("p", 62, 0)], [0])

  def test_madsh_m16_inserts_low_high_partial_product(self):
    # madsh.m16 r3.x, r0.x, r1.x, r2.x; shr.b r4.x, r3.x, 17
    program = ir3_program(MADSH_MAGIC_DIVIDE, END)
    madsh, shift, _ = decode_ir3(program)
    self.assertEqual((madsh.name, shift.name), ("madsh.m16", "shr.b"))

    # The encoded cat-3 source order is src0.low16 * src1.high16 << 16 + src2.
    regs = {("r", 0, 0): [15], ("r", 1, 0): [0xaaab0000], ("r", 2, 0): [7]}
    execute_ir3(program, regs)
    self.assertEqual(regs[("r", 3, 0)], [((0xaaab * 15 << 16) + 7) & 0xffffffff])
    self.assertEqual(regs[("r", 4, 0)], [2])

  def test_compiler_madsh_repeat_advances_all_three_sources(self):
    # (rpt3) madsh.m16 r8.x, (r)r5.x, (r)r3.w, (r)r6.y
    program = ir3_program(MADSH_REPEAT, END)
    madsh, _ = decode_ir3(program)
    self.assertEqual((madsh.name, madsh.repeat, madsh.repeat_srcs), ('madsh.m16', 3, (True, True, True)))
    regs = {
      **{('r', 5, component): [(component + 1) * 10] for component in range(4)},
      ('r', 3, 3): [1 << 16], ('r', 4, 0): [2 << 16], ('r', 4, 1): [3 << 16], ('r', 4, 2): [4 << 16],
      ('r', 6, 1): [100], ('r', 6, 2): [200], ('r', 6, 3): [300], ('r', 7, 0): [400],
    }
    execute_ir3(program, regs)
    self.assertEqual([regs[('r', 8, component)][0] for component in range(4)],
                     [((component + 1) * (component + 1) * 10 << 16) + (component + 1) * 100 for component in range(4)])

  def test_sfu_repeat_advances_source_and_destination(self):
    # (rpt3) exp2 r4.x, (r)r0.x
    program = ir3_program(SFU_EXP2_REPEAT, END)
    exp2, _ = decode_ir3(program)
    self.assertEqual((exp2.name, exp2.repeat, exp2.repeat_srcs), ("exp2", 3, (True,)))

    regs = {("r", 0, component): [bits] for component, bits in enumerate((0x00000000, 0x3f800000, 0x40000000, 0x40400000))}
    execute_ir3(program, regs)
    self.assertEqual([regs[("r", 4, component)][0] for component in range(4)],
                     [0x3f800000, 0x40000000, 0x40800000, 0x41000000])

  def test_swz_reads_all_sources_before_writing_destinations(self):
    # swz.u32u32 r1.x, r0.x, r0.x, r1.x
    program = ir3_program(SWZ_SWAP, END)
    swz, _ = decode_ir3(program)
    self.assertEqual((swz.name, swz.dst, swz.srcs),
                     ("swz", (("r", 1, 0), ("r", 0, 0)), (("r", 0, 0), ("r", 1, 0))))

    regs = {("r", 0, 0): [0x11111111], ("r", 1, 0): [0x22222222]}
    execute_ir3(program, regs)
    self.assertEqual(regs[("r", 0, 0)], [0x22222222])
    self.assertEqual(regs[("r", 1, 0)], [0x11111111])

  def test_global_load_bounds_error_identifies_opcode_and_program_counter(self):
    # ldg.u32 r0.x, g[r1.x], 1
    program = ir3_program(GLOBAL_LOAD, END)
    load, _ = decode_ir3(program)
    self.assertEqual(load.name, "ldg")

    def reject_range(address, size):
      self.assertEqual((address, size), (0x1234, 4))
      raise IndexError("test range")

    regs = {("r", 1, 0): [0x1234], ("r", 1, 1): [0]}
    with self.assertRaisesRegex(RuntimeError, r"requires a mapped-memory validator at PC 0"):
      execute_ir3(program, regs)
    with self.assertRaisesRegex(RuntimeError, r"IR3 ldg memory fault at PC 0, lane 0, address=0x1234") as raised:
      execute_ir3(program, regs, check_range=reject_range)
    self.assertIsInstance(raised.exception.__cause__, IndexError)

  def test_predicated_global_load_zeroes_a_speculative_out_of_range_lane(self):
    # predt; ldg.u32 r0.x, g[r1.x], 1; prede
    program = ir3_program(PREDICATED_GLOBAL_LOAD, END)
    regs = {('p', 62, 0): [1], ('r', 1, 0): [0x1234], ('r', 1, 1): [0]}
    execute_ir3(program, regs, check_range=lambda _address, _size: (_ for _ in ()).throw(IndexError('speculative')))
    self.assertEqual(regs[('r', 0, 0)], [0])
