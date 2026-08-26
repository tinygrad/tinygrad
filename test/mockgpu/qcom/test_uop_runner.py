import ctypes, struct
import unittest
from dataclasses import replace

from test.mockgpu.qcom.decoder import decode_ir3
from test.mockgpu.qcom.executor import execute_ir3
from test.mockgpu.qcom.runner import IR3UOpLoopTimeout, IR3UOpRunner
from test.mockgpu.qcom.corpus import (BACKWARD_BRANCH, BITWISE_SHIFT_COMPARE, END, FLOAT_ALU_COMPARE, FP8_HALF_COMPARE, GLOBAL_LOAD,
  MADSH_MAGIC_DIVIDE, MADSH_REPEAT, MESA_STD_HOT_LOOP, MESA_STD_HOT_PREHEADER, MULL_REPEAT, REPEATED_ADD, SHRG_HALF_DEST, SIGNED_BYTE_TO_HALF,
  ir3_program as pack_ir3)


def ir3_program(*instructions: int): return decode_ir3(struct.pack(f'<{len(instructions)}Q', *instructions))

def std_loop_program(actual_pcs=False):
  code = pack_ir3(*(MESA_STD_HOT_PREHEADER, MESA_STD_HOT_LOOP, END) if actual_pcs else (MESA_STD_HOT_LOOP, END))
  return code, decode_ir3(code)


class TestA630UOpRunner(unittest.TestCase):
  @staticmethod
  def std_loop_state(count=65, invalid_address=False):
    data = (ctypes.c_float * count)(*[float(index + 1) for index in range(count)])
    start, end = ctypes.addressof(data), ctypes.addressof(data) + ctypes.sizeof(data)
    address = end if invalid_address else start

    def check_range(value, size):
      if not (start <= value and value + size <= end): raise IndexError('out of bounds')

    regs = {('r', 0, 3): [0], ('r', 0, 2): [0], ('r', 1, 0): [0],
            ('c', 0, 2): [address & 0xffffffff], ('c', 0, 3): [address >> 32], ('c', 4, 0): [count]}
    return data, check_range, ((start, end),), regs

  def assert_machine_equivalent(self, words, initial):
    code = struct.pack(f'<{len(words)}Q', *words)
    program = decode_ir3(code)
    expected = {key: values.copy() for key, values in initial.items()}
    # A trace request deliberately disables acceleration, making the scalar
    # decoded-instruction executor an independent oracle for this comparison.
    execute_ir3(code, expected, trace={})

    actual = {key: values.copy() for key, values in initial.items()}
    self.assertEqual(IR3UOpRunner(min_instructions=1).try_run(program, 0, actual, [True] * len(next(iter(actual.values())))),
                     len(program) - 1)
    self.assertEqual(actual, expected)
    return expected

  def test_repeated_add_matches_machine_interpreter(self):
    initial = {**{('r', 0, component): [10 + component] for component in range(4)},
               **{('r', 4, component): [20 + component] for component in range(4)}}
    self.assert_machine_equivalent((*REPEATED_ADD.words, *END.words), initial)

  def test_madsh_and_shift_match_machine_interpreter(self):
    initial = {('r', 0, 0): [15], ('r', 1, 0): [0xaaab0000], ('r', 2, 0): [7]}
    self.assert_machine_equivalent((*MADSH_MAGIC_DIVIDE.words, *END.words), initial)

    # (rpt3) madsh.m16 r8.x, (r)r5.x, (r)r3.w, (r)r6.y
    repeated_initial = {
      **{('r', 5, component): [(component + 1) * 10] for component in range(4)},
      ('r', 3, 3): [1 << 16], ('r', 4, 0): [2 << 16], ('r', 4, 1): [3 << 16], ('r', 4, 2): [4 << 16],
      ('r', 6, 1): [100], ('r', 6, 2): [200], ('r', 6, 3): [300], ('r', 7, 0): [400],
    }
    self.assert_machine_equivalent((*MADSH_REPEAT.words, *END.words), repeated_initial)

  def test_mov_mull_shrg_and_compare_match_machine_interpreter(self):
    cases = (
      ((*SIGNED_BYTE_TO_HALF.words[:1], *END.words), {('hr', 0, 0): [0xfd, 0x80, 0x7f]}),
      ((*MULL_REPEAT.words, *END.words), {('r', 48, 1): [0, 3], ('r', 0, 3): [4928, 4928], ('r', 1, 0): [672, 672]}),
      ((*SHRG_HALF_DEST.words, *END.words), {('r', 1, 1): [0xffffffff, 0xffff0000], ('r', 2, 3): [0xffff0000, 0xaaaa0000]}),
      ((*BACKWARD_BRANCH.words[1:2], *END.words), {('r', 0, 0): [0, 1, 0xffffffff]}),
    )
    for words, initial in cases:
      with self.subTest(words=words): self.assert_machine_equivalent(words, initial)

  def test_compiler_float_alu_block_matches_machine_interpreter(self):
    def f32(value): return int.from_bytes(struct.pack('<f', value), 'little')
    initial = {('r', 2, 2): [f32(1.5), f32(10.0)], ('r', 0, 2): [f32(2.0), f32(-0.5)]}
    self.assert_machine_equivalent((*FLOAT_ALU_COMPARE.words, *END.words), initial)

  def test_compiler_float_to_int_mov_handles_finite_infinities_and_nan(self):
    # Mesa A630 output for a float32-to-int32 vector cast: (rpt3) mov.s32f32 r1.z, (r)r0.z.
    words = (0x20054B0600000002, *END.words)
    move, end = ir3_program(*words)
    self.assertEqual((move.name, move.types, move.repeat, move.repeat_srcs, end.name), ('mov', (1, 5), 3, (True,), 'end'))

    # -1.75, 2.25, +Inf, -Inf, and a NaN payload.  The scalar decoded executor
    # is the oracle for the emulator's defined non-finite conversion behavior.
    source = [0xbfe00000, 0x40100000, 0x7f800000, 0xff800000, 0x7fc01234]
    initial = {('r', register, component): source.copy()
               for register, component in ((0, 2), (0, 3), (1, 0), (1, 1))}
    expected = self.assert_machine_equivalent(words, initial)
    for destination in (('r', 1, 2), ('r', 1, 3), ('r', 2, 0), ('r', 2, 1)):
      self.assertEqual(expected[destination], [0xffffffff, 2, 0, 0, 0])

  def test_compiler_half_to_int_mov_handles_finite_infinities_and_nan(self):
    # Mesa A630 output for a float16-to-int32 vector cast: (rpt3) mov.s32f16 r0.z, (r)hr0.x.
    words = (0x30014B0200000000, *END.words)
    move, end = ir3_program(*words)
    self.assertEqual((move.name, move.types, move.repeat, move.repeat_srcs, move.source_half, end.name),
                     ('mov', (0, 5), 3, (True,), True, 'end'))

    # -1.75h, 2.25h, +Inf, -Inf, and a NaN payload.
    source = [0xbf00, 0x4080, 0x7c00, 0xfc00, 0x7e34]
    initial = {('hr', 0, component): source.copy() for component in range(4)}
    expected = self.assert_machine_equivalent(words, initial)
    for destination in (('r', 0, 2), ('r', 0, 3), ('r', 1, 0), ('r', 1, 1)):
      self.assertEqual(expected[destination], [0xffffffff, 2, 0, 0, 0])

  def test_compiler_signed_int_to_float_mov_preserves_negative_values(self):
    # Mesa A630 output for an int32-to-float32 vector cast: (rpt3) mov.f32s32 r1.z, (r)r0.z.
    words = (0x30144B0600000002, *END.words)
    move, end = ir3_program(*words)
    self.assertEqual((move.name, move.types, move.repeat, move.repeat_srcs, end.name), ('mov', (5, 1), 3, (True,), 'end'))

    # -7, INT_MIN, -1, and +7 as raw u32 register words.
    source = [0xfffffff9, 0x80000000, 0xffffffff, 7]
    initial = {('r', register, component): source.copy()
               for register, component in ((0, 2), (0, 3), (1, 0), (1, 1))}
    expected = self.assert_machine_equivalent(words, initial)
    for destination in (('r', 1, 2), ('r', 1, 3), ('r', 2, 0), ('r', 2, 1)):
      self.assertEqual(expected[destination], [0xc0e00000, 0xcf000000, 0xbf800000, 0x40e00000])

  def test_u32_float_u32_round_trip_near_two_to_the_32nd(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.decoder import IR3Instruction
    import test.mockgpu.qcom.executor as executor

    # Mesa A630 output for (rpt3) mov.f32u32 r1.z, (r)r0.z.  Mesa currently
    # emits s32 for the reverse cast, so model just that U32 destination form.
    u32_to_f32, = ir3_program(0x300C4B0600000002)
    f32_to_u32 = IR3Instruction('mov', ('r', 2, 2), (('r', 1, 2),), True, 0, repeat=3, repeat_srcs=(True,),
                                 types=(mesa.TYPE_F32, mesa.TYPE_U32))
    program = (u32_to_f32, f32_to_u32)
    self.assertEqual((u32_to_f32.types, f32_to_u32.types), ((mesa.TYPE_U32, mesa.TYPE_F32), (mesa.TYPE_F32, mesa.TYPE_U32)))

    # Values at the final float32 ULP below 2^32; the last two round up to
    # exactly 2^32, which wraps to zero in the scalar decoded executor.
    source = [0xffffff00, 0xffffff01, 0xffffff80, 0xffffffff]
    initial = {('r', register, component): source.copy()
               for register, component in ((0, 2), (0, 3), (1, 0), (1, 1))}
    expected = {key: values.copy() for key, values in initial.items()}
    real_decode = executor.decode_ir3
    def decode(code, gpu_id=630, program=program, decode_real=real_decode):
      return program if isinstance(code, bytes) and code == b'u32f32u32' else decode_real(code, gpu_id)
    setattr(executor, 'decode_ir3', decode)
    try: executor.execute_ir3(b'u32f32u32', expected, trace={})
    finally: setattr(executor, 'decode_ir3', real_decode)

    actual = {key: values.copy() for key, values in initial.items()}
    self.assertEqual(IR3UOpRunner(min_instructions=1).try_run(program, 0, actual, [True] * len(source)), len(program))
    self.assertEqual(actual, expected)
    for destination in (('r', 2, 2), ('r', 2, 3), ('r', 3, 0), ('r', 3, 1)):
      self.assertEqual(expected[destination], [0xffffff00, 0xffffff00, 0, 0])

  def test_float_rounding_lowering_handles_finite_infinities_and_nan(self):
    # Valid A630 Cat2 encodings, decoder-verified as full-float unary operations.
    cases = (
      ('floor.f', 0x4130000600000002, [0xc0000000, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc01234, 0x00000000]),
      ('ceil.f', 0x4150000600000002, [0xbf800000, 0x40000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x00000000]),
      ('rndaz.f', 0x4190000600000002, [0xc0000000, 0x40000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x80000000]),
    )
    # -1.75, 1.25, +Inf, -Inf, a NaN payload, and -0. Floor/ceil canonicalize
    # signed zero through Python integers in the scalar executor; rndaz keeps the sign.
    initial = {('r', 0, 2): [0xbfe00000, 0x3fa00000, 0x7f800000, 0xff800000, 0x7fc01234, 0x80000000]}
    for name, word, scalar_result in cases:
      with self.subTest(name=name):
        program = ir3_program(word, *END.words)
        self.assertEqual((program[0].name, program[0].source_half, program[1].name), (name, False, 'end'))
        expected = self.assert_machine_equivalent((word, *END.words), initial)
        self.assertEqual(expected[('r', 1, 2)], scalar_result)

  def test_half_float_constant_uses_full_slot_or_raw_low_bits(self):
    # Mesa fp8-conversion output: cmps.f.lt hr3.x, hr0.w, hc5.x.
    words = (*FP8_HALF_COMPARE.words, *END.words)
    compare, end = ir3_program(*words)
    self.assertEqual((compare.name, compare.srcs, compare.source_half, compare.condition, end.name),
                     ('cmps.f', (('hr', 0, 3), ('hc', 5, 0)), True, 0, 'end'))

    # A populated c5.x is converted from f32 to f16.  When c5.x has zero
    # upper bits, hc5.x's raw low 16 bits are used instead.
    initial = {
      ('hr', 0, 3): [0xa400, 0x3e00, 0x0000, 0x3c00],
      ('c', 5, 0): [0xc3e00000, 0x00000000, 0x3f800000, 0x00000000],
      ('hc', 5, 0): [0x0000, 0x3c00, 0xbe00, 0x3800],
    }
    expected = self.assert_machine_equivalent(words, initial)
    self.assertEqual(expected[('hr', 3, 0)], [0, 0, 1, 0])

  def test_bit_ops_and_signed_shift_lower_from_decoded_instructions(self):
    initial = {('r', 0, 2): [0xf00000f0, 0x0000000f], ('r', 2, 2): [0x0f00ff0f, 0xfffffff0]}
    self.assert_machine_equivalent((*BITWISE_SHIFT_COMPARE.words, *END.words), initial)

  def test_memory_or_masked_blocks_fall_back_without_mutating_registers(self):
    program = ir3_program(*GLOBAL_LOAD.words, *END.words)
    regs = {('r', 1, 0): [0x1234], ('r', 1, 1): [0]}
    runner = IR3UOpRunner(min_instructions=1)
    self.assertIsNone(runner.try_run(program, 0, regs, [True]))
    self.assertIsNone(runner.try_run(program, 0, regs, [False]))

  def test_partial_masks_write_through_only_active_lanes(self):
    from test.mockgpu.qcom.decoder import IR3Instruction
    import test.mockgpu.qcom.executor as executor
    words = (*FLOAT_ALU_COMPARE.words, *END.words)
    program = decode_ir3(struct.pack(f'<{len(words)}Q', *words))

    def f32(value): return int.from_bytes(struct.pack('<f', value), 'little')
    for mask in ([True, False], [False, True], [True, True]):
      with self.subTest(mask=mask):
        initial = {('r', 2, 2): [f32(1.5), f32(10.0)], ('r', 0, 2): [f32(2.0), f32(-0.5)]}
        # Oracle: the same block with the write mask expressed as predication over the predicate register.
        expected = {key: values.copy() for key, values in initial.items()}
        expected[('p', 62, 0)] = [int(active) for active in mask]
        predicated = (IR3Instruction('predt', None, (), False, 0),) + program[:3]
        real_decode = executor.decode_ir3
        def decode(code, gpu_id=630, program=predicated, decode_real=real_decode):
          return program if isinstance(code, bytes) and code == b'masktest' else decode_real(code, gpu_id)
        setattr(executor, 'decode_ir3', decode)
        try: executor.execute_ir3(b'masktest', expected, check_range=lambda a, s: None)
        finally: setattr(executor, 'decode_ir3', real_decode)
        expected.pop(('p', 62, 0))

        actual = {key: values.copy() for key, values in initial.items()}
        self.assertEqual(IR3UOpRunner(min_instructions=1).try_run(program, 0, actual, mask), 3)
        self.assertEqual(actual, expected)

  def test_native_blocks_split_at_configured_instruction_limit(self):
    from test.mockgpu.qcom.decoder import IR3Instruction
    def add(): return IR3Instruction('add.u', ('r', 0, 0), (('r', 0, 0), 1), False, 0)
    program = tuple(add() for _ in range(5)) + ir3_program(*END.words)
    runner, regs, pcs = IR3UOpRunner(min_instructions=1, max_instructions=2), {('r', 0, 0): [0]}, []
    pc = 0
    while pc < 5:
      next_pc = runner.try_run(program, pc, regs, [True])
      self.assertIsNotNone(next_pc)
      assert next_pc is not None
      pcs.append((pc, next_pc))
      pc = next_pc
    self.assertEqual((pcs, regs[('r', 0, 0)]), ([(0, 2), (2, 4), (4, 5)], [5]))

  def test_high_pressure_program_uses_scalar_policy(self):
    from test.mockgpu.qcom.decoder import IR3Instruction
    instructions = tuple(IR3Instruction('add.u', ('r', index // 4, index % 4),
      (('r', index // 4, index % 4), 1), False, 0) for index in range(48))
    runner = IR3UOpRunner(min_instructions=1, max_average_register_slots=40)
    program = instructions + ir3_program(*END.words)
    self.assertFalse(runner.can_run_blocks(program, lanes=3))
    self.assertTrue(runner.can_run_blocks(program, lanes=64))

  def test_native_block_budget_retains_runners_and_evicts_only_locations(self):
    from test.mockgpu.qcom.decoder import IR3Instruction
    end = ir3_program(*END.words)
    add_one = IR3Instruction('add.u', ('r', 0, 0), (('r', 0, 0), 1), False, 0)
    equivalent_programs = ((add_one, *end), tuple([add_one, *end]))
    second_signature = (IR3Instruction('add.u', ('r', 0, 0), (('r', 0, 0), 2), False, 0), *end)
    runner = IR3UOpRunner(min_instructions=1, max_compiled_blocks=1, max_block_locations=1, max_programs=1)
    for program in equivalent_programs:
      regs = {('r', 0, 0): [0]}
      self.assertEqual(runner.try_run(program, 0, regs, [True]), 1)
      self.assertEqual(regs[('r', 0, 0)], [1])
      self.assertLessEqual(len(runner.compiled), 1)
      self.assertLessEqual(len(runner.cache), 1)
      self.assertLessEqual(len(runner.program_blocks), 1)
    self.assertGreaterEqual(runner.stats.cache_evictions, 1)
    regs = {('r', 0, 0): [0]}
    self.assertIsNone(runner.try_run(second_signature, 0, regs, [True]))
    self.assertEqual((regs[('r', 0, 0)], len(runner.compiled), runner.stats.block_declines), ([0], 1, 1))

    reserved = IR3UOpRunner(min_instructions=1, max_compiled_blocks=2, max_narrow_compiled_blocks=1)
    self.assertEqual(reserved.try_run(equivalent_programs[0], 0, {('r', 0, 0): [0]}, [True]), 1)
    wide_regs = {('r', 0, 0): [0] * 16}
    self.assertEqual(reserved.try_run(second_signature, 0, wide_regs, [True] * 16), 1)
    self.assertEqual((wide_regs[('r', 0, 0)], len(reserved.compiled)), ([2] * 16, 2))

    pooled = IR3UOpRunner(min_instructions=1, max_compiled_blocks=3, max_narrow_compiled_blocks=2,
                          max_regular_narrow_compiled_blocks=1)
    self.assertEqual(pooled.try_run(equivalent_programs[0], 0, {('r', 0, 0): [0]}, [True]), 1)
    self.assertIsNone(pooled.try_run(second_signature, 0, {('r', 0, 0): [0]}, [True]))
    nop = IR3Instruction('nop', None, (), False, 0)
    priority_program = (second_signature[0], *(nop for _ in range(255)), *end)
    priority_regs = {('r', 0, 0): [0]}
    self.assertEqual(pooled.try_run(priority_program, 0, priority_regs, [True]), 256)
    self.assertEqual((priority_regs[('r', 0, 0)], set(pooled.compiled_classes.values())),
                     ([2], {'regular_narrow', 'priority_narrow'}))
    add_three = IR3Instruction('add.u', ('r', 0, 0), (('r', 0, 0), 3), False, 0)
    rotated_program = (add_three, *(nop for _ in range(255)), *end)
    rotated_regs = {('r', 0, 0): [0]}
    self.assertEqual(pooled.try_run(rotated_program, 0, rotated_regs, [True]), 256)
    self.assertEqual((rotated_regs[('r', 0, 0)], len(pooled.compiled), pooled.stats.cache_evictions), ([3], 2, 1))

  def test_native_mesa_natural_loop_matches_scalar_decoded_ir3(self):
    code, program = std_loop_program(actual_pcs=True)
    _data, check_range, bounds, initial = self.std_loop_state()
    expected = {key: values.copy() for key, values in initial.items()}
    execute_ir3(code, expected, start_pc=6, check_range=check_range, trace={})

    actual = {key: values.copy() for key, values in initial.items()}
    result = IR3UOpRunner(min_instructions=1).try_run_loop(program, 6, actual, [True],
                                                            check_range=check_range, memory_bounds=bounds)
    self.assertIsNotNone(result)
    assert result is not None
    self.assertEqual(result[0], 23)
    self.assertEqual(actual, expected)

  def test_native_header_guarded_loop_matches_scalar_control_flow(self):
    from test.mockgpu.qcom.decoder import IR3Instruction
    r0, r1, pred = ('r', 0, 0), ('r', 1, 0), ('p', 62, 0)
    program = (
      IR3Instruction('br', None, (pred,), False, 0, branch_offset=5, invert=True),
      IR3Instruction('add.u', r0, (r0, 1), False, 0),
      IR3Instruction('sub.u', r1, (r1, 1), False, 0),
      IR3Instruction('cmps.u', pred, (r1, 0), False, 0, condition=5),
      IR3Instruction('jump', None, (), False, 0, branch_offset=-4),
      *ir3_program(*END.words),
    )
    regs = {r0: [10], r1: [3], pred: [1]}
    runner = IR3UOpRunner(min_instructions=1)
    self.assertTrue(runner.has_loop(program))
    self.assertEqual(runner.try_run_loop(program, 0, regs, [True], max_steps=100), (5, 16))
    self.assertEqual((regs[r0], regs[r1], regs[pred], runner.stats.iterations), ([13], [0], [0], 3))

  def test_native_mesa_natural_loop_uses_one_native_call(self):
    _code, program = std_loop_program(actual_pcs=True)
    _data, check_range, bounds, regs = self.std_loop_state()
    runner = IR3UOpRunner(min_instructions=1)
    runner.stats.reset()
    result = runner.try_run_loop(program, 6, regs, [True], check_range=check_range, memory_bounds=bounds)
    self.assertIsNotNone(result)
    assert result is not None
    self.assertEqual(result[0], 23)
    self.assertEqual((runner.stats.attempts, runner.stats.runs, runner.stats.native_calls), (1, 1, 1))
    self.assertEqual((runner.stats.iterations, runner.stats.load_checks, runner.stats.load_rejections), (65, 65, 0))
    self.assertLess(runner.stats.native_calls, runner.stats.iterations)

  def test_invalid_memory_natural_loop_keeps_registers_transactional(self):
    _code, program = std_loop_program(actual_pcs=True)
    _data, check_range, bounds, regs = self.std_loop_state(invalid_address=True)
    before = {key: values.copy() for key, values in regs.items()}
    runner = IR3UOpRunner(min_instructions=1)
    with self.assertRaisesRegex(RuntimeError, r'IR3 ldg memory fault at PC 18'):
      runner.try_run_loop(program, 6, regs, [True], check_range=check_range, memory_bounds=bounds)
    self.assertEqual(regs, before)
    self.assertEqual((runner.stats.runs, runner.stats.native_calls, runner.stats.load_rejections), (0, 1, 1))

  def test_nonzero_load_offset_fault_reports_scalar_base_address(self):
    _code, decoded = std_loop_program(actual_pcs=True)
    program = list(decoded)
    address_reg, _offset, count = program[18].srcs
    program[18] = replace(program[18], srcs=(address_reg, 4, count))
    data, check_range, bounds, regs = self.std_loop_state(count=1)
    base = ctypes.addressof(data)
    before = {key: values.copy() for key, values in regs.items()}
    runner = IR3UOpRunner(min_instructions=1)
    with self.assertRaisesRegex(RuntimeError, rf'IR3 ldg memory fault at PC 18, lane 0, address={base:#x}'):
      runner.try_run_loop(tuple(program), 6, regs, [True], check_range=check_range, memory_bounds=bounds)
    self.assertEqual(regs, before)

  def test_incomplete_native_memory_bounds_fall_back_transactionally(self):
    _code, program = std_loop_program(actual_pcs=True)
    _data, check_range, _bounds, regs = self.std_loop_state()
    before = {key: values.copy() for key, values in regs.items()}
    runner = IR3UOpRunner(min_instructions=1)
    self.assertIsNone(runner.try_run_loop(program, 6, regs, [True], check_range=check_range, memory_bounds=()))
    self.assertEqual(regs, before)
    self.assertEqual((runner.stats.runs, runner.stats.native_calls, runner.stats.fallbacks, runner.stats.load_rejections), (0, 1, 1, 1))

  def test_partial_mask_and_non_candidate_natural_loops_fall_back(self):
    _data, check_range, bounds, initial = self.std_loop_state()
    _code, natural = std_loop_program(actual_pcs=True)
    partial = {key: values * 2 for key, values in initial.items()}
    partial_before = {key: values.copy() for key, values in partial.items()}
    runner = IR3UOpRunner(min_instructions=1)
    runner.stats.reset()
    self.assertIsNone(runner.try_run_loop(natural, 6, partial, [True, False], check_range=check_range, memory_bounds=bounds))
    self.assertEqual(partial, partial_before)

    non_candidate = ir3_program(*BACKWARD_BRANCH.words, *END.words)
    scalar = {('r', 0, 0): [3], ('p', 62, 0): [1]}
    scalar_before = {key: values.copy() for key, values in scalar.items()}
    self.assertIsNone(runner.try_run_loop(non_candidate, 0, scalar, [True]))
    self.assertEqual(scalar, scalar_before)
    self.assertEqual((runner.stats.attempts, runner.stats.runs, runner.stats.native_calls, runner.stats.fallbacks), (2, 0, 0, 2))

  def test_native_loop_fuel_exhaustion_falls_back_without_mutation(self):
    _code, program = std_loop_program(actual_pcs=True)
    _data, check_range, bounds, regs = self.std_loop_state(count=3)
    before = {key: values.copy() for key, values in regs.items()}
    runner = IR3UOpRunner(min_instructions=1)
    with self.assertRaisesRegex(IR3UOpLoopTimeout, r'fuel at PC 6'):
      runner.try_run_loop(program, 6, regs, [True], check_range=check_range, memory_bounds=bounds, max_steps=49)
    self.assertEqual(regs, before)
    self.assertEqual((runner.stats.runs, runner.stats.native_calls, runner.stats.fallbacks), (0, 1, 1))

  def test_executor_disables_timed_out_native_loop_before_scalar_replay(self):
    import test.mockgpu.qcom.executor as executor
    code, _program = std_loop_program(actual_pcs=True)
    _data, check_range, bounds, initial = self.std_loop_state(count=3)
    expected = {key: values.copy() for key, values in initial.items()}
    execute_ir3(code, expected, start_pc=6, check_range=check_range, trace={})

    class FuelLimitedRunner:
      def __init__(self):
        self.runner, self.loop_start_calls = IR3UOpRunner(min_instructions=1), 0

      def try_run_loop(self, program, start_pc, regs, exec_mask, **kwargs):
        if start_pc == 6:
          self.loop_start_calls += 1
          kwargs['max_steps'] = 49
        return self.runner.try_run_loop(program, start_pc, regs, exec_mask, **kwargs)

      def try_run(self, *args, **kwargs): return self.runner.try_run(*args, **kwargs)

    limited, original = FuelLimitedRunner(), executor._UOP_RUNNER
    actual = {key: values.copy() for key, values in initial.items()}
    setattr(executor, '_UOP_RUNNER', limited)
    try: execute_ir3(code, actual, start_pc=6, check_range=check_range, memory_bounds=bounds)
    finally: setattr(executor, '_UOP_RUNNER', original)
    self.assertEqual(actual, expected)
    self.assertEqual((limited.loop_start_calls, limited.runner.stats.native_calls), (1, 1))

  def test_adjacent_native_memory_bounds_do_not_merge(self):
    _code, program = std_loop_program(actual_pcs=True)
    data = (ctypes.c_ubyte * 8)(*range(8))
    start, end = ctypes.addressof(data), ctypes.addressof(data) + ctypes.sizeof(data)

    def check_range(address, size):
      if not (start <= address and address + size <= end): raise IndexError('out of bounds')

    regs = {('r', 0, 3): [0], ('r', 0, 2): [0], ('r', 1, 0): [0],
            ('r', 3, 0): [(start + 2) & 0xffffffff], ('r', 3, 1): [(start + 2) >> 32],
            ('c', 0, 2): [start & 0xffffffff], ('c', 0, 3): [start >> 32], ('c', 4, 0): [1]}
    before = {key: values.copy() for key, values in regs.items()}
    runner = IR3UOpRunner(min_instructions=1)
    self.assertIsNone(runner.try_run_loop(program, 6, regs, [True], check_range=check_range,
                                           memory_bounds=((start, start + 2), (start + 2, end))))
    self.assertEqual(regs, before)
    self.assertEqual((runner.stats.runs, runner.stats.native_calls, runner.stats.fallbacks, runner.stats.load_rejections), (0, 1, 1, 1))
