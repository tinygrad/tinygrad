import struct
import unittest

from test.mockgpu.qcom.decoder import decode_ir3
from test.mockgpu.qcom.executor import execute_ir3
from test.mockgpu.qcom.uop_runner import IR3UOpRunner
from test.mockgpu.qcom.corpus import (BACKWARD_BRANCH, BITWISE_SHIFT_COMPARE, END, FLOAT_ALU_COMPARE, GLOBAL_LOAD, MADSH_MAGIC_DIVIDE,
  MADSH_REPEAT, MULL_REPEAT, REPEATED_ADD, SHRG_HALF_DEST, SIGNED_BYTE_TO_HALF)


def ir3_program(*instructions: int): return decode_ir3(struct.pack(f'<{len(instructions)}Q', *instructions))


class TestA630UOpRunner(unittest.TestCase):
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
        executor.decode_ir3 = decode
        try: executor.execute_ir3(b'masktest', expected, check_range=lambda a, s: None)
        finally: executor.decode_ir3 = real_decode
        expected.pop(('p', 62, 0))

        actual = {key: values.copy() for key, values in initial.items()}
        self.assertEqual(IR3UOpRunner(min_instructions=1).try_run(program, 0, actual, mask), 3)
        self.assertEqual(actual, expected)
