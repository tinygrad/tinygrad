"""A630-1 regressions: constant-slot storage and typed half consumption.

The Tensor cases use ordinary allocations through HCQ2. Encoded-instruction
cases use one Workgroup and inert memory, including the mock's admitted cat4
constant form that the pinned compiler does not emit.
"""
import struct
import numpy as np
import pytest
from tinygrad import Tensor
from test.mockgpu.qcom.emu import Workgroup, decode
from test.mockgpu.qcom.test_emu import Memory


VALUES = np.array([-10.25, -3.5, -1.25, -0.125, 0, 0.125, 0.5, 1.25, 3.5, 10.25], dtype=np.float16)


def workgroup(constants, count=1):
  return Workgroup(struct.pack(f'<{len(constants)}I', *constants), (count, 1, 1), (0, 0, 0),
                   0xfc, 0xfc, 0xfc, Memory({}), 0, 0)


def run_word(group, word):
  instructions = decode(struct.pack('<2Q', word, 6 << 55))
  group.run(instructions)
  return instructions[0]


@pytest.mark.parametrize('operation', ['add', 'multiply', 'multiply_add'])
def test_non_flut_half_constants_through_hcq2(operation):
  # Retain the reviewer's three failing ordinary expressions and expectations.
  a = Tensor(VALUES)
  if operation == 'add':
    result, expected = a + 2.345703125, VALUES + np.float16(2.345703125)
  elif operation == 'multiply':
    result, expected = a * 1.234375, VALUES * np.float16(1.234375)
  else:
    result = a * 1.234375 + 2.345703125
    expected = VALUES * np.float16(1.234375) + np.float16(2.345703125)
  np.testing.assert_array_equal(result.numpy(), expected)


def test_compiler_emitted_mad_f16_constant_operands_in_workgroup():
  # Exact pinned Mesa word and F32 constant slots from the original red trace.
  constants = [0] * 18
  constants[16:18] = [0x40162000, 0x3f9e0000]
  group = workgroup(constants, len(VALUES))
  group.registers[1][1] = VALUES.view(np.uint16)
  instruction = run_word(group, 0x7300880110109011)
  assert instruction.op == 'mad.f16'
  assert instruction.operands['SRC1'].kind == instruction.operands['SRC3'].kind == 'constant'
  expected = VALUES * np.float16(1.234375) + np.float16(2.345703125)
  np.testing.assert_array_equal(group.registers[1][1].view(np.float16), expected)


@pytest.mark.parametrize('value,expected_bits', [
  (1.0006, 0x3c00), (-1.0006, 0xbc00),
  (2051, 0x6801), (-2051, 0xe801),
  (65536, 0x7bff), (-65536, 0xfbff),
  (2**-25, 0x0000), (-2**-25, 0x8000),
  (2**-24, 0x0001), (-2**-24, 0x8001),
  (0.0, 0x0000), (-0.0, 0x8000),
])
def test_implicit_constant_demotion_uses_default_conversion_rounding(value, expected_bits):
  # Model constant demotion as the narrowing COV it replaces: ROUND_ZERO.
  # These nonrepresentable values specify the mock rule; no hardware oracle
  # is claimed for the tie, subnormal, overflow or signed-zero boundaries.
  bits = struct.unpack('<I', struct.pack('<f', value))[0]
  group = workgroup([bits])
  word = (2 << 61) | (6 << 53) | (2 << 32) | 0x1000  # absneg.f hr0.z, hc0.x
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2], [expected_bits])


@pytest.mark.parametrize('dest_half', [False, True])
def test_constant_demotion_precedes_arithmetic_and_uses_source_precision(dest_half):
  group = workgroup([0x3f8013a9])  # F32 1.0006, demoted to half 1.0 before ADD.
  group.registers[1][1] = np.array([-1], np.float16).view(np.uint16)
  word = (2 << 61) | (int(not dest_half) << 46) | (2 << 32) | (1 << 16) | 0x1000
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[dest_half][2], [0])


def test_full_float_constant_keeps_its_full_source_precision():
  group = workgroup([0x3f8013a9])
  group.registers[0][1] = np.array([-1], np.float32).view(np.uint32)
  word = (2 << 61) | (1 << 52) | (2 << 32) | (1 << 16) | 0x1000
  run_word(group, word)
  expected = np.float32(1.0006) - np.float32(1)
  np.testing.assert_array_equal(group.registers[0][2].view(np.float32), [expected])


def test_half_float_comparison_demotes_before_writing_boolean_result():
  group = workgroup([0x3f8013a9])
  group.registers[1][1] = 0x3c00
  # cmps.f.eq: 1.0006 demotes to 1.0, so the half operands compare equal.
  word = (2 << 61) | (5 << 53) | (4 << 48) | (2 << 32) | (1 << 16) | 0x1000
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2], [1])


@pytest.mark.parametrize('modifier,expected', [(0, -2.345703125), (1, 2.345703125), (2, 2.345703125), (3, -2.345703125)])
def test_constant_float_modifiers_follow_numeric_demotion(modifier, expected):
  group = workgroup([0xc0162000])
  word = (2 << 61) | (6 << 53) | (2 << 32) | (modifier << 14) | 0x1000
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2].view(np.float16), [expected])


def test_repeated_half_constant_sources_advance_by_full_slots():
  constants = np.array([2.345703125, 1.234375, -3.5], np.float32).view(np.uint32)
  group = workgroup(constants)
  word = (2 << 61) | (2 << 40) | (1 << 43) | (2 << 32) | (1 << 16) | 0x1000
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2:5, 0].view(np.float16), constants.view(np.float32))


@pytest.mark.parametrize('half', [False, True])
def test_unsigned_constant_comparison_truncates_only_at_half_width(half):
  group = workgroup([0x12340002])
  group.registers[half][1] = 3
  # cmps.u.lt destination, constant, register. Truncation must precede compare.
  word = (2 << 61) | (20 << 53) | (int(not half) << 52) | (2 << 32) | (1 << 16) | 0x1000
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[half][2], [int(half)])


def test_signed_half_constant_uses_integer_low_bits():
  group = workgroup([0x12348001])
  word = (2 << 61) | (17 << 53) | (1 << 46) | (2 << 32) | (1 << 16) | 0x1000
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[0][2], [0xffff8001])


def test_half_gpr_float_bits_are_not_constant_slots():
  group = workgroup([0x40162000])
  group.registers[1][0] = 0x3c01
  word = (2 << 61) | (6 << 53) | (2 << 32)
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2], [0x3c01])


def test_half_float_table_immediate_keeps_its_table_index():
  group = workgroup([0x40162000])
  word = (2 << 61) | (2 << 32) | (1 << 16) | (5 << 11) | (1 << 10) | 5
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2], [0x4248])  # half pi


@pytest.mark.parametrize('rounding,expected', [(0, 0x3c00), (1, 0x3c01)])
def test_explicit_full_float_constant_conversion_keeps_output_rounding(rounding, expected):
  group = workgroup([0x3f8013a9])
  word = (1 << 61) | (rounding << 55) | (1 << 53) | (1 << 50) | (2 << 32)
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2], [expected])


@pytest.mark.parametrize('src_type,constant,expected', [(0, 0x3f8013a9, 1.0), (2, 0xdead3c01, 15361.0), (4, 0xbeef8001, -32767.0)])
def test_explicit_half_constant_conversion_selects_its_source_type(src_type, constant, expected):
  group = workgroup([constant])
  word = (1 << 61) | (1 << 53) | (src_type << 50) | (1 << 46) | (2 << 32)
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[0][2].view(np.float32), [expected])


@pytest.mark.parametrize('src_type,constant,expected', [(0, 0x3f8013a9, 0x3c00), (2, 0xdead3c01, 0x3c01), (4, 0xbeef8001, 0x8001)])
def test_same_type_half_constant_move_consumes_the_declared_type(src_type, constant, expected):
  group = workgroup([constant])
  word = (1 << 61) | (1 << 53) | (src_type << 50) | (src_type << 46) | (2 << 32)
  run_word(group, word)
  np.testing.assert_array_equal(group.registers[1][2], [expected])


def test_admitted_cat4_constant_uses_the_same_typed_demotion_rule():
  # Mesa 25.2.7 ir3.c rejects const/imm for compiler cat4 sources. The emulator
  # already accepts this multisrc encoding, so cover its chosen mock semantics.
  group = workgroup([0x3f8013a9])
  word = (4 << 61) | (2 << 32) | 0x1000  # rcp hr0.z, hc0.x
  instruction = run_word(group, word)
  assert instruction.op == 'rcp'
  assert instruction.operands['SRC'].kind == 'constant'
  np.testing.assert_array_equal(group.registers[1][2].view(np.float16), [1.0])
