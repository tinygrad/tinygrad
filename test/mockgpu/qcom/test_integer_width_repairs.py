"""A630-5/6: source modifiers, arithmetic results and destination widths.

These finite-data checks run decoded instructions in inert Workgroups. Scalar
Python arithmetic supplies the expected results independently of NumPy and the
interpreter's conversion helpers; explicit NOT instructions supply a second
check of folded modifiers. No native guest memory or hardware is involved.
"""
import itertools
import operator
import struct
import numpy as np
import pytest
from test.mockgpu.qcom.emu import Instruction, Operand, Workgroup, decode


def machine(a, b, half, constants=()):
  group = Workgroup(struct.pack(f'<{len(constants)}I', *constants), (len(a), 1, 1), (0, 0, 0),
                    0xfc, 0xfc, 0xfc, object(), 0, 0)
  modulus = 2**(16 if half else 32)
  group.registers[half][0] = [x % modulus for x in a]
  group.registers[half][1] = [x % modulus for x in b]
  return group


def cat2(opcode, half, dest_half, src1=0, src2=1, dst=8, modifiers=(0, 0)):
  # Mesa cat2 shares source precision, with a separate destination conversion.
  return ((2 << 61) | (opcode << 53) | (int(not half) << 52) | (int(half != dest_half) << 46) |
          (dst << 32) | (src2 << 16) | src1 | (modifiers[0] << 14) | (modifiers[1] << 30))


def run(group, *words):
  instructions = decode(struct.pack(f'<{len(words) + 1}Q', *words, 6 << 55))
  group.run(instructions)
  return instructions


def signed_integer(value, width):
  value %= 2**width
  return value if value < 2**(width - 1) else value - 2**width


@pytest.mark.parametrize('half', [False, True])
@pytest.mark.parametrize('dest_half', [False, True])
@pytest.mark.parametrize('opcode', [54, 55, 56])
@pytest.mark.parametrize('modifiers', [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize('constant_role', [None, 0, 1])
def test_folded_shift_modifiers_match_scalar_arithmetic_and_materialized_not(half, dest_half, opcode, modifiers, constant_role):
  # A late half mask cannot undo high complement bits shifted into the result.
  # Also protect signed ASHR and the low-five-bit shift-count contract, including
  # a complement on SRC2 and counts at/above each source precision.
  pairs = list(itertools.product([0, 1, 0x7fff, 0x8000, 0xffff, 0x12345678, 0x80000000, 0xffffffff],
                                 [0, 1, 2, 7, 15, 16, 31, 32, 33, 63, 0xffffffff]))
  a, b = ([pair[i] for pair in pairs] for i in (0, 1))
  constants = (0xabcd8000, 0xdead0021)
  if constant_role == 0:
    a = [constants[0]] * len(a)
  if constant_role == 1:
    b = [constants[1]] * len(b)
  sources = [0x1000 if constant_role == 0 else 0, 0x1001 if constant_role == 1 else 1]
  folded, expanded = (machine(a, b, half, constants) for _ in range(2))
  instruction = run(folded, cat2(opcode, half, dest_half, sources[0], sources[1], modifiers=modifiers))[0]
  assert instruction.op == {54: 'shl.b', 55: 'shr.b', 56: 'ashr.b'}[opcode]

  words = []
  for index, modifier in enumerate(modifiers):
    if modifier:
      words.append(cat2(30, half, half, sources[index], 0, dst=6 + index))
      sources[index] = 6 + index
  run(expanded, *words, cat2(opcode, half, dest_half, *sources))

  width = 16 if half else 32
  expected = []
  for left, right in zip(a, b):
    left, right = left % 2**width, right % 2**width
    if modifiers[0]:
      left = 2**width - 1 - left
    if modifiers[1]:
      right = 2**width - 1 - right
    if opcode == 56:
      left = signed_integer(left, width)
    scale = 2**(right % 32)
    result = left * scale if opcode == 54 else left // scale
    expected.append((result % 2**width) % 2**(16 if dest_half else 32))
  assert expanded.registers[dest_half][8].tolist() == expected
  assert folded.registers[dest_half][8].tolist() == expected


@pytest.mark.parametrize('dest_half', [False, True])
def test_repeated_half_constant_complements_are_narrowed_before_each_shift(dest_half):
  constants = (0xabcd0000, 0xdead7fff, 0xbeef8000, 0x1234ffff)
  group = machine([0], [1], True, constants)
  # REPEAT=3; SRC1_R advances by complete constant slots. SRC2 stays fixed.
  word = cat2(55, True, dest_half, 0x1000, modifiers=(1, 0)) | (3 << 40) | (1 << 43)
  run(group, word)
  assert group.registers[dest_half][8:12, 0].tolist() == [32767, 16384, 16383, 0]


MULTIPLY_SOURCES = [(None, 0)] + list(itertools.product([0, 1], [300, -300, 32767, -32768]))


@pytest.mark.parametrize('half', [False, True])
@pytest.mark.parametrize('dest_half', [False, True])
@pytest.mark.parametrize('constant_role,constant_value', MULTIPLY_SOURCES)
def test_mul_s24_keeps_the_signed_product_until_destination_conversion(half, dest_half, constant_role, constant_value):
  # Products cross 16-bit boundaries in both directions, but half sources fit
  # exactly in signed 16 bits. The oracle receives the original numeric values.
  values = [-32768, -32767, -301, -1, 0, 1, 301, 32767]
  pairs = list(itertools.product(values, repeat=2))
  a, b = ([pair[i] for pair in pairs] for i in (0, 1))
  if constant_role == 0:
    a = [constant_value] * len(a)
  if constant_role == 1:
    b = [constant_value] * len(b)
  # Half constants ignore the upper 16 bits. Full MUL_S24 ignores the upper
  # eight bits while sign-interpreting its selected low-24-bit integer.
  slot = (0xbeef0000 | (constant_value % 2**16)) if half else (0xa5000000 | (constant_value % 2**24))
  group = machine(a, b, half, (slot,))
  sources = [0x1000 if constant_role == 0 else 0, 0x1000 if constant_role == 1 else 1]
  assert run(group, cat2(49, half, dest_half, *sources))[0].op == 'mul.s24'
  expected = [(left * right) % 2**(16 if dest_half else 32) for left, right in zip(a, b)]
  assert group.registers[dest_half][8].tolist() == expected


@pytest.mark.parametrize('dest_half', [False, True])
def test_full_mul_s24_wraps_its_product_at_32_bits(dest_half):
  # Hand-calculated products include selected 24-bit extrema and overflow.
  group = machine([0x007fffff, 0x00800000, 0x01ffffff, 0xffffffff],
                  [0x007fffff, 0x00800000, 2, 0x00800000], False)
  run(group, cat2(49, False, dest_half))
  expected = [0xff000001, 0x00000000, 0xfffffffe, 0x00800000]
  assert group.registers[dest_half][8].tolist() == [value % 2**(16 if dest_half else 32) for value in expected]


NARROW_OPERATIONS = [(16, 'add.u', operator.add), (17, 'add.s', operator.add), (18, 'sub.u', operator.sub),
                     (19, 'sub.s', operator.sub), (22, 'min.u', min), (23, 'min.s', min),
                     (24, 'max.u', max), (25, 'max.s', max)]


@pytest.mark.parametrize('opcode,name,operation', NARROW_OPERATIONS)
@pytest.mark.parametrize('dest_half', [False, True])
@pytest.mark.parametrize('constant_role', [None, 0, 1])
def test_ordinary_half_integer_results_still_complete_before_widening(opcode, name, operation, dest_half, constant_role):
  # The MUL_S24 exception must not turn a half ADD/SUB into full arithmetic or
  # change signed/unsigned constant comparisons and their destination extension.
  a, b = [65535, 32767, 32768, 0, 1, 0x1234], [1, 1, 65535, 65535, 32768, 0x4321]
  slot = 0xbeefffff
  if constant_role == 0:
    a = [65535] * len(a)
  if constant_role == 1:
    b = [65535] * len(b)
  group = machine(a, b, True, (slot,))
  sources = [0x1000 if constant_role == 0 else 0, 0x1000 if constant_role == 1 else 1]
  assert run(group, cat2(opcode, True, dest_half, *sources))[0].op == name
  expected = []
  for left, right in zip(a, b):
    if name.endswith('.s'):
      left, right = signed_integer(left, 16), signed_integer(right, 16)
    result = operation(left, right) % 2**16
    if name.endswith('.s'):
      result = signed_integer(result, 16)
    expected.append(result % 2**(16 if dest_half else 32))
  assert group.registers[dest_half][8].tolist() == expected


def test_half_multiply_then_widen_remains_distinct_from_a_full_product():
  # An explicitly half destination loses bits; a later COV must not recover
  # them. A630-6 preserves the wider product only when that destination allows it.
  half_product, full_product = (machine([300, -300], [300, 300], True) for _ in range(2))
  run(half_product, cat2(49, True, True))
  run(full_product, cat2(49, True, False))
  conversion = Instruction('mov', {'SRC_TYPE': 4, 'DST_TYPE': 5},
                           {'DST': Operand('register', 9), 'SRC': Operand('register', 8, True)}, 1 << 61)
  half_product.alu(conversion, np.arange(2), 0)
  assert half_product.registers[0][9].tolist() == [24464, 0xffffa070]
  assert full_product.registers[0][8].tolist() == [90000, 0xfffea070]
