"""Address-register indexing preserves per-lane data and checks every selected span."""
import struct
import numpy as np
import pytest
from test.mockgpu.qcom.emu import Workgroup, decode, validate_constant_footprint
from test.mockgpu.qcom.test_emu import Memory


def group_with_addresses(addresses, constants=b'', *, constant_demotion=True):
  group = Workgroup(constants, (len(addresses), 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc,
                    Memory({}), 0, 0, constant_demotion=constant_demotion)
  group.registers[True][0] = np.array(addresses, dtype=np.int16).view(np.uint16)
  return group


def run_words(group, *words):
  group.run(decode(struct.pack('<' + 'Q' * (len(words) + 1), *words, 6 << 55)))


# Mesa25.2.7 cat1 MOVA override: S16 source/destination, implicit a0.x (244).
MOVA = 0x201100f400000000


@pytest.mark.parametrize('left,right', [(8, 9), (16, 13), (200, 201)])
def test_swizzle_source_bits_are_not_mistaken_for_relative_move_flags(left, right):
  # Cat1 opcode2 uses source-register bits where opcode0 has relative flags.
  group = group_with_addresses([0, 0])
  group.registers[False][left], group.registers[False][right] = [11, 22], [33, 44]
  swap = (1 << 61) | (2 << 57) | (3 << 50) | (3 << 46) | (right << 32) | (left << 16) | (right << 8) | left
  run_words(group, swap)
  np.testing.assert_array_equal(group.registers[False][left], [33, 44])
  np.testing.assert_array_equal(group.registers[False][right], [11, 22])


def test_immediate_mova_hazard_marker_is_not_a_repeated_source():
  group = group_with_addresses([0, 0])
  group.registers[False][23] = [11, 22]
  immediate_mova = MOVA | (1 << 54) | (1 << 43) | 23
  relative_read = (1 << 61) | (3 << 50) | (3 << 46) | (8 << 32) | (1 << 11)
  run_words(group, immediate_mova, relative_read)
  np.testing.assert_array_equal(group.registers[False][8], [11, 22])


def test_mova_and_relative_register_read_select_each_lanes_value():
  group = group_with_addresses([20, 23, 21, 22])
  group.registers[False][20:24] = np.array([[11], [22], [33], [44]], dtype=np.uint32)
  # MOV.U32 r2.x, r<a0.x>: bit11 selects relative source, bit10=0 selects GPR.
  relative_read = (1 << 61) | (3 << 50) | (3 << 46) | (8 << 32) | (1 << 11)
  run_words(group, MOVA, relative_read)
  np.testing.assert_array_equal(group.registers[False][8], [11, 44, 22, 33])


def test_relative_register_destination_keeps_lanes_and_source_values_separate():
  group = group_with_addresses([20, 23, 21, 22])
  group.registers[False][0] = [101, 202, 303, 404]
  # Cat1 DST_REL=bit49; destination offset is unsigned in bits39:32.
  relative_write = (1 << 61) | (3 << 50) | (3 << 46) | (1 << 49)
  run_words(group, MOVA, relative_write)
  np.testing.assert_array_equal(group.registers[False][20:24],
                                [[101, 0, 0, 0], [0, 0, 303, 0], [0, 0, 0, 404], [0, 202, 0, 0]])


@pytest.mark.parametrize('offset,typ', [(244, 4), (245, 2)])
def test_relative_destination_offset_survives_the_mova_display_alias(offset, typ):
  # Mesa displays these encodings as MOVA/MOVA1 and omits the relative OFFSET.
  # The unsigned destination offset still combines with the signed a0 value.
  group = group_with_addresses([-offset, 1 - offset])
  group.registers[True][2] = [101, 202]
  write = (1 << 61) | (typ << 50) | (typ << 46) | (1 << 49) | (offset << 32) | 2
  run_words(group, MOVA, write)
  assert group.registers[True][0, 0] == 101
  assert group.registers[True][1, 1] == 202
  np.testing.assert_array_equal(group.registers[True][2], [101, 202])


def test_relative_source_offsets_are_signed():
  group = group_with_addresses([21, 24, 22, 23])
  group.registers[False][20:24] = np.array([[11], [22], [33], [44]], dtype=np.uint32)
  relative_read = (1 << 61) | (3 << 50) | (3 << 46) | (8 << 32) | (1 << 11) | 0x3ff
  run_words(group, MOVA, relative_read)
  np.testing.assert_array_equal(group.registers[False][8], [11, 44, 22, 33])


def test_relative_constants_use_current_address_for_each_lane():
  constants = np.array([0.25, 1.5, 2.75, 3.125], dtype=np.float32).tobytes()
  group = group_with_addresses([3, 0, 2, 1], constants)
  # Cat1 bit10 selects constants rather than the GPR bank.
  relative_read = (1 << 61) | (1 << 50) | (1 << 46) | (8 << 32) | (1 << 11) | (1 << 10)
  instructions = decode(struct.pack('<3Q', MOVA, relative_read, 6 << 55))
  validate_constant_footprint(instructions, 4)
  group.run(instructions)
  np.testing.assert_array_equal(group.registers[False][8].view(np.float32), [3.125, 0.25, 2.75, 1.5])


def test_relative_packed_half_constants_use_half_slot_units():
  group = group_with_addresses([0, 1], struct.pack('<I', 0x40b1bd00), constant_demotion=False)
  convert = (1 << 61) | (1 << 46) | (8 << 32) | (1 << 11) | (1 << 10)
  run_words(group, MOVA, convert)
  np.testing.assert_array_equal(group.registers[False][8].view(np.float32), [-1.25, 2.345703125])


def test_relative_cat2_constant_source_preserves_other_operand():
  group = group_with_addresses([3, 0, 2, 1], struct.pack('<4I', 10, 20, 30, 40))
  group.registers[False][1] = [3, 4, 5, 6]
  # ADD.U r2.x, c<a0.x>, r0.y: cat2 relative multisrc shares the signed10-bit offset.
  add = (2 << 61) | (16 << 53) | (1 << 52) | (8 << 32) | (1 << 16) | 0xc00
  run_words(group, MOVA, add)
  np.testing.assert_array_equal(group.registers[False][8], [43, 14, 35, 26])


@pytest.mark.parametrize('address', [-1, 244, 32767])
def test_bad_relative_register_index_rejects_before_any_destination_lane_changes(address):
  group = group_with_addresses([20, address])
  group.registers[False][8] = 77
  relative_read = (1 << 61) | (3 << 50) | (3 << 46) | (8 << 32) | (1 << 11)
  with pytest.raises(RuntimeError, match='relative register'):
    run_words(group, MOVA, relative_read)
  np.testing.assert_array_equal(group.registers[False][8], [77, 77])


def test_bad_relative_destination_does_not_partially_scatter():
  group = group_with_addresses([20, 244])
  group.registers[False][0] = [101, 202]
  relative_write = (1 << 61) | (3 << 50) | (3 << 46) | (1 << 49)
  with pytest.raises(RuntimeError, match='relative register'):
    run_words(group, MOVA, relative_write)
  np.testing.assert_array_equal(group.registers[False][20], [0, 0])


@pytest.mark.parametrize('address', [-1, 4])
def test_relative_constant_outside_upload_is_rejected_without_negative_wrap(address):
  group = group_with_addresses([0, address], struct.pack('<4I', 10, 20, 30, 40))
  group.registers[False][8] = 77
  relative_read = (1 << 61) | (3 << 50) | (3 << 46) | (8 << 32) | 0xc00
  with pytest.raises(RuntimeError, match='constant register'):
    run_words(group, MOVA, relative_read)
  np.testing.assert_array_equal(group.registers[False][8], [77, 77])
