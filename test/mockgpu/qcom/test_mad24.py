"""Signed 24-bit multiply-add emitted by QCOMCL image address arithmetic."""
import struct
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from test.mockgpu.qcom import emu
from test.mockgpu.qcom.test_control import state
from test.mockgpu.qcom.test_qcomcl import requires_qcomcl, source_kernel


def mad24_word(destination=3):
  # ir3-cat3.xml: opcode5, full sources r0.x/r0.y/r0.z, destination r0.w.
  return (3 << 61) | (5 << 55) | (destination << 32) | (1 << 47) | (2 << 16)


def test_signed_mad24_uses_low_signed_24_bits_and_full_accumulator():
  left = [0x007fffff, 0x00800000, 0xfffffff9, 0xaa800001]
  right = [3, -3, -17, 0x00ffffff]
  accumulator = [0x7fffffff, 0x81234567, 0xdeadbeef, 0x89abcdef]
  group = state()
  for index, values in enumerate((left, right, accumulator)):
    group.registers[0][index] = np.array([value & 0xffffffff for value in values], dtype=np.uint32)
  group.run(emu.decode(struct.pack('<2Q', mad24_word(), 6 << 55)))
  expected = []
  for a, b, c in zip(left, right, accumulator):
    a, b = ((value & 0xffffff) - (0x1000000 if value & 0x800000 else 0) for value in (a, b))
    expected.append((a * b + c) & 0xffffffff)
  np.testing.assert_array_equal(group.registers[0][3], expected)


def test_signed_mad24_rejects_half_destination():
  # Mesa ir3_cf.c and ir3_validate.c prohibit mad.x24 with 16-bit in/out.
  with pytest.raises(RuntimeError, match='full registers'):
    emu.decode(struct.pack('<Q', mad24_word() | (1 << 46)))


@requires_qcomcl
def test_compiler_generated_signed_mad24_with_negative_inputs_and_wrapping_sum(monkeypatch):
  source = '''__kernel void signed_mad(__global int *out, __global const int *left,
                                      __global const int *right, __global const int *accumulator) {
    uint i = get_global_id(0);
    out[i] = mad24(left[i], right[i], accumulator[i]);
  }'''
  left = np.array([8388607, -8388608, -7, 1234567, -8000000], np.int32)
  right = np.array([-13, 3, -17, 2000000, -8000000], np.int32)
  accumulator = np.array([2147483647, -2147483648, -123456789, 987654321, 19], np.int32)
  executed: set[str] = set()
  original = emu.execute

  def record(image, *args, **kwargs):
    result = original(image, *args, **kwargs)
    executed.update(instruction.op for instruction in emu.decode(image))
    return result

  monkeypatch.setattr(emu, 'execute', record)
  actual = source_kernel(source, 'signed_mad', Tensor.empty(5, dtype=dtypes.int32), (1, 1, 1), (5, 1, 1),
                         inputs=(Tensor(left), Tensor(right), Tensor(accumulator))).numpy()
  expected = (left.astype(np.int64) * right + accumulator).astype(np.int32)
  np.testing.assert_array_equal(actual, expected)
  assert 'mad.s24' in executed, executed
