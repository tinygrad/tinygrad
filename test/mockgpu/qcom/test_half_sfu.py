"""Half special functions share the lane engine without losing operand precision."""
import struct
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from test.mockgpu.qcom.emu import Workgroup, decode
from test.mockgpu.qcom.test_emu import Memory


@pytest.mark.parametrize('opcode,values,expected', [
  (9, [0.25, 1, 4, 16], [2, 1, 0.5, 0.25]),
  (10, [0.25, 1, 2, 16], [-2, 0, 1, 4]),
  (11, [-2, 0, 1, 4], [0.25, 1, 2, 16]),
])
@pytest.mark.parametrize('full_destination', [False, True])
def test_encoded_half_special_functions_use_half_sources(opcode, values, expected, full_destination):
  # Mesa 25.2.7 ir3-cat4.xml assigns HRSQ/HLOG2/HEXP2 opcodes 9/10/11.
  # FULL=0 selects half sources; DST_CONV widens the already-half result.
  group = Workgroup(bytes(16), (4, 1, 1), (0, 0, 0), 0xfc, 0xfc, 0xfc, Memory({}), 0, 0)
  group.registers[True][0] = np.array(values, dtype=np.float16).view(np.uint16)
  word = (4 << 61) | (opcode << 53) | (int(full_destination) << 46) | (4 << 32)
  group.run(decode(struct.pack('<2Q', word, 6 << 55)))
  result = group.registers[not full_destination][4].view(np.float32 if full_destination else np.float16)
  np.testing.assert_array_equal(result, expected)


def test_compiler_generated_half_exponential_preserves_fractional_values():
  # This ordinary Tensor path previously failed admission at Mesa's HEXP2.
  values = np.array([-2, -0.5, 0, 0.5, 1, 2, 4], dtype=np.float16)
  actual = Tensor(values, dtype=dtypes.half).exp2().numpy()
  expected = np.exp2(values.astype(np.float32)).astype(np.float16)
  np.testing.assert_allclose(actual, expected, rtol=0.003, atol=0.001)
