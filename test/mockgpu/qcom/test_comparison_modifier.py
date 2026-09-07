"""Floating comparison branches emitted by the ordinary QCOMCL compiler."""
import struct
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from test.mockgpu.qcom import emu
from test.mockgpu.qcom.test_qcomcl import requires_qcomcl, source_kernel


@requires_qcomcl
@pytest.mark.parametrize('comparison,compare', [('<', np.less), ('<=', np.less_equal), ('>', np.greater),
                                               ('>=', np.greater_equal), ('==', np.equal), ('!=', np.not_equal)])
@pytest.mark.parametrize('negate', [False, True])
def test_compiled_float_branches_include_nan_and_threshold_neighbors(comparison, compare, negate, monkeypatch):
  # Volatile stores keep the compiler's control-flow comparison observable.
  condition = f'fabs(input[i]) {comparison} 1.5f'
  if negate: condition = f'!({condition})'
  source = f'''__kernel void compare_values(__global volatile uint *out, __global const float *input) {{
    uint i = get_global_id(0);
    if ({condition}) {{
      for (uint k = 0; k < 3; k++) out[i * 3 + k] = 17 + k;
    }} else {{
      for (uint k = 0; k < 3; k++) out[i * 3 + k] = 81 + k;
    }}
  }}'''
  below = np.nextafter(np.float32(1.5), np.float32(0))
  above = np.nextafter(np.float32(1.5), np.float32(2))
  values = np.array([-np.inf, -above, -1.5, -below, -0.0, 0.0, below, 1.5, above, np.inf, np.nan], dtype=np.float32)
  comparisons: list[tuple[int, int]] = []
  original = emu.execute

  def record(image, *args, **kwargs):
    result = original(image, *args, **kwargs)
    comparisons.extend((ins.fields['COND'], ins.fields.get('SAT', 0)) for ins in emu.decode(image) if ins.op == 'cmps.f')
    return result

  monkeypatch.setattr(emu, 'execute', record)
  actual = source_kernel(source, 'compare_values', Tensor.empty(len(values), 3, dtype=dtypes.uint32),
                         (1, 1, 1), (len(values), 1, 1), inputs=(Tensor(values),)).numpy()
  expected_condition = compare(np.abs(values), np.float32(1.5))
  if negate: expected_condition = ~expected_condition
  expected = np.where(expected_condition[:, None], np.array([17, 18, 19]), np.array([81, 82, 83]))
  np.testing.assert_array_equal(actual, expected)
  assert comparisons
  if comparison in ('<=', '>='):
    assert any(saturate for _, saturate in comparisons), comparisons


@pytest.mark.parametrize('change', [1 << 52, 248 << 32, 1 << 48, 2 << 53])
def test_other_comparison_modifier_forms_remain_unsupported(change):
  # The observed F32 LE predicate word cannot admit half sources, ordinary
  # destinations, an unqualified condition, or a different comparison opcode.
  word = 0x40b104f8106c8005 ^ change
  with pytest.raises(RuntimeError, match='unsupported comparison modifier|unsupported instruction'):
    emu.decode(struct.pack('<Q', word))
