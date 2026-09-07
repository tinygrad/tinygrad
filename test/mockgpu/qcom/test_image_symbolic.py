"""Scalar arguments follow all signature views of each logical buffer."""
import pytest
from test.mockgpu.qcom.test_image_integration import run_image_program


@pytest.mark.parametrize('half_storage', [False, True])
def test_symbolic_image_mutation_binds_integer_bounds_through_jit_replays(half_storage):
  run_image_program(f"""
from collections import Counter
from tinygrad import TinyJit, Variable
from tinygrad.runtime.ops_qcom import QCOMComputeQueue
observed_mixed_signature = []
original_kernargs = QCOMComputeQueue.kernargs
def record_kernargs(self, call, program, data):
  buffer_count = len(program.arg.globals)
  counts = Counter(slot for _, slot, _, _ in data.signature if slot < buffer_count)
  if any(count > 1 for count in counts.values()) and program.arg.vars:
    scalar_types = [dtype for _, slot, dtype, _ in data.signature if slot >= buffer_count]
    assert len(scalar_types) == 1 and scalar_types[0].name == 'int'
    observed_mixed_signature.append(True)
  return original_kernargs(self, call, program, data)
QCOMComputeQueue.kernargs = record_kernargs

@TinyJit
def mutate(image, bound):
  masked = image[:, :bound].pad((None, (0, image.shape[1] - bound)), value=2)
  image.assign(image + masked).realize()
  return image

values = (np.arange(256, dtype=np.float32).reshape(4, 64) / 8 - 7).astype(np.float16 if {half_storage!r} else np.float32)
image = Tensor(values).contiguous().realize()
expected = values.copy()
for size in (3, 15, 32, 1, 63):
  actual = mutate(image, Variable('columns', 1, 63).bind(size)).numpy()
  increment = np.full_like(expected, 2)
  increment[:, :size] = expected[:, :size]
  expected += increment
  np.testing.assert_array_equal(actual, expected)
assert observed_mixed_signature
""", half_storage, required_ops=('ldib.b', 'stib.b'))
