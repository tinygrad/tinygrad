"""Promoting a memory view to an image must preserve the value dtype of its load."""
import pytest
from tinygrad import dtypes
from tinygrad.codegen.late.coalesce import pm_simplify_add_image
from tinygrad.helpers import Context, Target, is_image_shape
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from test.mockgpu.qcom.test_image_integration import run_image_program


@pytest.mark.parametrize('dtype', [dtypes.half, dtypes.float])
def test_image_promotion_preserves_load_dtype_beside_scalar_buffer_load(dtype):
  # Coalescing can promote four adjacent components while a masked scalar read
  # of the same allocation remains a buffer access. Their arithmetic must stay
  # well typed; changing only the vector load's dtype leaves a mixed-type ADD.
  buffer = UOp.param(0, dtype, 256)
  vector = UOp(Ops.SHRINK, src=(buffer, UOp.const(0), UOp.const(4))).load()
  scalar = buffer.index(UOp.const(5)).load()
  value = vector.index(0) + scalar
  renderer = Renderer(Target(device='QCOM', arch='a630,IMAGE_PITCH_ALIGNMENT=64'))
  with Context(IMAGE=1):
    promoted = graph_rewrite(value, pm_simplify_add_image, ctx=({}, renderer), bottom_up=True)
  assert promoted.dtype == dtype
  assert all(source.dtype == dtype for source in promoted.src)
  params = [u for u in promoted.toposort() if u.op is Ops.PARAM]
  assert len(params) == 2 and {u.arg.slot for u in params} == {0}
  assert any(is_image_shape(u._shape) for u in params)
  assert any(not is_image_shape(u._shape) for u in params)


@pytest.mark.parametrize('half_input', [False, True])
def test_image_mutation_preserves_half_and_float_loads_in_mixed_views(half_input):
  run_image_program(f"""
from collections import Counter
from tinygrad.runtime.ops_qcom import QCOMComputeQueue
observed_mixed_views = []
original_kernargs = QCOMComputeQueue.kernargs
def record_kernargs(self, call, program, data):
  counts = Counter(slot for _, slot, _, _ in data.signature if slot < len(program.arg.globals))
  if any(count > 1 for count in counts.values()):
    observed_mixed_views.append(True)
  return original_kernargs(self, call, program, data)
QCOMComputeQueue.kernargs = record_kernargs
values = (np.arange(256, dtype=np.float32).reshape(4, 64) / 8 - 7).astype(np.float16 if {half_input!r} else np.float32)
image = Tensor(values).contiguous().realize()
expected = values.copy()
for bound in (3, 15, 32, 1, 63):
  masked = image[:, :bound].pad((None, (0, 64 - bound)), value=2)
  image.assign(image + masked).realize()
  increment = np.full_like(expected, 2)
  increment[:, :bound] = expected[:, :bound]
  expected += increment
  actual = image.numpy()
  assert actual.dtype == expected.dtype
  np.testing.assert_array_equal(actual, expected)
assert observed_mixed_views
""", half_storage=half_input, required_ops=('ldib.b', 'stib.b'))
