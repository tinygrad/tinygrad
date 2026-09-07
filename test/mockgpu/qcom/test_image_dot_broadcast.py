"""Image matmul must broadcast batch axes before folding them into convolution groups."""
import pytest

from tinygrad import Tensor
from tinygrad.helpers import Context
from test.mockgpu.qcom.test_image_integration import run_image_program


@pytest.mark.parametrize("left_shape,right_shape", [
  ((3, 5), (5, 7)),
  ((2, 3, 5), (2, 5, 7)),
  ((2, 3, 5), (5, 7)),
  ((3, 5), (2, 5, 7)),
  ((2, 1, 3, 5), (1, 4, 5, 7)),
  ((2, 3, 4, 5), (2, 1, 5, 7)),
  ((2, 1, 4, 5), (2, 3, 5, 7)),
  ((2, 1, 3, 4, 5), (1, 4, 1, 5, 7)),
  ((5,), (5,)),
  ((3, 5), (5,)),
  ((5,), (5, 7)),
  ((2, 3, 5), (5,)),
], ids=["matrix", "batched", "right", "left", "both", "right-singleton", "left-singleton", "multiple-axes",
        "vector-vector", "matrix-vector", "vector-matrix", "batched-matrix-vector"])
@pytest.mark.parametrize("half_storage", [False, True], ids=["float-storage", "half-storage"])
def test_image_matmul_broadcast_matches_cpu_references(left_shape, right_shape, half_storage):
  run_image_program(f"""
import torch
from tinygrad.helpers import Context
torch.set_num_threads(1)
rng = np.random.default_rng(97)
left = rng.uniform(-1, 1, {left_shape!r}).astype(np.float32)
right = rng.uniform(-1, 1, {right_shape!r}).astype(np.float32)
for image_mode in (0, 1):
  # FLOAT16 changes image storage only. Preserve original float32 input arrays
  # for both Tensor paths, and round only the independent image-mode reference.
  reference_left, reference_right = left, right
  if image_mode and {half_storage!r}:
    reference_left = left.astype(np.float16).astype(np.float32)
    reference_right = right.astype(np.float16).astype(np.float32)
    assert np.any(left != reference_left) and np.any(right != reference_right)
  expected = reference_left @ reference_right
  torch_expected = (torch.tensor(reference_left) @ torch.tensor(reference_right)).numpy()
  with Context(IMAGE=image_mode):
    actual = (Tensor(left) @ Tensor(right)).numpy()
  assert actual.shape == expected.shape == torch_expected.shape
  assert actual.dtype == np.float32
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(actual, torch_expected, rtol=1e-6, atol=1e-6)
  if image_mode == 0:
    executed_image_ops.clear()
    executed_component_bytes.clear()
""", half_storage=half_storage)


@pytest.mark.parametrize("left_shape,right_shape", [
  ((3, 5), (2, 5, 7)),
  ((2, 1, 3, 5), (1, 4, 5, 7)),
  ((2, 3, 4, 5), (2, 1, 5, 7)),
], ids=["left", "both", "right-singleton"])
@pytest.mark.parametrize("half_storage", [False, True], ids=["float-storage", "half-storage"])
def test_image_matmul_gradients_reduce_broadcast_axes(left_shape, right_shape, half_storage):
  run_image_program(f"""
import torch
from tinygrad.helpers import Context
torch.set_num_threads(1)
rng = np.random.default_rng(101)
left = rng.uniform(-1, 1, {left_shape!r}).astype(np.float32)
right = rng.uniform(-1, 1, {right_shape!r}).astype(np.float32)
cotangent = rng.uniform(-1, 1, (left @ right).shape).astype(np.float32)
for image_mode in (0, 1):
  reference_left, reference_right = left, right
  if image_mode and {half_storage!r}:
    reference_left = left.astype(np.float16).astype(np.float32)
    reference_right = right.astype(np.float16).astype(np.float32)
  torch_left = torch.tensor(reference_left, requires_grad=True)
  torch_right = torch.tensor(reference_right, requires_grad=True)
  ((torch_left @ torch_right) * torch.tensor(cotangent)).sum().backward()
  with Context(IMAGE=image_mode):
    actual_left, actual_right = Tensor(left), Tensor(right)
    gradients = ((actual_left @ actual_right) * Tensor(cotangent)).sum().gradient(actual_left, actual_right)
    # Differentiating expansion must sum the replicated batches back into the
    # original operand shapes; gradient cotangents do not acquire half storage.
    for gradient, reference in zip(gradients, (torch_left.grad, torch_right.grad)):
      actual = gradient.numpy()
      assert actual.shape == tuple(reference.shape)
      np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-6, atol=1e-6)
  if image_mode == 0:
    executed_image_ops.clear()
    executed_component_bytes.clear()
""", half_storage=half_storage)


@pytest.mark.parametrize("image_mode", [0, 1])
@pytest.mark.parametrize("half_storage", [0, 1])
def test_image_matmul_rejects_incompatible_batch_and_contracting_axes(image_mode, half_storage):
  # Batch incompatibility must fail at shape validation, before flattening can
  # divide by zero or accidentally reinterpret a different grouping as valid.
  with Context(IMAGE=image_mode, FLOAT16=half_storage):
    for left_shape, right_shape in [((2, 3, 5), (3, 5, 7)), ((2, 1, 3, 5), (3, 2, 5, 7)), ((2, 3, 4, 5), (3, 2, 5, 7))]:
      with pytest.raises(IndexError):
        Tensor.empty(left_shape, device="CPU").matmul(Tensor.empty(right_shape, device="CPU"))
    with pytest.raises(RuntimeError):
      Tensor.empty(2, 3, 5, device="CPU").matmul(Tensor.empty(2, 4, 7, device="CPU"))
