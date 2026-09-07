"""Explicit references for FLOAT16 image storage with original float32 inputs.

These tests do not change shared backend tolerances or their float32 references.
They check the image frontend's explicit half casts, followed by float32 math.
"""
import pytest

from test.mockgpu.qcom.test_image_integration import run_image_program


def test_half_image_matmul_quantizes_original_inputs_before_float32_accumulation():
  run_image_program("""
rng = np.random.default_rng(71)
left = rng.uniform(-2, 2, (3, 5)).astype(np.float32)
right = rng.uniform(-2, 2, (5, 7)).astype(np.float32)
stored_left = left.astype(np.float16).astype(np.float32)
stored_right = right.astype(np.float16).astype(np.float32)
assert np.any(left != stored_left) and np.any(right != stored_right)
actual = (Tensor(left) @ Tensor(right)).numpy()
expected = stored_left @ stored_right
np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
""", half_storage=True)


def test_half_image_convolution_quantizes_original_inputs_and_weights():
  run_image_program("""
import torch
import torch.nn.functional as functional
torch.set_num_threads(1)
rng = np.random.default_rng(73)
values = rng.uniform(-2, 2, (1, 3, 4, 5)).astype(np.float32)
weights = rng.uniform(-2, 2, (2, 3, 2, 3)).astype(np.float32)
stored_values = values.astype(np.float16).astype(np.float32)
stored_weights = weights.astype(np.float16).astype(np.float32)
assert np.any(values != stored_values) and np.any(weights != stored_weights)
actual = Tensor(values).conv2d(Tensor(weights), padding=1, stride=2).numpy()
expected = functional.conv2d(torch.tensor(stored_values), torch.tensor(stored_weights), padding=1, stride=2).numpy()
np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
""", half_storage=True)


def test_half_image_pointwise_gradients_use_stored_values_and_float32_sums():
  # Image lowering elides adjacent float->half->float casts without storage
  # (pm_simplify_add_image). The forward contiguous half buffers still round
  # inputs/weights. For these pointwise products, dX is the stored weight and
  # dW is a float32 sum of stored inputs; no extra half gradient store exists.
  run_image_program("""
values = (np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3) - 10) / 7
weights = np.array([0.12345, -1.2345, 0.33337, 2.71828], np.float32).reshape(4, 1, 1, 1)
stored_values = values.astype(np.float16).astype(np.float32)
stored_weights = weights.astype(np.float16).astype(np.float32)
assert np.any(values != stored_values) and np.any(weights != stored_weights)
left, kernel = Tensor(values), Tensor(weights)
left_gradient, weight_gradient = left.conv2d(kernel, groups=4).sum().gradient(left, kernel)
expected_left = np.broadcast_to(stored_weights.reshape(1, 4, 1, 1), values.shape)
expected_weight = stored_values.sum(axis=(0, 2, 3)).reshape(weights.shape)
np.testing.assert_array_equal(left_gradient.numpy(), expected_left)
np.testing.assert_array_equal(weight_gradient.numpy(), expected_weight)
""", half_storage=True)


@pytest.mark.parametrize("left_shape,right_shape", [
  ((2, 3, 5), (2, 5, 7)),
  ((2, 3, 5), (5, 7)),
  ((2, 3, 4, 5), (5, 7)),
  ((2, 3, 4, 5), (3, 5, 7)),
  ((3, 5), (2, 5, 7)),
  ((2, 1, 3, 5), (1, 4, 5, 7)),
], ids=["batched", "broadcast-right", "broadcast-right-multiple-axes", "broadcast-right-leading-axis", "broadcast-left", "broadcast-both"])
def test_half_image_matmul_batch_and_broadcast_storage(left_shape, right_shape):
  # image_dot lowers each broadcasted batch to image_conv2d; only its input
  # and weight stores round, while products and their reduction stay float32.
  run_image_program(f"""
rng = np.random.default_rng(79)
left = rng.uniform(-2, 2, {left_shape!r}).astype(np.float32)
right = rng.uniform(-2, 2, {right_shape!r}).astype(np.float32)
stored_left = left.astype(np.float16).astype(np.float32)
stored_right = right.astype(np.float16).astype(np.float32)
assert np.any(left != stored_left) and np.any(right != stored_right)
actual = (Tensor(left) @ Tensor(right)).numpy()
expected = stored_left @ stored_right
assert np.max(np.abs(expected - left @ right)) > 1e-5
np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
""", half_storage=True)


@pytest.mark.parametrize("operation,values_shape,weights_shape,options", [
  ("conv2d", (2, 4, 4, 5), (6, 2, 2, 3), {"groups": 2, "padding": 1}),
  ("conv2d", (1, 3, 5, 6), (5, 3, 2, 3), {"stride": (2, 1), "padding": 1}),
  ("conv2d", (1, 4, 5, 6), (4, 1, 2, 2), {"groups": 4, "dilation": (2, 1), "padding": 1}),
  ("conv_transpose2d", (1, 3, 3, 4), (3, 2, 2, 3), {"stride": (2, 1), "padding": 1, "output_padding": (1, 0)}),
  ("conv_transpose2d", (1, 4, 3, 4), (4, 3, 2, 2), {"groups": 2, "dilation": (2, 1), "padding": 1}),
], ids=["grouped", "strided", "depthwise-dilated", "transposed-strided", "transposed-grouped-dilated"])
@pytest.mark.parametrize("backward", [False, True], ids=["forward", "gradients"])
def test_half_image_convolution_variants_storage_and_gradients(operation, values_shape, weights_shape, options, backward):
  # conv_transpose2d only reshapes/flips/zero-inserts before image_conv2d's
  # storage boundary. Bias is added after the float32 reduction and stays float32.
  # Backward follows the stored forward operands; adjacent gradient half casts
  # are elided by pm_simplify_add_image when no half storage separates them.
  run_image_program(f"""
import torch
import torch.nn.functional as functional
torch.set_num_threads(1)
rng = np.random.default_rng(83)
values = rng.uniform(-1, 1, {values_shape!r}).astype(np.float32)
weights = rng.uniform(-1, 1, {weights_shape!r}).astype(np.float32)
options = {options!r}
channels = weights.shape[1] * options.get("groups", 1) if {operation!r} == "conv_transpose2d" else weights.shape[0]
bias = rng.uniform(-1, 1, channels).astype(np.float32)
stored_values = values.astype(np.float16).astype(np.float32)
stored_weights = weights.astype(np.float16).astype(np.float32)
assert np.any(values != stored_values) and np.any(weights != stored_weights)
assert np.any(bias != bias.astype(np.float16).astype(np.float32))
left, kernel, offset = Tensor(values), Tensor(weights), Tensor(bias)
actual = getattr(left, {operation!r})(kernel, offset, **options)
reference_left = torch.tensor(stored_values, requires_grad=True)
reference_kernel = torch.tensor(stored_weights, requires_grad=True)
reference_bias = torch.tensor(bias, requires_grad=True)
reference_op = getattr(functional, {operation!r})
expected = reference_op(reference_left, reference_kernel, reference_bias, **options)
full_precision = reference_op(torch.tensor(values), torch.tensor(weights), torch.tensor(bias), **options).detach().numpy()
assert np.max(np.abs(expected.detach().numpy() - full_precision)) > 1e-5
if {backward!r}:
  # Nonuniform, non-half-representable cotangents expose unintended rounding
  # in backward even when the forward operands were correctly stored as half.
  cotangent = rng.uniform(-1, 1, tuple(expected.shape)).astype(np.float32)
  assert np.any(cotangent != cotangent.astype(np.float16).astype(np.float32))
  gradients = (actual * Tensor(cotangent)).sum().gradient(left, kernel, offset)
  (expected * torch.tensor(cotangent)).sum().backward()
np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), rtol=1e-6, atol=1e-6)
if {backward!r}:
  for actual_gradient, expected_gradient in zip(gradients, (reference_left.grad, reference_kernel.grad, reference_bias.grad)):
    np.testing.assert_allclose(actual_gradient.numpy(), expected_gradient.numpy(), rtol=1e-6, atol=1e-6)
""", half_storage=True)


@pytest.mark.parametrize("query_heads,key_heads,query_length,key_length,mask_kind", [
  (2, 2, 3, 3, "none"),
  (2, 2, 4, 4, "causal"),
  (2, 2, 3, 5, "none"),
  (2, 2, 5, 3, "causal"),
  (4, 2, 3, 5, "none"),
  (2, 2, 3, 5, "additive"),
  (4, 2, 3, 5, "additive"),
], ids=["plain", "causal", "unequal-lengths", "causal-unequal-lengths", "gqa", "additive-mask", "gqa-additive-mask"])
def test_half_image_attention_rounds_at_both_matmul_storage_boundaries(query_heads, key_heads, query_length, key_length, mask_kind):
  # scaled_dot_product_attention: q/k half stores -> float32 score, mask and
  # softmax -> probability/v half stores -> float32 output accumulation.
  # In particular, neither the additive mask nor the softmax math is half.
  run_image_program(f"""
rng = np.random.default_rng(89)
query = rng.uniform(-2, 2, (2, {query_heads}, {query_length}, 4)).astype(np.float32)
key = rng.uniform(-2, 2, (2, {key_heads}, {key_length}, 4)).astype(np.float32)
value = rng.uniform(-2, 2, (2, {key_heads}, {key_length}, 5)).astype(np.float32)
def stored(array): return array.astype(np.float16).astype(np.float32)
assert all(np.any(array != stored(array)) for array in (query, key, value))
repeats = {query_heads} // {key_heads}
stored_key = np.repeat(stored(key), repeats, axis=1)
stored_value = np.repeat(stored(value), repeats, axis=1)
scores = (stored(query) @ stored_key.swapaxes(-2, -1)) / np.float32(2)
mask = None
if {mask_kind!r} == "causal":
  scores = np.where(np.tri({query_length}, {key_length}, dtype=bool), scores, np.float32(-np.inf))
elif {mask_kind!r} == "additive":
  mask = rng.uniform(-1, 1, ({query_length}, {key_length})).astype(np.float32)
  assert np.any(mask != stored(mask))
  scores = scores + mask
exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
assert np.any(probabilities != stored(probabilities))
expected = stored(probabilities) @ stored_value
# This counterfactual rejects omitting the second, probability storage boundary.
assert np.max(np.abs(expected - probabilities @ stored_value)) > 1e-5
actual = Tensor(query).scaled_dot_product_attention(
  Tensor(key), Tensor(value), attn_mask=None if mask is None else Tensor(mask),
  is_causal={mask_kind == 'causal'!r}, enable_gqa={query_heads != key_heads!r}).numpy()
np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
""", half_storage=True)
