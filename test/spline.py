import numpy as np

from tinygrad.helpers import dtypes
from tinygrad.tensor import Tensor

np.random.seed(1337)
min_bin_height =  0.001
min_bin_width = 0.001
min_derivative =  0.001
inverse = True
tail_bound = 5.0


inputs = np.random.rand(1, 1, 361).astype(np.float32)
unnormalized_derivatives = np.random.rand(1, 1, 361, 9).astype(np.float32)
unnormalize_heights = np.random.rand(1, 1, 361, 10).astype(np.float32)
unnormalize_width = np.random.rand(1, 1, 361, 10).astype(np.float32)

tiny_inputs = Tensor(inputs)

np_inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
tiny_inside_interval_mask = (tiny_inputs >= -tail_bound) * (tiny_inputs <= tail_bound)
assert np.allclose(np_inside_interval_mask, tiny_inside_interval_mask.numpy())

tiny_outside_interval_mask = tiny_inside_interval_mask == False
np_outside_interval_mask = ~np_inside_interval_mask
assert np.allclose(np_outside_interval_mask, tiny_outside_interval_mask.numpy())

tiny_unnormalized_derivatives = Tensor(unnormalized_derivatives)
shape = list(tiny_unnormalized_derivatives.shape)
shape[-1] = 1  # Last dimension has size 1
tiny_constant = Tensor.full(tuple(shape), np.log(np.exp(1 - min_derivative) - 1))
tiny_unnormalized_derivatives = tiny_constant.cat(tiny_unnormalized_derivatives, dim=-1).cat(tiny_constant, dim=-1)
np_constant = np.log(np.exp(1 - min_derivative) - 1)
np_unnormalized_derivatives = np.pad(unnormalized_derivatives, ((0, 0), (0, 0), (0, 0), (1, 1)))
np_unnormalized_derivatives[..., 0] = np_constant
np_unnormalized_derivatives[..., -1] = np_constant

assert np.allclose(np_unnormalized_derivatives, tiny_unnormalized_derivatives.numpy())

tiny_outputs, tiny_logabsdet = Tensor.zeros_like(tiny_inputs), Tensor.zeros_like(tiny_inputs)
tiny_outputs = tiny_outputs + tiny_outside_interval_mask.cast(dtype=dtypes.int8)

np_outputs, np_logabsdet = np.zeros_like(inputs), np.zeros_like(inputs)
np_outputs[np_outside_interval_mask] = inputs[np_outside_interval_mask]
np_logabsdet[np_outside_interval_mask] = 0

assert np.allclose(np_outputs, tiny_outputs.numpy())
assert np.allclose(np_logabsdet, tiny_logabsdet.numpy())


def group_consecutive(lst):
  ranges = [[lst[0]]]
  for x in lst[1:]:
    if x - ranges[-1][-1] == 1:   ranges[-1].append(x)
    else: ranges.append([x])
  return ranges


for indices in group_consecutive(np_inside_interval_mask.nonzero()[-1]):
  print(f"{indices[0]} to {indices[-1]}")
  print(tiny_inputs[indices[0]:indices[-1]])

p_inputs = tiny_inputs.squeeze(0)
print(p_inputs.cumsum(axis=1).numpy())
print(inputs.squeeze(0).cumsum(axis=-1))
right = 69
left = 420
n_widths = inputs.squeeze(0)
np_cum_widths = (right - left) * np.pad(n_widths.cumsum(axis=-1), pad_width=((0, 0), (1, 0)), mode='constant',
                                        constant_values=0.0) + left
np_cum_widths[..., 0], np_cum_widths[..., -1] = left, right

def pad_lr(t, fill_l, fill_r):
  constant_l = Tensor.full(get_shape(t), fill_l)
  constant_r = Tensor.full(get_shape(t), fill_r)
  return constant_l.cat(t, dim=-1).cat(constant_r, dim=-1)
def get_shape(unnormalized_widths):
  shape = list(unnormalized_widths.shape)
  shape[-1] = 1  # Last dimension has size 1
  return tuple(shape)
tiny_widths = tiny_inputs.squeeze(0)
tiny_cum_widths = pad_lr(((right - left) * tiny_widths[..., :-1].cumsum(axis=1) + left), left, right)
assert np.allclose(np_cum_widths, tiny_cum_widths.numpy())


def take_along_axis(arr, indices):
  result = []
  for i, index in enumerate(indices):
    result.append(arr[i, index])

  # Convert list of tinygrad.Tensors to a single 1D tinygrad.Tensor
  result = Tensor.stack(result)
  return result

np_bin_locations = np_cum_widths
np_bin_idx = np.sum(n_widths[..., None] >= np_bin_locations, axis=-1) - 1
print(np.take_along_axis(np_cum_widths, np_bin_idx, axis=-1).shape)
tiny_bin_locations = tiny_cum_widths
tiny_bin_idx = (tiny_inputs[..., None] >= tiny_bin_locations).sum(axis=-1) - 1
print(take_along_axis(tiny_inputs, tiny_bin_idx).shape)