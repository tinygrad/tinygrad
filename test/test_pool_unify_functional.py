import numpy as np
import pytest

from numpy.lib.stride_tricks import sliding_window_view

from tinygrad import Tensor


def _numpy_unfold_1d(arr: np.ndarray, size: int, step: int) -> np.ndarray:
  if size > arr.shape[0]:
    raise ValueError("size must be <= len(arr)")
  view = sliding_window_view(arr, size)
  return view[::step]


def _numpy_pool(arr: np.ndarray, kernel: tuple[int, int], stride: tuple[int, int], dilation: tuple[int, int], padding: int, *, mode: str) -> np.ndarray:
  pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
  pad_value = 0.0 if mode == "avg" else np.finfo(np.float32).min
  padded = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)
  eff_k = ((kernel[0] - 1) * dilation[0] + 1, (kernel[1] - 1) * dilation[1] + 1)
  view = sliding_window_view(padded, eff_k, axis=(-2, -1))
  view = view[:, :, ::stride[0], ::stride[1], :, :]
  view = view[..., ::dilation[0], ::dilation[1]]
  if mode == "avg":
    return view.mean(axis=(-1, -2))
  if mode == "max":
    return view.max(axis=(-1, -2))
  raise ValueError(f"unsupported mode {mode}")


def _numpy_conv2d(x: np.ndarray, w: np.ndarray, stride: tuple[int, int], dilation: tuple[int, int], groups: int) -> np.ndarray:
  eff_k = ((w.shape[2] - 1) * dilation[0] + 1, (w.shape[3] - 1) * dilation[1] + 1)
  view = sliding_window_view(x, eff_k, axis=(-2, -1))
  view = view[:, :, ::stride[0], ::stride[1], :, :]
  view = view[..., ::dilation[0], ::dilation[1]]

  bs, cin, out_h, out_w, k0, k1 = view.shape
  cin_per_group = cin // groups
  rcout = w.shape[0] // groups
  view_grouped = view.reshape(bs, groups, cin_per_group, out_h, out_w, k0, k1)

  outputs = []
  for g in range(groups):
    win = view_grouped[:, g].transpose(0, 3, 4, 1, 5, 6)
    weight = w[g * rcout:(g + 1) * rcout]
    conv = np.tensordot(win, weight, axes=([3, 4, 5], [1, 2, 3]))
    outputs.append(conv.transpose(0, 3, 1, 2))
  return np.concatenate(outputs, axis=1)


@pytest.mark.parametrize("length", [7, 8, 9])
@pytest.mark.parametrize("size", [1, 2, 3, 5])
@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_unfold_matches_numpy(length: int, size: int, step: int) -> None:
  if size > length:
    pytest.skip("window larger than tensor")
  data = np.arange(length, dtype=np.float32)
  tensor = Tensor(data)
  actual = tensor.unfold(0, size, step).realize().numpy()
  expected = _numpy_unfold_1d(data, size, step)
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hw", [7, 8])
@pytest.mark.parametrize("kernel", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_avg_pool2d_matches_numpy(hw: int, kernel: tuple[int, int], stride: tuple[int, int], dilation: int, padding: int) -> None:
  seed = hw + 10 * kernel[0] + 100 * stride[0] + 1000 * dilation + 10000 * padding
  generator = np.random.default_rng(seed)
  arr = generator.standard_normal((2, 3, hw, hw)).astype(np.float32)
  dilation_t = (dilation, dilation)
  tensor = Tensor(arr.copy())
  actual = tensor.avg_pool2d(kernel_size=kernel, stride=stride, dilation=dilation, padding=padding).realize().numpy()
  expected = _numpy_pool(arr, kernel, stride, dilation_t, padding, mode="avg")
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hw", [7, 8])
@pytest.mark.parametrize("kernel", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_max_pool2d_matches_numpy(hw: int, kernel: tuple[int, int], stride: tuple[int, int], dilation: int, padding: int) -> None:
  seed = 1 + hw + 10 * kernel[0] + 100 * stride[0] + 1000 * dilation + 10000 * padding
  generator = np.random.default_rng(seed)
  arr = generator.standard_normal((2, 3, hw, hw)).astype(np.float32)
  dilation_t = (dilation, dilation)
  tensor = Tensor(arr.copy())
  actual = tensor.max_pool2d(kernel_size=kernel, stride=stride, dilation=dilation, padding=padding).realize().numpy()
  expected = _numpy_pool(arr, kernel, stride, dilation_t, padding, mode="max")
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("kernel", [(3, 3), (2, 2)])
@pytest.mark.parametrize("groups", [1, "depthwise"])
def test_conv2d_matches_numpy(stride: int, dilation: int, kernel: tuple[int, int], groups: int | str) -> None:
  bs, cin, hw = 2, 4, 9
  dilation_t = (dilation, dilation)
  stride_t = (stride, stride)
  g = cin if groups == "depthwise" else 1
  cout = cin if g == cin else 6
  seed = 2 + stride + 10 * dilation + 100 * kernel[0] + (0 if g == 1 else 1000)
  generator = np.random.default_rng(seed)
  x = generator.standard_normal((bs, cin, hw, hw)).astype(np.float32)
  w = generator.standard_normal((cout, cin // g, kernel[0], kernel[1])).astype(np.float32)

  tensor_x = Tensor(x.copy())
  tensor_w = Tensor(w.copy())
  actual = tensor_x.conv2d(tensor_w, groups=g, stride=stride, dilation=dilation).realize().numpy()
  expected = _numpy_conv2d(x, w, stride_t, dilation_t, g)
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
