import os
from contextlib import contextmanager

import numpy as np
import pytest

from tinygrad import Tensor

np.random.seed(0)

FAST = os.getenv("POOL_PARITY_FAST", "0") == "1"

UNFOLD_LENGTHS = [7, 8, 9] if not FAST else [7]
UNFOLD_SIZES = [1, 2, 3, 5] if not FAST else [1, 3]
UNFOLD_STEPS = [1, 2, 3, 4] if not FAST else [1, 3]

POOL_HW = [7, 8] if not FAST else [7]
POOL_KERNELS = [(1, 1), (2, 2), (3, 3)] if not FAST else [(1, 1), (3, 3)]
POOL_STRIDES = [(1, 1), (2, 2), (3, 3)] if not FAST else [(1, 1), (3, 3)]
POOL_DILATIONS = [1, 2] if not FAST else [1]
POOL_PADDINGS = [0, 1] if not FAST else [0]

CONV_STRIDES = [1, 2, 3] if not FAST else [1, 2]
CONV_DILATIONS = [1, 2] if not FAST else [1]
CONV_KERNELS = [(3, 3), (2, 2)] if not FAST else [(3, 3)]
CONV_GROUPS = [1, "depthwise"] if not FAST else [1, "depthwise"]


@contextmanager
def pool_impl(mode:str|None):
  old = os.environ.get("POOL_IMPL")
  try:
    if mode is None:
      os.environ.pop("POOL_IMPL", None)
    else:
      os.environ["POOL_IMPL"] = mode
    yield
  finally:
    if old is None:
      os.environ.pop("POOL_IMPL", None)
    else:
      os.environ["POOL_IMPL"] = old


def _assert_pool_parity(fn):
  with pool_impl(None):
    main = fn().realize().numpy()
  with pool_impl("ALT"):
    alt = fn().realize().numpy()
  np.testing.assert_allclose(main, alt, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("length", UNFOLD_LENGTHS)
@pytest.mark.parametrize("size", UNFOLD_SIZES)
@pytest.mark.parametrize("step", UNFOLD_STEPS)
def test_unfold_parity(length:int, size:int, step:int):
  if size > length:
    pytest.skip("window larger than tensor")
  base = Tensor.arange(length).realize()

  def run():
    return base.unfold(0, size, step)

  _assert_pool_parity(run)


@pytest.mark.parametrize("hw", POOL_HW)
@pytest.mark.parametrize("kernel", POOL_KERNELS)
@pytest.mark.parametrize("stride", POOL_STRIDES)
@pytest.mark.parametrize("dilation", POOL_DILATIONS)
@pytest.mark.parametrize("padding", POOL_PADDINGS)
def test_avg_pool2d_parity(hw:int, kernel:tuple[int, int], stride:tuple[int, int], dilation:int, padding:int):
  eff = tuple((k - 1) * dilation + 1 for k in kernel)
  if any((e > hw + 2 * padding) for e in eff):
    pytest.skip("kernel exceeds padded dimension")
  base = Tensor.rand(2, 3, hw, hw).realize()

  def run():
    return base.avg_pool2d(kernel_size=kernel, stride=stride, dilation=dilation, padding=padding)

  _assert_pool_parity(run)


@pytest.mark.parametrize("hw", POOL_HW)
@pytest.mark.parametrize("kernel", POOL_KERNELS)
@pytest.mark.parametrize("stride", POOL_STRIDES)
@pytest.mark.parametrize("dilation", POOL_DILATIONS)
@pytest.mark.parametrize("padding", POOL_PADDINGS)
def test_max_pool2d_parity(hw:int, kernel:tuple[int, int], stride:tuple[int, int], dilation:int, padding:int):
  eff = tuple((k - 1) * dilation + 1 for k in kernel)
  if any((e > hw + 2 * padding) for e in eff):
    pytest.skip("kernel exceeds padded dimension")
  base = Tensor.rand(2, 3, hw, hw).realize()

  def run():
    return base.max_pool2d(kernel_size=kernel, stride=stride, dilation=dilation, padding=padding)

  _assert_pool_parity(run)


@pytest.mark.parametrize("stride", CONV_STRIDES)
@pytest.mark.parametrize("dilation", CONV_DILATIONS)
@pytest.mark.parametrize("kernel", CONV_KERNELS)
@pytest.mark.parametrize("groups", CONV_GROUPS)
def test_conv2d_parity(stride:int, dilation:int, kernel:tuple[int, int], groups:int|str):
  bs, cin, hw = 2, 4, 9
  g = cin if groups == "depthwise" else 1
  cout = cin if g == cin else 6
  eff = tuple((k - 1) * dilation + 1 for k in kernel)
  if any(e > hw for e in eff):
    pytest.skip("kernel exceeds input dimension")
  x = Tensor.rand(bs, cin, hw, hw).realize()
  w = Tensor.rand(cout, cin // g, *kernel).realize()

  def run():
    return x.conv2d(w, groups=g, stride=stride, dilation=dilation)

  _assert_pool_parity(run)
