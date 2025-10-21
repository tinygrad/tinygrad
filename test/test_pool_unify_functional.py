import os
from contextlib import contextmanager

import numpy as np
import pytest

from tinygrad import Tensor

np.random.seed(0)


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


@pytest.mark.parametrize("length", [7, 8, 9])
@pytest.mark.parametrize("size", [1, 2, 3, 5])
@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_unfold_parity(length:int, size:int, step:int):
  if size > length:
    pytest.skip("window larger than tensor")
  base = Tensor.arange(length).realize()

  def run():
    return base.unfold(0, size, step)

  _assert_pool_parity(run)


@pytest.mark.parametrize("hw", [7, 8])
@pytest.mark.parametrize("kernel", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_avg_pool2d_parity(hw:int, kernel:tuple[int, int], stride:tuple[int, int], dilation:int, padding:int):
  eff = tuple((k - 1) * dilation + 1 for k in kernel)
  if any((e > hw + 2 * padding) for e in eff):
    pytest.skip("kernel exceeds padded dimension")
  base = Tensor.rand(2, 3, hw, hw).realize()

  def run():
    return base.avg_pool2d(kernel_size=kernel, stride=stride, dilation=dilation, padding=padding)

  _assert_pool_parity(run)


@pytest.mark.parametrize("hw", [7, 8])
@pytest.mark.parametrize("kernel", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_max_pool2d_parity(hw:int, kernel:tuple[int, int], stride:tuple[int, int], dilation:int, padding:int):
  eff = tuple((k - 1) * dilation + 1 for k in kernel)
  if any((e > hw + 2 * padding) for e in eff):
    pytest.skip("kernel exceeds padded dimension")
  base = Tensor.rand(2, 3, hw, hw).realize()

  def run():
    return base.max_pool2d(kernel_size=kernel, stride=stride, dilation=dilation, padding=padding)

  _assert_pool_parity(run)


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("kernel", [(3, 3), (2, 2)])
@pytest.mark.parametrize("groups", [1, "depthwise"])
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
