import os
from contextlib import contextmanager

import pytest

from tinygrad import Tensor, Device
from tinygrad.codegen import full_rewrite
from tinygrad.helpers import CPU_LLVM


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


def _uops_signature(tensor:Tensor) -> list[str]:
  schedule = tensor.schedule()
  sig:list[str] = []
  for sched_item in schedule:
    sink = sched_item.ast.sink()
    for uop in full_rewrite(sink):
      sig.append(repr(uop))
  return sig


def _kernel_text(tensor:Tensor) -> list[str]:
  schedule = tensor.schedule()
  texts:list[str] = []
  for sched_item in schedule:
    sink = sched_item.ast.sink()
    renderer = Device[Device.DEFAULT].renderer
    program = renderer.render(full_rewrite(sink))
    norm = "\n".join(line.strip() for line in program.splitlines() if line.strip())
    texts.append(norm)
  return texts


def _assert_parity(build):
  with pool_impl(None):
    main = build()
    main_uops = _uops_signature(main)
  with pool_impl("ALT"):
    alt = build()
    alt_uops = _uops_signature(alt)
  assert main_uops == alt_uops

  if Device.DEFAULT == "CPU" and CPU_LLVM:
    with pool_impl(None):
      main_text = _kernel_text(build())
    with pool_impl("ALT"):
      alt_text = _kernel_text(build())
    assert main_text == alt_text


def test_kernel_unfold_parity():
  def build():
    base = Tensor.arange(8)
    return base.unfold(0, 3, 2)

  _assert_parity(build)


def test_kernel_pool_parity():
  def build():
    t = Tensor.arange(1*2*6*6).reshape(1, 2, 6, 6)
    return t.avg_pool2d(kernel_size=(3, 3), stride=(2, 2), dilation=1, padding=1)

  _assert_parity(build)


def test_kernel_max_pool_parity():
  def build():
    t = Tensor.arange(1*2*6*6).reshape(1, 2, 6, 6)
    return t.max_pool2d(kernel_size=(3, 3), stride=(1, 1), dilation=1, padding=0)

  _assert_parity(build)


def test_kernel_conv_parity():
  def build():
    x = Tensor.arange(1*4*7*7).reshape(1, 4, 7, 7)
    w = Tensor.arange(4*1*3*3).reshape(4, 1, 3, 3)
    return x.conv2d(w, groups=4, stride=2, dilation=1)

  _assert_parity(build)
