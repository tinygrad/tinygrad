#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.state import get_parameters
from tinygrad.ops import LazyOp, LoadOps
from tinygrad.jit import TinyJit, JIT_SUPPORTED_DEVICE
from tinygrad.helpers import dtypes, CI
from tinygrad.lazy import Device

from examples.llama import Transformer

# for speed
def derandomize(x):
  if isinstance(x, LazyOp):
    if x.op == LoadOps.RAND: x.op = LoadOps.EMPTY
    x.src = tuple([derandomize(s) for s in x.src])
  else:
    x.op = derandomize(x.op)
  return x

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = derandomize(p.lazydata)
    p.realize()

def helper_test_jitted_correctness(gen, train, train_jit):
  nojit = train(*gen()).numpy()
  for _ in range(5): jit = train_jit(*gen()).numpy()
  np.testing.assert_allclose(nojit, jit, rtol=1e-3, atol=1e-5)

@unittest.skipUnless(Device.DEFAULT in JIT_SUPPORTED_DEVICE, "needs JIT")
class TestJittedModels(unittest.TestCase):
  def test_jitted_tiny_llama(self):
    old_type = Tensor.default_type
    Tensor.default_type = dtypes.float16

    args_tiny = {"dim": 1024, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    model = Transformer(**args_tiny)
    derandomize_model(model)
    def test(t): return model(t, 0).realize()

    @TinyJit
    def test_jit(t): return model(t, 0).realize()
    helper_test_jitted_correctness(lambda: (Tensor([[1,]]),), test, test_jit)
    Tensor.default_type = old_type

  @unittest.skipUnless(not CI, "huge for CI")
  def test_jitted_stable_diffusion(self):
    from examples.stable_diffusion import UNetModel
    model = UNetModel()
    derandomize_model(model)
    def test(t, t2): return model(t, 801, t2).realize()

    @TinyJit
    def test_jit(t, t2): return model(t, 801, t2).realize()
    helper_test_jitted_correctness(lambda: (Tensor.randn(1, 4, 16, 16),Tensor.randn(1, 77, 768)), test, test_jit)

if __name__ == "__main__":
  unittest.main()
