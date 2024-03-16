from typing import List
import unittest
from test.helpers import assert_jit_cache_len
from tinygrad.ops import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.features.jit import TinyJit

def helper_test_corealize(outs: List[Tensor], kernel_count: int):
  GlobalCounters.reset()
  Tensor.corealize(outs)
  assert GlobalCounters.kernel_count == kernel_count

class TestCorealize(unittest.TestCase):
  def test_simple_group(self):
    a, b = Tensor([1,2]).realize(), Tensor([3,4]).realize()
    helper_test_corealize([a+b, a*b], 1)

  def test_ungroup_reduce(self):
    a, b = Tensor([1]).realize(), Tensor([3,4]).realize()
    helper_test_corealize([a.float(), b.sum()+1], 2)

  def test_simple_jit_group(self):
    @TinyJit
    def fxn(a, b): return a + b, a * b
    for i in range(3):
      a, b = Tensor([i, 1, 2]), Tensor([i])
      fxn(a, b)
    assert_jit_cache_len(fxn, 1)

  def test_jit_ungroup_reduce(self):
    @TinyJit
    def fxn(a, b): return a + b, a.sum()
    for i in range(3):
      a, b = Tensor([i, 1, 2]), Tensor([i])
      fxn(a, b)
    assert_jit_cache_len(fxn, 2)
