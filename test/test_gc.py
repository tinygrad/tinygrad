#!/usr/bin/env python
import gc
import unittest
import numpy as np
from tinygrad.device import Buffer
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.ops import PatternMatcher, UOp, UOps, UPat
from tinygrad.tensor import Tensor

def tensors_allocated():
  return sum([isinstance(x, Tensor) for x in gc.get_objects()])

def bufs_allocated():
  return sum([isinstance(x, Buffer) for x in gc.get_objects()])

class TestGC(unittest.TestCase):

  def test_gc(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(4, 4, requires_grad=True)
    b = Tensor.zeros(4, 4, requires_grad=True)
    (a*b).mean().backward()
    assert (tensors_allocated() > 0)
    del a,b
    assert (tensors_allocated() == 1) # one for Tensor._device_rng_counters

  def test_gc_complex(self):
    Tensor.manual_seed(0)
    a = Tensor(np.zeros((4, 4), dtype=np.float32), requires_grad=True)
    b = Tensor.rand(4, 4, requires_grad=True)
    assert (tensors_allocated() == 4)
    (a*b).mean().backward()
    assert (tensors_allocated() == 5)
    del b
    assert (tensors_allocated() == 3)
    b = Tensor(np.zeros((4, 4), dtype=np.float32), requires_grad=True)
    print(tensors_allocated())
    (a*b).mean().backward()
    print(tensors_allocated())
    assert (tensors_allocated() == 5)
    del b
    assert (tensors_allocated() == 3)

  @unittest.expectedFailure
  def test_pattern_matcher_gc(self):
    init = bufs_allocated()
    buf = Tensor.ones(4, 4).contiguous().realize().lazydata.buffer
    uop = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), buf)
    matcher = PatternMatcher([
      (UPat(UOps.DEFINE_GLOBAL, name="x"), lambda x:UOp(UOps.DEFINE_GLOBAL, x.dtype, (), 0))
    ])
    ret = matcher.rewrite(uop)
    del uop
    del buf
    assert ret.arg == 0
    self.assertEqual(bufs_allocated()-init, 0)

if __name__ == '__main__':
  unittest.main()
