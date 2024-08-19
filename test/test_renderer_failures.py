import unittest
from typing import List, cast
import numpy as np
from tinygrad.device import Buffer, Device
from tinygrad.dtype import PtrDType, DType, dtypes
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import dedup, flatten
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.ops import BinaryOps, UOp, UOps
from tinygrad.renderer import Program
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.lazy import LazyBuffer

def _test_uop_result(inputs:List[Tensor], stores:List[UOp]):
  for x in inputs: x.realize()
  assert all(x.op is UOps.STORE for x in stores)
  # NOTE: we only toposort the stores
  uops: List[UOp] = []
  def _recursive_add(uop:UOp) -> List[UOp]: return flatten([_recursive_add(x) for x in uop.src])+[uop]
  uops = dedup(flatten(_recursive_add(st) for st in stores))
  outbufs = [Buffer(Device.DEFAULT, 1, cast(DType,u.src[2].dtype)).allocate() for u in uops if u.op is UOps.STORE]
  inbufs = [cast(LazyBuffer,x.lazydata).base.buffer for x in inputs]
  src = Device[Device.DEFAULT].renderer.render("test", uops)
  ei = CompiledRunner(Program("test", src, Device.DEFAULT, uops=uops))
  ei.exec(outbufs+inbufs)
  return [np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)) for x in outbufs]

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "uops are for cstyle")
class TestCStyleFailures(unittest.TestCase):
  def test_inline_const_alu(self):
    a = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    b = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 1)
    idx = UOp.const(dtypes.int, 0)
    ld = UOp(UOps.LOAD, dtypes.int, (b, idx))
    alu = ld.alu(BinaryOps.MAX, UOp.const(dtypes.int, dtypes.min(dtypes.int)))
    store = UOp.store(a, idx, alu)
    # CLANG doesn't use the max function
    ret = _test_uop_result([Tensor([1])], [store])[0]
    self.assertEqual(ret[0], 1)

if __name__ == '__main__':
  unittest.main()
