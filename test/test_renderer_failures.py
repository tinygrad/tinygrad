import unittest
from typing import List, cast
import numpy as np
from tinygrad.codegen.uops import UOp, UOps
from tinygrad.device import Buffer, Device
from tinygrad.dtype import PtrDType, DType, dtypes
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import dedup, flatten
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.ops import BinaryOps
from tinygrad.renderer import Program
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.lazy import LazyBuffer

def _test_uop_result(inputs:List[Tensor], uops:List[UOp]):
  for x in inputs: x.realize()
  outbufs = [Buffer(Device.DEFAULT, 1, cast(DType,u.src[2].dtype)).allocate() for u in uops if u.op is UOps.STORE]
  inbufs = [cast(LazyBuffer,x.lazydata).base.buffer for x in inputs]
  src = Device[Device.DEFAULT].renderer.render("test", uops)
  ei = CompiledRunner(Program("test", src, Device.DEFAULT, uops=uops))
  ei.exec(outbufs+inbufs)
  return [np.frombuffer(x.as_buffer(), _to_np_dtype(x.dtype)) for x in outbufs]

def _get_uops_from_stores(stores:List[UOp]) -> List[UOp]:
  assert all(x.op is UOps.STORE for x in stores)
  # NOTE: we only toposort the stores
  uops: List[UOp] = []
  def _recursive_add(uop:UOp) -> List[UOp]: return flatten([_recursive_add(x) for x in uop.src])+[uop]
  uops = dedup(flatten(_recursive_add(st) for st in stores))
  return uops

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
    ret = _test_uop_result([Tensor([1])], _get_uops_from_stores([store]))[0]
    self.assertEqual(ret[0], 1)

  # simplified version of test_padto_where_multireduce
  def test_cast_out_of_scope(self):
    g = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), 0)
    c0 = UOp.const(dtypes.int, 0)
    c4 = UOp.const(dtypes.int, 4)
    acc0 = c4.alu(BinaryOps.ADD, c4)
    r0 = UOp(UOps.RANGE, dtypes.int, (c0, c4), (1, 0, False))
    alu0 = UOp(UOps.ALU, dtypes.int, (r0, acc0), BinaryOps.ADD)
    phi0 = UOp(UOps.PHI, dtypes.int, (acc0, alu0))
    cast0 = UOp(UOps.CAST, dtypes.int64, (acc0,))
    er0 = UOp(UOps.ENDRANGE, None, (r0,))
    gate0 = UOp(UOps.ALU, dtypes.bool, (acc0, c0), BinaryOps.CMPNE)
    # we want to have the IF come after the cast, but not actually dependent on it
    # (i.e we don't want to have the cast have multiple deps and thus become a var inside the range loop)
    if0 = UOp(UOps.IF, None, (gate0, cast0))
    store0 = UOp.store(g, c0, cast0)
    eif0 = UOp(UOps.ENDIF, None, (if0,))

    uops = [g, c0, c4, acc0, r0, alu0, phi0, cast0, er0, gate0, if0, store0, eif0]
    ret = _test_uop_result([Tensor([1])], uops)[0]
    self.assertEqual(ret[0], np.int32(14).astype(np.int64))

if __name__ == '__main__':
  unittest.main()
