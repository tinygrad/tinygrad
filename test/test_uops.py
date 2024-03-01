from typing import Optional, Tuple, Any, List
import unittest, math
import numpy as np
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.helpers import getenv
from tinygrad.device import Buffer, Device
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.device import CompiledASTRunner, Compiled
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.runtime.ops_python import exec_alu
from test.test_dtype import is_dtype_supported

def _uops_to_prg(uops):
  src = Device[Device.DEFAULT].compiler.render("test", uops)
  has_local = Device[Device.DEFAULT].compiler.linearizer_opts.has_local
  return CompiledASTRunner("test", src, Device[Device.DEFAULT], [1] if has_local else None, [1] if has_local else None)

def uop(uops:List[UOp], uop:UOps, dtype:Optional[DType], vin:Tuple[UOp, ...], arg:Any=None) -> UOp:
  uops.append(UOp(uop, dtype, tuple(vin), arg))
  return uops[-1]

def _test_single_value(vals, op, dts):
  uops = []
  output_dtype = dts[-1] if op is TernaryOps.WHERE else dtypes.bool if op is BinaryOps.CMPLT else dts[0]
  buf_store = uop(uops, UOps.DEFINE_GLOBAL, PtrDType(output_dtype), (), (0, 'data0'))
  buf_loads = [uop(uops, UOps.DEFINE_GLOBAL, PtrDType(dtype), (), (i+1, f'data{i+1}')) for i,dtype in enumerate(dts)]
  loads = (uop(uops, UOps.LOAD, dtype, [buf_loads[i], uop(uops, UOps.CONST, dtypes.int32, (), 0)]) for i,dtype in enumerate(dts))
  alu = uop(uops, UOps.ALU, output_dtype, loads, op)
  uop(uops, UOps.STORE, None, (buf_store, uop(uops, UOps.CONST, dtypes.int32, (), 0), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype)
  buf2 = [Buffer(Device.DEFAULT, 1, dtype).copyin(np.array([a], dtype=dtype.np).data) for a,dtype in zip(vals, dts)]
  prg = _uops_to_prg(uops)
  prg.exec([buf]+buf2)
  ret = np.empty(1, output_dtype.np)
  buf.copyout(ret.data)
  return ret[0]

def _test_single_value_const(vals, op, dts):
  uops = []
  output_dtype = dts[-1] if op is TernaryOps.WHERE else dtypes.bool if op is BinaryOps.CMPLT else dts[0]
  buf_store = uop(uops, UOps.DEFINE_GLOBAL, PtrDType(output_dtype), (), (0, 'data0'))
  loads = (uop(uops, UOps.CONST, dtype, [], a) for a,dtype in zip(vals, dts))
  alu = uop(uops, UOps.ALU, output_dtype, loads, op)
  uop(uops, UOps.STORE, None, (buf_store, uop(uops, UOps.CONST, dtypes.int32, (), 0), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype)
  prg = _uops_to_prg(uops)
  prg.exec([buf])
  ret = np.empty(1, output_dtype.np)
  buf.copyout(ret.data)
  return ret[0]

class TestUOps(unittest.TestCase):
  def _equal(self, v1, v2):
    if not (math.isnan(v1) and math.isnan(v2)): self.assertAlmostEqual(v1, v2, places=5) if v1.dtype != np.bool_ else self.assertEqual(v1, v2)

  def _test_uop_fxn(self, op, fxn, dts=(PtrDType(dtypes.float32), )):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        self._equal(f([a], op, dts), fxn(a))

  def _test_bop_fxn(self, op, fxn, dts=(PtrDType(dtypes.float32), )*2, no_b_zero=False):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        for b in [-3.0, 1.0] + ([] if no_b_zero else [0.0]):
          self._equal(f([a,b], op, dts), fxn(a,b))

  def _test_top_fxn(self, op, fxn, dts=(PtrDType(dtypes.float32), )*3):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0, 1]:
        for b in [-3.0, 3.0]:
          for c in [-4.0, 4.0]:
            self._equal(f([a,b,c], op, dts), fxn(a,b,c))

@unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled), "only test for compiled backends")
class TestFloatUOps(TestUOps):
  def test_neg(self): self._test_uop_fxn(UnaryOps.NEG, lambda a: -a)
  def test_exp2(self): self._test_uop_fxn(UnaryOps.EXP2, lambda a: np.exp2(a))
  def test_log2(self): self._test_uop_fxn(UnaryOps.LOG2, lambda a: math.log2(a) if a > 0 else float('-inf' if a==0 else 'nan'))
  def test_sin(self): self._test_uop_fxn(UnaryOps.SIN, lambda a: math.sin(a))
  def test_sqrt(self): self._test_uop_fxn(UnaryOps.SQRT, lambda a: math.sqrt(a) if a >= 0 else float('nan'))

  def test_add(self): self._test_bop_fxn(BinaryOps.ADD, lambda a,b: a+b)
  def test_sub(self): self._test_bop_fxn(BinaryOps.SUB, lambda a,b: a-b)
  def test_mul(self): self._test_bop_fxn(BinaryOps.MUL, lambda a,b: a*b)
  def test_div(self): self._test_bop_fxn(BinaryOps.DIV, lambda a,b: a/b if b != 0 else a*float('inf'))
  def test_max(self): self._test_bop_fxn(BinaryOps.MAX, lambda a,b: max(a,b))
  def test_cmplt(self): self._test_bop_fxn(BinaryOps.CMPLT, lambda a,b: a<b)
  # MOD isn't tested on floats

  def test_where(self):
    self._test_top_fxn(TernaryOps.WHERE, lambda a,b,c: b if a!=0 else c, (PtrDType(dtypes.bool), PtrDType(dtypes.float), PtrDType(dtypes.float)))

# TODO: fix this on all the backends
@unittest.skipIf(not isinstance(Device[Device.DEFAULT], Compiled) or getenv('ARM64', False), "only test for compiled backends, broken on some")
class TestNonFloatUOps(TestUOps):
  def test_neg_int32(self): self._test_uop_fxn(UnaryOps.NEG, lambda a: -a, (PtrDType(dtypes.int32), ))
  def test_add_int32(self): self._test_bop_fxn(BinaryOps.ADD, lambda a,b: int(a)+int(b), (PtrDType(dtypes.int32), PtrDType(dtypes.int32)))
  def test_sub_int32(self): self._test_bop_fxn(BinaryOps.SUB, lambda a,b: int(a)-int(b), (PtrDType(dtypes.int32), PtrDType(dtypes.int32)))
  def test_mul_int32(self): self._test_bop_fxn(BinaryOps.MUL, lambda a,b: int(a)*int(b), (PtrDType(dtypes.int32), PtrDType(dtypes.int32)))
  def test_div_int32(self):
    self._test_bop_fxn(BinaryOps.DIV, lambda a,b: int(a/b), (PtrDType(dtypes.int32), PtrDType(dtypes.int32)), no_b_zero=True)
  def test_mod_int32(self):
    self._test_bop_fxn(BinaryOps.MOD,
                       lambda a,b: abs(int(a))%abs(int(b))*(1,-1)[a<0], (PtrDType(dtypes.int32), PtrDType(dtypes.int32)), no_b_zero=True)
  def test_cmplt_int32(self): self._test_bop_fxn(BinaryOps.CMPLT, lambda a,b: float(a<b), (PtrDType(dtypes.int32), PtrDType(dtypes.int32)))
  @unittest.skipUnless(is_dtype_supported(dtypes.bool), "dtype not supported")
  def test_mul_bool(self): self._test_bop_fxn(BinaryOps.MUL, lambda a,b: bool(a) and bool(b), (PtrDType(dtypes.bool), PtrDType(dtypes.bool)))
  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "dtype not supported")
  def test_where_float16(self):
    self._test_top_fxn(TernaryOps.WHERE, lambda a,b,c: b if a!=0 else c, (PtrDType(dtypes.bool), PtrDType(dtypes.float16), PtrDType(dtypes.float16)))

class TestExecALU(TestUOps):
  def test_sqrt(self):
    self.assertEqual(exec_alu(UnaryOps.SQRT, dtypes.int, (0,)), 0)

if __name__ == '__main__':
  unittest.main(verbosity=2)
