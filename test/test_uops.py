from typing import Optional, Tuple, Any, List
import unittest, math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import CI, DEBUG, getenv
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.device import Buffer, Device
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, exec_alu
from tinygrad.renderer import Program
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import CompiledRunner, lower_schedule_item
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.codegen.uops import UOpGraph
from test.helpers import is_dtype_supported

def _uops_to_prg(uops_list, print=False):
  uops = UOpGraph()
  for l in uops_list: uops.add(l.uop, l.dtype, l.vin, l.arg)
  src = Device[Device.DEFAULT].renderer.render("test", uops)
  if print: uops.print()
  has_local = Device[Device.DEFAULT].renderer.has_local
  return CompiledRunner(Program("test", src, Device.DEFAULT, [1,1,1] if has_local else None, [1,1,1] if has_local else None, uops=uops))

def uop(uops:List[UOp], uop:UOps, dtype:Optional[DType], vin:Tuple[UOp, ...], arg:Any=None) -> UOp:
  uops.append(UOp(uop, dtype, tuple(vin), arg))
  return uops[-1]

def _test_single_value(vals, op, dts):
  uops = []
  output_dtype = dts[-1] if op is TernaryOps.WHERE else dtypes.bool if op is BinaryOps.CMPLT else dts[0]
  buf_store = uop(uops, UOps.DEFINE_GLOBAL, PtrDType(output_dtype), (), (0, True))
  buf_loads = [uop(uops, UOps.DEFINE_GLOBAL, PtrDType(dtype), (), (i+1, False)) for i,dtype in enumerate(dts)]
  loads = (uop(uops, UOps.LOAD, dtype, [buf_loads[i], uop(uops, UOps.CONST, dtypes.int32, (), 0)]) for i,dtype in enumerate(dts))
  alu = uop(uops, UOps.ALU, output_dtype, loads, op)
  uop(uops, UOps.STORE, None, (buf_store, uop(uops, UOps.CONST, dtypes.int32, (), 0), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  buf2 = [Buffer(Device.DEFAULT, 1, dtype).allocate().copyin(np.array([a], dtype=dtype.np).data) for a,dtype in zip(vals, dts)]
  prg = _uops_to_prg(uops)
  prg.exec([buf]+buf2)
  ret = np.empty(1, output_dtype.np)
  buf.copyout(ret.data)
  return ret[0]

def _test_single_value_const(vals, op, dts):
  uops = []
  output_dtype = dts[-1] if op is TernaryOps.WHERE else dtypes.bool if op is BinaryOps.CMPLT else dts[0]
  buf_store = uop(uops, UOps.DEFINE_GLOBAL, PtrDType(output_dtype), (), (0, True))
  loads = (uop(uops, UOps.CONST, dtype, [], a) for a,dtype in zip(vals, dts))
  alu = uop(uops, UOps.ALU, output_dtype, loads, op)
  uop(uops, UOps.STORE, None, (buf_store, uop(uops, UOps.CONST, dtypes.int32, (), 0), alu))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  prg = _uops_to_prg(uops)
  prg.exec([buf])
  ret = np.empty(1, output_dtype.np)
  buf.copyout(ret.data)
  return ret[0]

def _test_uops_result(output_dtype, uops, res):
  # uops = []
  buf_store = uop(uops, UOps.DEFINE_GLOBAL, PtrDType(output_dtype), (), (0, True))
  # res = output_fn(uops)
  uop(uops, UOps.STORE, None, (buf_store, uop(uops, UOps.CONST, dtypes.int32, (), 0), res))
  buf = Buffer(Device.DEFAULT, 1, output_dtype).allocate()
  prg = _uops_to_prg(uops, print=True)
  prg.exec([buf])
  ret = np.empty(1, output_dtype.np)
  buf.copyout(ret.data)
  return ret[0]

class TestUOps(unittest.TestCase):
  def _equal(self, v1, v2):
    assert isinstance(v2, (float, int, bool))
    if isinstance(v2, float):
      np.testing.assert_allclose(v1, v2, rtol=2e-7)
    else:
      np.testing.assert_equal(v1, v2)

  def _test_uop_fxn(self, op, fxn, dts=(dtypes.float32, )):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        a = dtypes.as_const(a, dts[0])
        self._equal(f([a], op, dts), fxn(a))

  def _test_bop_fxn(self, op, fxn, dts=(dtypes.float32, )*2, no_b_zero=False, no_b_neg=False):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0.0, 1.0]:
        for b in [-3.0, 1.0] + ([] if no_b_zero else [0.0]):
          a = dtypes.as_const(a, dts[0])
          b = dtypes.as_const(abs(b) if no_b_neg else b, dts[1])
          self._equal(f([a,b], op, dts), fxn(a,b))

  def _test_top_fxn(self, op, fxn, dts=(dtypes.float32, )*3):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [-2.0, 0, 1]:
        for b in [-3.0, 3.0]:
          for c in [-4.0, 4.0]:
            a = dtypes.as_const(a, dts[0])
            b = dtypes.as_const(b, dts[1])
            c = dtypes.as_const(c, dts[2])
            self._equal(f([a,b,c], op, dts), fxn(a,b,c))

class TestFloatUOps(TestUOps):
  def test_neg(self): self._test_uop_fxn(UnaryOps.NEG, lambda a: -a)
  def test_exp2(self): self._test_uop_fxn(UnaryOps.EXP2, lambda a: np.exp2(a))
  def test_log2(self): self._test_uop_fxn(UnaryOps.LOG2, lambda a: math.log2(a) if a > 0 else float('-inf' if a==0 else 'nan'))
  def test_sin(self): self._test_uop_fxn(UnaryOps.SIN, lambda a: math.sin(a))
  def test_sqrt(self): self._test_uop_fxn(UnaryOps.SQRT, lambda a: math.sqrt(a) if a >= 0 else float('nan'))
  def test_recip(self): self._test_uop_fxn(UnaryOps.RECIP, lambda a: 1/a if a != 0 else float('inf'))

  def test_add(self): self._test_bop_fxn(BinaryOps.ADD, lambda a,b: a+b)
  def test_sub(self): self._test_bop_fxn(BinaryOps.SUB, lambda a,b: a-b)
  def test_mul(self): self._test_bop_fxn(BinaryOps.MUL, lambda a,b: a*b)
  def test_max(self): self._test_bop_fxn(BinaryOps.MAX, lambda a,b: max(a,b))
  def test_cmplt(self): self._test_bop_fxn(BinaryOps.CMPLT, lambda a,b: a<b)
  # MOD isn't tested on floats

  def test_where(self):
    self._test_top_fxn(TernaryOps.WHERE, lambda a,b,c: b if a!=0 else c, (dtypes.bool, dtypes.float, dtypes.float))

class TestNonFloatUOps(TestUOps):
  def test_neg_int32(self): self._test_uop_fxn(UnaryOps.NEG, lambda a: -a, (dtypes.int32, ))
  def test_add_int32(self): self._test_bop_fxn(BinaryOps.ADD, lambda a,b: int(a)+int(b), (dtypes.int32, dtypes.int32))
  def test_sub_int32(self): self._test_bop_fxn(BinaryOps.SUB, lambda a,b: int(a)-int(b), (dtypes.int32, dtypes.int32))
  def test_mul_int32(self): self._test_bop_fxn(BinaryOps.MUL, lambda a,b: int(a)*int(b), (dtypes.int32, dtypes.int32))
  @unittest.skipUnless(getenv("PTX"), "only ptx uses bitshifts")
  def test_shr_int32(self): self._test_bop_fxn(BinaryOps.SHR, lambda a,b: int(a)>>int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  @unittest.skipUnless(getenv("PTX"), "only ptx uses bitshifts")
  def test_shl_int32(self): self._test_bop_fxn(BinaryOps.SHL, lambda a,b: int(a)<<int(b), (dtypes.int32, dtypes.int32), no_b_neg=True)
  def test_div_int32(self):
    self._test_bop_fxn(BinaryOps.IDIV, lambda a,b: int(a/b), (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_mod_int32(self):
    self._test_bop_fxn(BinaryOps.MOD,
                       lambda a,b: abs(int(a))%abs(int(b))*(1,-1)[a<0], (dtypes.int32, dtypes.int32), no_b_zero=True)
  def test_cmplt_int32(self): self._test_bop_fxn(BinaryOps.CMPLT, lambda a,b: float(a<b), (dtypes.int32, dtypes.int32))
  @unittest.skipUnless(is_dtype_supported(dtypes.bool), "dtype not supported")
  def test_mul_bool(self): self._test_bop_fxn(BinaryOps.MUL, lambda a,b: bool(a) and bool(b), (dtypes.bool, dtypes.bool))
  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "dtype not supported")
  def test_where_float16(self):
    self._test_top_fxn(TernaryOps.WHERE, lambda a,b,c: b if a!=0 else c, (dtypes.bool, dtypes.float16, dtypes.float16))

class TestBoolUOps(TestUOps):
  def _test_uop_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        self._equal(f([a], op, (dtypes.bool, )*1), fxn(a))

  def _test_bop_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        for b in [False, True]:
          self._equal(f([a,b], op, (dtypes.bool, )*2), fxn(a,b))

  def _test_top_bool_fxn(self, op, fxn):
    for f in [_test_single_value, _test_single_value_const]:
      for a in [False, True]:
        for b in [False, True]:
          for c in [False, True]:
            self._equal(f([a,b,c], op, (dtypes.bool, )*3), fxn(a,b,c))

  def test_not_bool(self): self._test_uop_bool_fxn(UnaryOps.NEG, lambda a: not a)
  def test_add_bool(self): self._test_bop_bool_fxn(BinaryOps.ADD, lambda a,b: a or b)
  def test_mul_bool(self): self._test_bop_bool_fxn(BinaryOps.MUL, lambda a,b: a and b)
  def test_xor_bool(self): self._test_bop_bool_fxn(BinaryOps.XOR, lambda a,b: a != b)
  def test_cmpne_bool(self): self._test_bop_bool_fxn(BinaryOps.CMPNE, lambda a,b: a != b)
  def test_cmplt_bool(self): self._test_bop_bool_fxn(BinaryOps.CMPLT, lambda a,b: a < b)
  def test_where_bool(self): self._test_top_bool_fxn(TernaryOps.WHERE, lambda a,b,c: b if a else c)

class TestExecALU(TestUOps):
  def test_sqrt(self):
    self.assertEqual(exec_alu(UnaryOps.SQRT, dtypes.float, (0.0,)), 0.0)

  def test_recip(self):
    np.testing.assert_allclose(exec_alu(UnaryOps.RECIP, dtypes.float32, (3.0,)), 1.0/3.0)
    np.testing.assert_allclose(exec_alu(UnaryOps.RECIP, dtypes.float32, (-3.0,)), 1.0/-3.0)

  def test_div(self):
    self.assertEqual(exec_alu(BinaryOps.IDIV, dtypes.int8, (8, 2)), 4)
    self.assertEqual(exec_alu(BinaryOps.IDIV, dtypes.int8, (7, 3)), 2)
    self.assertEqual(exec_alu(BinaryOps.IDIV, dtypes.int8, (7, -3)), -2)
    self.assertEqual(exec_alu(BinaryOps.IDIV, dtypes.int8, (-50, 6)), -8)

    np.testing.assert_allclose(exec_alu(BinaryOps.MUL, dtypes.float32, (8.0, exec_alu(UnaryOps.RECIP, dtypes.float32, (2.0,)))), 4.0)
    np.testing.assert_allclose(exec_alu(BinaryOps.MUL, dtypes.float32, (7.0, exec_alu(UnaryOps.RECIP, dtypes.float32, (3.0,)))), 2+(1.0/3.0))
    np.testing.assert_allclose(exec_alu(BinaryOps.MUL, dtypes.float32, (7.0, exec_alu(UnaryOps.RECIP, dtypes.float32, (-3.0,)))), -2-(1.0/3.0))

  def test_bool_neg(self):
    self.assertEqual(exec_alu(UnaryOps.NEG, dtypes.bool, (False,)), True)
    self.assertEqual(exec_alu(UnaryOps.NEG, dtypes.bool, (True,)), False)

  def test_bool_cmplt(self):
    self.assertEqual(exec_alu(BinaryOps.CMPLT, dtypes.bool, (False, False)), False)
    self.assertEqual(exec_alu(BinaryOps.CMPLT, dtypes.bool, (False, True)), True)
    self.assertEqual(exec_alu(BinaryOps.CMPLT, dtypes.bool, (True, False)), False)
    self.assertEqual(exec_alu(BinaryOps.CMPLT, dtypes.bool, (True, True)), False)

  def test_bool_where(self):
    self.assertEqual(exec_alu(TernaryOps.WHERE, dtypes.bool, (False, False, False)), False)
    self.assertEqual(exec_alu(TernaryOps.WHERE, dtypes.int, (False, 2, 4)), 4)
    np.testing.assert_allclose(exec_alu(TernaryOps.WHERE, dtypes.float, (False, 2.2, 4.5)), 4.5)

  def test_overflow(self):
    self.assertEqual(exec_alu(BinaryOps.ADD, dtypes.uint8, (250, 250)), 244)
    self.assertEqual(exec_alu(BinaryOps.ADD, dtypes.uint8, (256, 0)), 0)
    self.assertEqual(exec_alu(BinaryOps.SUB, dtypes.uint8, (0, 1)), 255)
    self.assertEqual(exec_alu(BinaryOps.SUB, dtypes.uint8, (0, 1000)), 24)

    self.assertEqual(exec_alu(BinaryOps.ADD, dtypes.int8, (127, 0)), 127)
    self.assertEqual(exec_alu(BinaryOps.ADD, dtypes.int8, (-128, 0)), -128)
    self.assertEqual(exec_alu(BinaryOps.SUB, dtypes.int8, (-100, 100)), 56)
    self.assertEqual(exec_alu(BinaryOps.SUB, dtypes.int8, (-1000, 0)), 24)
    self.assertEqual(exec_alu(BinaryOps.SUB, dtypes.int8, (-130, 0)), 126)

    self.assertEqual(exec_alu(BinaryOps.ADD, dtypes.int8, (1, 1)), 2)
    self.assertEqual(exec_alu(BinaryOps.ADD, dtypes.int8, (-128, 0)), -128)

class TestConstantFolding(unittest.TestCase):
  def test_cast_const(self):
    t = Tensor(1, dtype=dtypes.float).cast(dtypes.int)
    si = create_schedule([t.lazydata])
    assert len(si) == 0

  def test_bitcast_const(self):
    t = Tensor(1, dtype=dtypes.float).bitcast(dtypes.int)
    si = create_schedule([t.lazydata])
    assert len(si) == 1
    ji = lower_schedule_item(si[-1])
    assert any(uop.uop is UOps.BITCAST for uop in ji.prg.p.uops), f"{[uop.uop for uop in ji.prg.p.uops]} does not contain bitcast"

class TestGatedStoreRewrite(unittest.TestCase):
  @unittest.skip("not yet implemented")
  def test_wrap_store_parents(self):
    # wraps all store parents in the valid branch
    uops = UOpGraph()
    gmem = uops.add(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), (0, True))
    gidx0 = uops.add(UOps.SPECIAL, dtypes.int, (), (0, 'gidx0', 4))
    idx = gidx0 * UOp.const(dtypes.int, 2)
    value = uops.add(UOps.CONST, dtypes.float, (), 42.0)

    gate = uops.add(UOps.ALU, dtypes.bool, (gidx0, UOp.const(dtypes.int, 1)), arg=BinaryOps.CMPLT)
    uops.add(UOps.STORE, None, (gmem, idx, value, gate))
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render("test", uops))
    if_uop = next(u for u in uops if u.uop is UOps.IF)
    endif = next(u for u in uops if u.uop is UOps.ENDIF)
    assert endif.vin[0] is if_uop
    nested_uops = tuple(uops.uops[uops.uops.index(if_uop)+1:uops.uops.index(endif)])
    assert nested_uops == (gmem, gidx0, idx, value)

  @unittest.skip("not yet implemented")
  def test_wrap_some_parents(self):
    # some parents are used outside the branch
    uops = UOpGraph()
    gmem0 = uops.add(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), (0, True))
    gmem1 = uops.add(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), (), (1, True))
    gidx0 = uops.add(UOps.SPECIAL, dtypes.int, (), (0, 'gidx0', 4))
    idx = gidx0 * UOp.const(dtypes.int, 2)
    value0 = uops.add(UOps.CONST, dtypes.float, (), 42.0)
    value1 = uops.add(UOps.CONST, dtypes.float, (), 43.0)

    gate = uops.add(UOps.ALU, dtypes.bool, (gidx0, UOp.const(dtypes.int, 1)), arg=BinaryOps.CMPLT)
    uops.add(UOps.STORE, None, (gmem0, idx, value0, gate))
    uops.add(UOps.STORE, None, (gmem1, idx, value1))
    if DEBUG >= 4: print(Device[Device.DEFAULT].renderer.render("test", uops))
    if_uop = next(u for u in uops if u.uop is UOps.IF)
    endif = next(u for u in uops if u.uop is UOps.ENDIF)
    assert endif.vin[0] is if_uop
    nested_uops = tuple(uops.uops[uops.uops.index(if_uop)+1:uops.uops.index(endif)])
    assert nested_uops == (gmem0, value0)

class TestLocalAccess(unittest.TestCase):
  # NOTE: this is failing on METAL CI, no idea why. Works locally.
  @unittest.skipIf(Device.DEFAULT == "METAL" and CI, "failing only in CI")
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  def test_local_basic(self):
    uops = []
    smem = uop(uops, UOps.DEFINE_LOCAL, PtrDType(dtypes.float32), (), ('smem', 16))
    st = uop(uops, UOps.STORE, None, (smem, uop(uops, UOps.CONST, dtypes.int32, (), 0), uop(uops, UOps.CONST, dtypes.float32, (), 42.0)))
    barr = uop(uops, UOps.BARRIER, None, (st,))
    sres = uop(uops, UOps.LOAD, dtypes.float32, (smem, uop(uops, UOps.CONST, dtypes.int32, (), 0), barr))
    self.assertEqual(_test_uops_result(dtypes.float32, uops, sres), 42)

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_shared, "test requires shared memory")
  def test_local_indirect(self):
    uops = []
    smem = uop(uops, UOps.DEFINE_LOCAL, PtrDType(dtypes.int32), (), ('smem', 16))
    st1 = uop(uops, UOps.STORE, None, (smem, uop(uops, UOps.CONST, dtypes.int32, (), 1), uop(uops, UOps.CONST, dtypes.int32, (), 2)))
    st2 = uop(uops, UOps.STORE, None, (smem, uop(uops, UOps.CONST, dtypes.int32, (), 2), uop(uops, UOps.CONST, dtypes.int32, (), 42)))
    barr = uop(uops, UOps.BARRIER, None, (st1,st2))
    ofs = uop(uops, UOps.LOAD, dtypes.int32, (smem, uop(uops, UOps.CONST, dtypes.int32, (), 1), barr))
    sres = uop(uops, UOps.LOAD, dtypes.int32, (smem, ofs))
    self.assertEqual(_test_uops_result(dtypes.int32, uops, sres), 42)

@unittest.skipUnless(Device.DEFAULT in {"CUDA"} and getenv("PTX"), "This only tests assembly backends")
class TestAssembly(unittest.TestCase):
  def test_bitshift_left(self):
    uops = UOpGraph()
    g1 = uops.add(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int32), (), (0, True))
    c1 = uops.add(UOps.CONST, dtypes.int, (), 2)
    c2 = uops.add(UOps.CONST, dtypes.int, (), 3)
    l1 = uops.add(UOps.LOAD, dtypes.int, (g1, c1))
    a1 = uops.add(UOps.ALU, dtypes.int, (l1, c1), BinaryOps.MUL)
    a2 = uops.add(UOps.ALU, dtypes.int, (l1, c2), BinaryOps.MUL)
    uops.add(UOps.SINK, None, (a1,a2))
    Device[Device.DEFAULT].renderer.render("test", uops)
    self.assertEqual(uops.uops[-1].arg, BinaryOps.MUL)
    self.assertEqual(uops.uops[-2].arg, BinaryOps.SHL)

  def test_bitshift_right(self):
    uops = UOpGraph()
    g1 = uops.add(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int32), (), (0, True))
    c1 = uops.add(UOps.CONST, dtypes.int, (), 2)
    c2 = uops.add(UOps.CONST, dtypes.int, (), 3)
    l1 = uops.add(UOps.LOAD, dtypes.int, (g1, c1))
    a1 = uops.add(UOps.ALU, dtypes.int, (l1, c1), BinaryOps.IDIV)
    a2 = uops.add(UOps.ALU, dtypes.int, (l1, c2), BinaryOps.IDIV)
    uops.add(UOps.SINK, None, (a1,a2))
    Device[Device.DEFAULT].renderer.render("test", uops)
    self.assertEqual(uops.uops[-1].arg, BinaryOps.IDIV)
    self.assertEqual(uops.uops[-2].arg, BinaryOps.SHR)

if __name__ == '__main__':
  unittest.main(verbosity=2)
