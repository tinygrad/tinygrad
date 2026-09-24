import unittest
import z3
from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, UOp, AxisType, exec_alu
from tinygrad.uop.validate import uops_to_z3, validate_index_with_z3
from test.helpers import to_uops_list

def Variable(name, nmin, nmax): return UOp.variable(name, nmin, nmax, param=True)

class TestValidateOOB(unittest.TestCase):
  """Test z3 validation of index bounds for different ALU ops and patterns."""

  # basic index patterns
  def test_const_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      to_uops_list([buf.index(UOp.const(0)).load()])  # valid
      to_uops_list([buf.index(UOp.const(15)).load()])  # valid (last element)
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.const(16)).load()])  # off by one
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.const(42)).load()])  # way out

  def test_variable_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      to_uops_list([buf.index(Variable("i", 0, 15)).load()])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("i", 0, 20)).load()])  # oob
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("i", -5, 10)).load()])  # negative

  def test_range_with_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      r = UOp.range(42, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r.valid(r < 16)).load()])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.valid(r < 17)).load()])  # oob

  def test_variable_with_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      v = Variable("v", -5, 80)
      to_uops_list([buf.index(v.valid((v >= 0) & (v < 16))).load()])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(v.valid(v < 20)).load()])  # negative not masked

  def test_gated_store(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      v = Variable("v", 0, 20)
      to_uops_list([buf.index(v.valid(v < 16)).store(0)])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(v.valid(v < 20)).store(0)])  # oob

  # ALU ops in index
  def test_floordiv(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      to_uops_list([buf.index(UOp.range(32, 0, AxisType.GLOBAL) // 2).load()])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.range(34, 0, AxisType.GLOBAL) // 2).load()])  # 0..16 oob

  def test_mod(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      r = UOp.range(100, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r % 16).load()])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r % 20).load()])  # 0..19 oob

  def test_shr(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      to_uops_list([buf.index(UOp.range(64, 0, AxisType.GLOBAL) >> 2).load()])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(UOp.range(128, 0, AxisType.GLOBAL) >> 2).load()])  # 0..31 oob

  def test_shl(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 64)
      r = UOp.range(8, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r << 2).load()])  # 0..28 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r << 4).load()])  # 0..112 oob

  def test_and(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      r = UOp.range(100, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r & 15).load()])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r & 31).load()])  # 0..31 oob
      # align masks round down to a multiple of 2^k
      to_uops_list([buf.index((r & -4).valid(r < 16)).load()])  # 0..12 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r & -2).load()])  # 0..100 oob
      # other masks can't be modeled as mod
      with self.assertRaisesRegex(RuntimeError, "z3 int AND only supports"):
        to_uops_list([buf.index(r & 21).load()])

  def test_max(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      to_uops_list([buf.index(Variable("v", -10, 15).maximum(0)).load()])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(Variable("v2", -10, 20).maximum(0)).load()])  # 0..20 oob

  def test_xor_in_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      r = UOp.range(32, 0, AxisType.GLOBAL)
      to_uops_list([buf.index(r.valid((r < 8) ^ ((r >= 8) & (r < 16)))).load()])  # 0..15 valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.valid((r < 10) ^ (r >= 20))).load()])  # 0..9,20..31 oob

  # cast patterns
  def test_float_cast_in_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      r = UOp.range(20, 0)
      i = (r.cast(dtypes.float) * 0.68).trunc().cast(dtypes.int)
      to_uops_list([buf.index(i.valid((i >= 0) & (i < 16))).load()])
      # a float entirely out of the int range has no value, not an empty one
      f = UOp.variable("f", 3e9, 4e9, dtypes.float32, param=True).cast(dtypes.int)
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(f).load()])

  def test_float_cast_in_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 1)
      r = UOp.range(20, 0)
      unknown = r.cast(dtypes.float).cast(dtypes.bool)  # a bool from a float is unconstrained
      to_uops_list([buf.index(r.valid((r < 1) & unknown)).load()])
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.valid(unknown)).load()])

  def test_bitcast_in_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 16)
      r = UOp.range(16, 0)
      # the WEBGPU shift: int -> uint, shift, back to int
      i = (r.cast(dtypes.int).bitcast(dtypes.uint) << UOp.const(1).cast(dtypes.uint)).bitcast(dtypes.int)
      to_uops_list([buf.index(i.valid(i < 16)).load()])
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(i).load()])  # 0..30 oob
      # a negative char reads as a large uchar
      c = Variable("c", -128, -113).cast(dtypes.char)
      to_uops_list([UOp.param(1, dtypes.int, 144).index(c.bitcast(dtypes.uchar).cast(dtypes.int)).load()])  # 128..143 valid
      # the bits of a float are any int
      with self.assertRaises(RuntimeError):
        to_uops_list([buf.index(r.cast(dtypes.float).bitcast(dtypes.int)).load()])

  def test_bool_cast_in_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf = UOp.param(0, dtypes.int, 1)
      r = UOp.range(20, 0)
      to_uops_list([buf.index(r.valid(r.cast(dtypes.bool).logical_not())).load()])  # only r=0 valid

  # load result as index/mask
  def test_load_as_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf0 = UOp.param(0, dtypes.int, 16)
      buf1 = UOp.param(1, dtypes.int, 64)
      r = UOp.range(42, 0, AxisType.GLOBAL)
      ld0 = buf0.index(r.valid(r < 8)).load().cast(dtypes.weakint)
      to_uops_list([buf1.index((ld0 * 2).valid((ld0 >= 0) & (ld0 < 32))).load()])  # valid
      with self.assertRaises(RuntimeError):
        to_uops_list([buf1.index((ld0 * 2).valid((ld0 >= 0) & (ld0 < 64))).load()])  # oob

  def test_load_from_shrink_as_index(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf0 = UOp.param(0, dtypes.int, 16)
      buf1 = UOp.param(1, dtypes.int, 64)
      shrink = UOp(Ops.SHRINK, src=(buf0, UOp.const(0, dtypes.int), UOp.const(4)))
      ld0 = shrink.load().index(0)
      to_uops_list([buf1.index(ld0.valid((ld0 >= 0) & (ld0 < 64))).load()])

  def test_load_bool_as_mask(self):
    with Context(CHECK_OOB=1, SPEC=2):
      buf_bool = UOp.param(0, dtypes.bool, 16)
      buf_int = UOp.param(1, dtypes.int, 8)
      gidx = UOp(Ops.SPECIAL, src=(UOp.const(16),), arg="gidx0")
      ld_bool = buf_bool.index(gidx).load()
      with self.assertRaises(RuntimeError):
        to_uops_list([buf_int.index(gidx.valid(ld_bool)).load()])  # gidx 0..15, buf_int size 8

  # local memory
  def test_gated_local(self):
    with Context(CHECK_OOB=1, SPEC=2):
      gbuf = UOp.param(0, dtypes.uint, 400)
      sbuf = UOp.placeholder((8,), dtypes.uint, slot=0, addrspace=AddrSpace.LOCAL)
      gidx = UOp(Ops.SPECIAL, src=(UOp.const(416),), arg="gidx0")
      lidx = UOp(Ops.SPECIAL, src=(UOp.const(10),), arg="lidx0")
      store = sbuf.index(lidx.valid(lidx < 8)).store(UOp.const(1))
      load = sbuf.after(store).index(lidx.valid(lidx < 8)).load()
      to_uops_list([gbuf.index(gidx.valid(gidx < 400)).store(load)])  # valid: local store and load gated to 8, global store gated to 400
      with self.assertRaises(RuntimeError):
        to_uops_list([gbuf.index(gidx.valid(gidx < 400)).store(sbuf.after(store).index(lidx).load())])  # lidx 0..9 into 8
      with self.assertRaises(RuntimeError):
        to_uops_list([gbuf.index(gidx).store(load)])  # gidx 0..415 into 400

class TestZ3Shifts(unittest.TestCase):
  def test_integer_shifts(self):
    for dtype in dtypes.ints:
      a = UOp.variable("a", dtype.min, dtype.max, dtype, param=True)
      b = Variable("b", 0, 2*dtype.bitsize)
      values = {dtype.min, dtype.min+1, 0, 1, 3, dtype.max-1, dtype.max}
      if dtype in dtypes.sints: values.update((-3, -1))
      for op in (Ops.SHL, Ops.SHR):
        solver = z3.Solver(ctx=z3.Context())
        result, za, zb = uops_to_z3(solver, a.alu(op, b), a, b)
        for count in (0, 1, dtype.bitsize-1, dtype.bitsize, dtype.bitsize+1, 2*dtype.bitsize):
          constant, = uops_to_z3(solver, a.alu(op, UOp.const(count)))
          for value in values:
            with self.subTest(dtype=str(dtype), op=op.name, value=str(value), count=str(count)):
              # The bounds checker models mathematical integers, like its ADD/MUL and the original constant shifts.
              expected = exec_alu(op, dtype, [value, count], truncate_output=False)
              actual = z3.simplify(z3.substitute(result, (za, z3.IntVal(value, ctx=solver.ctx)), (zb, z3.IntVal(count, ctx=solver.ctx))))
              self.assertEqual(actual.as_long(), expected)
              self.assertEqual(z3.simplify(z3.substitute(constant, (za, z3.IntVal(value, ctx=solver.ctx)))).as_long(), expected)

  def test_lowered_shift_count(self):
    n = UOp.variable("n", 0, 63, dtypes.uint32, param=True)
    idx = UOp.const(65535, dtypes.uint64).alu(Ops.SHR, n)
    self.assertTrue(validate_index_with_z3(65536, idx, UOp.const(True)))
    self.assertFalse(validate_index_with_z3(65535, idx, UOp.const(True)))

  def test_weak_shifts_do_not_truncate(self):
    a = Variable("a", -(2**1200), 2**1200+7)
    b = Variable("b", 0, 1024)
    for op in (Ops.SHL, Ops.SHR):
      solver = z3.Solver(ctx=z3.Context())
      result, za, zb = uops_to_z3(solver, a.alu(op, b), a, b)
      for value in (-2**1200, -1, 0, 7, 2**1200+7):
        for count in (0, 1, 63, 64, 65, 129, 1024):
          with self.subTest(op=op.name, value=str(value), count=str(count)):
            actual = z3.simplify(z3.substitute(result, (za, z3.IntVal(value, ctx=solver.ctx)), (zb, z3.IntVal(count, ctx=solver.ctx))))
            self.assertEqual(actual.as_long(), exec_alu(op, dtypes.weakint, [value, count]))

  def test_symbolic_shift_bounds(self):
    n = Variable("n", 0, 3)
    for shift in (n, n.eq(0).where(0, n.eq(1).where(1, n.eq(2).where(2, 3)))):
      with self.subTest(shift=shift.render()):
        for dtype in (*dtypes.ints, dtypes.weakint):
          right = UOp.const(127, dtype) >> shift
          self.assertTrue(validate_index_with_z3(128, right, UOp.const(True)))
          self.assertFalse(validate_index_with_z3(127, right, UOp.const(True)))
          left = UOp.const(1, dtype) << shift
          self.assertTrue(validate_index_with_z3(9, left, UOp.const(True)))
          self.assertFalse(validate_index_with_z3(8, left, UOp.const(True)))

  def test_loaded_shift(self):
    shift = UOp.param(0, dtypes.uint32, 1).index(0).load() & 31
    idx = UOp.const(256, dtypes.uint32) >> shift
    self.assertTrue(validate_index_with_z3(257, idx, UOp.const(True)))
    self.assertFalse(validate_index_with_z3(256, idx, UOp.const(True)))

  def test_negative_shift_is_not_assumed_safe(self):
    n = Variable("n", -1, 3)
    for op in (Ops.SHL, Ops.SHR):
      idx = UOp.const(1, dtypes.int32).alu(op, n)
      self.assertFalse(validate_index_with_z3(9, idx, UOp.const(True)))
      self.assertTrue(validate_index_with_z3(9, idx, n >= 0))

if __name__ == "__main__":
  unittest.main()
