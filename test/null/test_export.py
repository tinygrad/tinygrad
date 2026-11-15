import unittest
from tinygrad import Tensor, Variable, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, ParamArg
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.function import function
import tinygrad.schedule as schedule

def exported(*tensors:Tensor) -> tuple[UOp, tuple[UOp, ...]]:
  # capture the exported function of scheduling these tensors
  got:list[tuple[UOp, tuple[UOp, ...]]] = []
  orig = schedule.export
  def spy(big_sink:UOp):
    got.append(ret:=orig(big_sink))
    return ret
  schedule.export = spy
  try: Tensor.linear_with_vars(*tensors)
  finally: schedule.export = orig
  assert len(got) == 1, f"expected one export, got {len(got)}"
  return got[0]

def buf(n=4, dtype=dtypes.float32) -> Tensor: return Tensor.empty(n, dtype=dtype, device="NULL").realize()

class TestExport(unittest.TestCase):
  def assert_exportable(self, fn:UOp, externs:tuple[UOp, ...]):
    for u in fn.toposort(enter_calls=True):
      self.assertFalse(u.op is Ops.BUFFER and u.addrspace is AddrSpace.GLOBAL)
      if isinstance(u.arg, ParamArg): self.assertTrue(u.arg.buffer is None and u.arg.val is None, f"runtime state in {u}")
      if u.op is Ops.ALLOC: self.assertLess(u.arg.slot, 0, "ALLOCs are canonical")
      if u.op is Ops.EXTERN: self.assertEqual((u.dtype, u.device), (externs[u.arg.slot].dtype, externs[u.arg.slot].device))
    # UOps are hash-consed, so a lossless round trip through the wire format gives back the identical UOp
    self.assertIs(parse_uir(render_uir(fn)), fn)

  def test_add(self):
    a, b = buf(), buf()
    fn, externs = exported(a+b)
    self.assert_exportable(fn, externs)
    self.assertTrue({a.uop.buf_uop, b.uop.buf_uop} <= set(externs))
    self.assertNotIn(Ops.BUFFER, {u.op for u in fn.toposort()})

  def test_same_function_different_buffers(self):
    a, b, c, d = buf(), buf(), buf(), buf()
    self.assertIs(exported(a+b)[0], exported(c+d)[0])

  def test_different_function(self):
    a, b = buf(), buf()
    self.assertIsNot(exported(a+b)[0], exported(a*b)[0])

  def test_variable_loses_value(self):
    a = buf()
    v = Variable("v", 1, 4)
    fn, externs = exported(a*v.bind(2))
    self.assert_exportable(fn, externs)
    self.assertIn(v.bind(2), externs)
    self.assertIs(fn, exported(a*v.bind(3))[0])
    self.assertIsNot(fn, exported(a*Variable("v", 1, 5).bind(3))[0])

  def test_symbolic_shape(self):
    a = Tensor.empty(8, device="NULL").realize()
    v = Variable("v", 1, 8)
    fn, externs = exported(a[:v.bind(3)].contiguous())
    self.assert_exportable(fn, externs)
    self.assertIs(fn, exported(a[:v.bind(5)].contiguous())[0])

  def test_view_input(self):
    a = Tensor.empty(8, device="NULL").realize()
    self.assert_exportable(*exported(a[2:6].contiguous()+1))

  def test_bitcast_input(self):
    self.assert_exportable(*exported(buf().bitcast(dtypes.int32)+1))

  def test_assign(self):
    a = buf()
    fn, externs = exported(a.assign(a+1))
    self.assert_exportable(fn, externs)
    self.assertEqual(externs, (a.uop.buf_uop,))

  def test_rand(self):
    self.assert_exportable(*exported(Tensor.rand(4, device="NULL")))

  def test_multi(self):
    a = Tensor.empty(4, 4, device="NULL").shard(("NULL:0", "NULL:1"), axis=0).realize()
    fn, externs = exported(a*2)
    self.assert_exportable(fn, externs)
    self.assertTrue(any(isinstance(e.device, tuple) for e in externs))

  def test_function(self):
    @function
    def f(x, y): return (x*y+1).contiguous()
    fn, externs = exported(f(buf(), buf()))
    self.assert_exportable(fn, externs)
    self.assertIn(Ops.CALL, {u.op for u in fn.toposort()})

  def test_precompiled_function(self):
    @function(precompile=True)
    def f(x, y): return (x*y+1).contiguous()
    self.assert_exportable(*exported(f(buf(), buf())))

  def test_alloc_is_canonical(self):
    @function
    def f(x): return (x+1).contiguous()*2
    self.assertIs(exported(f(buf()))[0], exported(f(buf()))[0])

class TestExportCache(unittest.TestCase):
  def test_cached(self):
    v = Variable("v", 1, 4)
    a, b, c, d = buf(), buf(), buf(), buf()
    (a+b).contiguous().mul(v.bind(2)).linear_with_vars()
    n = len(schedule.export_cache)
    linear, var_vals = (c+d).contiguous().mul(v.bind(3)).linear_with_vars()
    self.assertEqual(len(schedule.export_cache), n)
    self.assertEqual(var_vals, {"v": 3})
    # the cached linear is mapped back to this storage
    self.assertTrue({c.uop.buf_uop, d.uop.buf_uop} <= set(linear.toposort()))
    self.assertNotIn(Ops.EXTERN, {u.op for u in linear.toposort()})
    self.assertNotIn(Ops.ALLOC, {u.op for u in linear.toposort()})

  def test_no_runtime_state(self):
    Tensor.rand(4, device="NULL").add(buf()).contiguous().sum().linear_with_vars()
    for linear in schedule.export_cache.values():
      for u in linear.toposort(enter_calls=True):
        self.assertIsNot(u.op, Ops.BUFFER)
        self.assertFalse(u.is_bound_var)

if __name__ == "__main__":
  unittest.main()
