import unittest
from tinygrad import Tensor, Variable, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp
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
    schedule.assert_exportable(fn)
    # UOps are hash-consed, so a lossless round trip through the wire format gives back the identical UOp
    self.assertIs(parse_uir(render_uir(fn)), fn)
    ext = sorted({u.arg.slot:u for u in fn.toposort(enter_calls=True) if u.op is Ops.EXTERN}.items())
    self.assertEqual([s for s,_ in ext], list(range(len(externs))), "EXTERN slots are dense")
    for (_, e), x in zip(ext, externs):
      self.assertIsNone(e.arg.name)
      self.assertEqual((e.dtype, e.addrspace, e.device), (x.dtype, x.addrspace, x.device))

  def test_add(self):
    a, b = buf(), buf()
    fn, externs = exported(a+b)
    self.assert_exportable(fn, externs)
    self.assertEqual(len(externs), 3)
    self.assertTrue(all(e.op is Ops.BUFFER and e.arg.buffer is not None for e in externs))
    self.assertTrue({a.uop.buf_uop, b.uop.buf_uop} <= set(externs))
    self.assertNotIn(Ops.BUFFER, {u.op for u in fn.toposort()})

  def test_same_function_different_buffers(self):
    a, b, c, d = buf(), buf(), buf(), buf()
    self.assertIs(exported(a+b)[0], exported(c+d)[0])

  def test_different_function(self):
    a, b = buf(), buf()
    self.assertIsNot(exported(a+b)[0], exported(a*b)[0])

  def test_variable_loses_value_and_name(self):
    a = buf()
    v, w = Variable("v", 1, 4), Variable("w", 1, 4)
    fn, externs = exported(a*v.bind(2))
    self.assert_exportable(fn, externs)
    ext = [u for u in fn.toposort() if u.op is Ops.EXTERN and u.arg.vmin_vmax is not None]
    self.assertEqual(len(ext), 1)
    self.assertEqual(ext[0].arg.vmin_vmax, (1, 4))
    self.assertIn(v.bind(2), externs)
    # the value and the name are not part of the function, the range is
    self.assertIs(fn, exported(a*v.bind(3))[0])
    self.assertIs(fn, exported(a*w.bind(3))[0])
    self.assertIsNot(fn, exported(a*Variable("v", 1, 5).bind(3))[0])

  def test_symbolic_shape(self):
    a = Tensor.empty(8, device="NULL").realize()
    v = Variable("v", 1, 8)
    fn, externs = exported(a[:v.bind(3)].contiguous())
    self.assert_exportable(fn, externs)
    self.assertIs(fn, exported(a[:v.bind(5)].contiguous())[0])

  def test_view_input(self):
    a = Tensor.empty(8, device="NULL").realize()
    fn, externs = exported(a[2:6].contiguous()+1)
    self.assert_exportable(fn, externs)

  def test_bitcast_input(self):
    a = buf()
    fn, externs = exported(a.bitcast(dtypes.int32)+1)
    self.assert_exportable(fn, externs)

  def test_assign(self):
    a = buf()
    fn, externs = exported(a.assign(a+1))
    self.assert_exportable(fn, externs)
    self.assertEqual(externs, (a.uop.buf_uop,))

  def test_rand(self):
    fn, externs = exported(Tensor.rand(4, device="NULL"))
    self.assert_exportable(fn, externs)

  def test_multi(self):
    a = Tensor.empty(4, 4, device="NULL").shard(("NULL:0", "NULL:1"), axis=0).realize()
    fn, externs = exported(a*2)
    self.assert_exportable(fn, externs)
    self.assertTrue(any(isinstance(e.device, tuple) for e in externs))

  def test_function(self):
    @function
    def f(x, y): return (x*y+1).contiguous()
    a, b = buf(), buf()
    fn, externs = exported(f(a, b))
    self.assert_exportable(fn, externs)
    self.assertIn(Ops.CALL, {u.op for u in fn.toposort()})

  def test_precompiled_function(self):
    @function(precompile=True)
    def f(x, y): return (x*y+1).contiguous()
    a, b = buf(), buf()
    fn, externs = exported(f(a, b))
    self.assert_exportable(fn, externs)

  def test_alloc_is_canonical(self):
    @function
    def f(x): return (x+1).contiguous()*2
    a, b = buf(), buf()
    fn, _ = exported(f(a))
    self.assertIs(fn, exported(f(b))[0])

if __name__ == "__main__":
  unittest.main()
