import unittest
from tinygrad import Tensor, Variable, dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, ParamArg
from tinygrad.uop.uir import render_uir, parse_uir
from tinygrad.function import function
import tinygrad.schedule as schedule

def exported(*tensors:Tensor) -> tuple[UOp, tuple[UOp, ...]]:
  got:list[tuple[UOp, tuple[UOp, ...]]] = []
  orig = schedule.export
  schedule.export = lambda big_sink: got.append(ret:=orig(big_sink)) or ret
  try: Tensor.linear_with_vars(*tensors)
  finally: schedule.export = orig
  return got[0]

def buf(n=8) -> Tensor: return Tensor.empty(n, device="NULL").realize()

class TestExport(unittest.TestCase):
  def test_exportable(self):
    @function
    def f(x, y): return (x*y+1).contiguous()
    v = Variable("v", 1, 8)
    multi = Tensor.empty(4, 4, device="NULL").shard(("NULL:0", "NULL:1"), axis=0).realize()
    for t in [buf()[2:6].contiguous()+1, buf().bitcast(dtypes.int32)+1, buf()[:v.bind(3)].contiguous()*v.bind(3), multi*2, f(buf(), buf()),
              Tensor.rand(4, device="NULL")]:
      fn, externs = exported(t)
      for u in fn.toposort(enter_calls=True):
        self.assertIsNot(u.op, Ops.BUFFER)
        if isinstance(u.arg, ParamArg): self.assertTrue(u.arg.buffer is None and u.arg.val is None, f"runtime state in {u}")
        if u.op is Ops.EXTERN: self.assertEqual(u.device, externs[u.arg.slot].device)
      # UOps are hash-consed, so a lossless round trip through the wire format gives back the identical UOp
      self.assertIs(parse_uir(render_uir(fn)), fn)

  def test_same_function(self):
    v = Variable("v", 1, 4)
    a, b, c, d = buf(), buf(), buf(), buf()
    fn, _ = exported((a+b).contiguous()*v.bind(2))
    self.assertIs(fn, exported((c+d).contiguous()*v.bind(3))[0])
    # mapped back to this storage and these values
    linear, var_vals = ((c+d).contiguous()*v.bind(3)).linear_with_vars()
    self.assertEqual(var_vals, {"v": 3})
    self.assertTrue({c.uop.buf_uop, d.uop.buf_uop} <= set(linear.toposort()))

if __name__ == "__main__":
  unittest.main()
