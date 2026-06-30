import unittest

from tinygrad.codegen import to_program, to_program_cache
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.isa.x86 import X86Ops, X86Renderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp


class TestRegalloc(unittest.TestCase):
  def test_stack_dealloc_emits_before_sink(self):
    renderer = X86Renderer(Target("CPU", arch="x86_64"))
    out = UOp.placeholder((1,), dtypes.float32, 0)
    local = UOp.placeholder((1,), dtypes.float32, 1, addrspace=AddrSpace.REG)
    idx = UOp.const(dtypes.int32, 0)
    store_local = local.index(idx).store(UOp.const(dtypes.float32, 3.0))
    store_out = out.index(idx).store(local.index(idx).load())
    sink = UOp.sink(store_local, store_out, arg=KernelInfo(name="x86_regalloc_stack_dealloc"))
    to_program_cache.clear()
    lin = to_program(sink, renderer).src[1].src

    sub = [i for i,u in enumerate(lin) if u.op is Ops.INS and u.arg is X86Ops.SUBi]
    add = [i for i,u in enumerate(lin) if u.op is Ops.INS and u.arg is X86Ops.ADDi]
    ret = next(i for i,u in enumerate(lin) if u.op is Ops.INS and u.arg is X86Ops.RET)
    self.assertEqual(len(sub), 1)
    self.assertEqual(len(add), 1)
    self.assertLess(sub[0], add[0])
    self.assertLess(add[0], ret)
    self.assertIs(lin[-1].op, Ops.SINK)


if __name__ == "__main__":
  unittest.main()
