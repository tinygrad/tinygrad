import platform, unittest
from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import Target
from tinygrad.renderer.isa.x86 import X86Renderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp

@unittest.skipUnless(Device.DEFAULT == 'CPU' and platform.machine() in ('x86_64', 'AMD64'), 'requires native x86 CPU')
class TestKernelSignature(unittest.TestCase):
  def test_stack_arguments(self):
    inputs = [Tensor([i], dtype=dtypes.int32).realize().uop for i in range(1, 9)]
    out = UOp.new_buffer('CPU', 1, dtypes.int32)
    params = [UOp.param(i, dtypes.int32, 1) for i in range(9)]
    factor = UOp.variable('factor', 0, 10, dtypes.int32)
    sink = params[-1].index(0).store(sum((i+1)*p.index(0).load() for i,p in enumerate(params[:-1])) + factor).sink(
      arg=KernelInfo(name='stack_signature'), tag=1)
    prg = to_program(sink, X86Renderer(Target('CPU', 'X86', Device['CPU'].arch)))
    self.assertEqual([p.arg.slot for p in prg.arg.params if not p.is_variable], list(range(9)))
    self.assertEqual([p[1] for p in prg.to_elf().signature], list(range(10)))
    run_linear(UOp(Ops.LINEAR, src=(prg.call(*inputs, out),)), var_vals={'factor': 2}, wait=True)
    self.assertEqual(out.buffer.as_memoryview().cast('i').tolist(), [206])

  def test_sparse_buffer_signature(self):
    p = UOp.param(3, dtypes.int32, 1)
    prg = to_program(p.index(0).store(7).sink(arg=KernelInfo(name='sparse_signature'), tag=1), X86Renderer(Target('CPU', 'X86', Device['CPU'].arch)))
    self.assertEqual(prg.arg.globals, (3,))
    self.assertEqual([p[1] for p in prg.to_elf().signature], [0])

if __name__ == '__main__': unittest.main()
