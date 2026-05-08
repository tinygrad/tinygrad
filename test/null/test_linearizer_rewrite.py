import unittest
from tinygrad import Tensor, Context, Device
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import KernelInfo

class TestLinearizerRewrite(unittest.TestCase):
  def test_reduction(self):
    t = Tensor.ones((64,64), device="NULL").contiguous().realize()
    out = (t*2).sum(axis=1)
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      opts_to_apply.append(Opt(OptOps.UNROLL, 0, 4))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device["CPU"].renderer)
      print(prg.src[3].arg)

  def test_arange(self):
    out = Tensor.arange(32, device="NULL")
    with Context(SPLIT_REDUCEOP=0, DEVECTORIZE=0):
      si = out.schedule_linear().src[-1]
      opts_to_apply = []
      opts_to_apply.append(Opt(OptOps.UPCAST, 0, 4))
      ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=tuple(opts_to_apply)))
      prg = to_program(ast, Device["CPU"].renderer)
      print(prg.src[3].arg)

  def test_kernel_info(self):
    out = Tensor.arange(4, device="NULL")
    si = out.schedule_linear().src[-1]

    ast = si.src[0].replace(arg=KernelInfo(opts_to_apply=()))
    prg = to_program(ast, Device["CPU"].renderer)
    assert prg.src[0].arg.applied_opts == (), f"expected no opts, got {prg}"

    prg = to_program(ast.replace(arg=KernelInfo()), Device["CPU"].renderer)
    assert prg.src[0].arg.applied_opts != (), f"expected opts to apply, got {prg.src[0].arg.applied_opts}"

    prg = to_program(ast.replace(arg=KernelInfo(name="custom")), Device["CPU"].renderer)
    self.assertEqual(prg.arg.name, "custom")

class TestMATVEC(unittest.TestCase):
  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "MATVEC path requires has_local")
  def test_elementwise_reduce_not_matvec(self):
    a = Tensor.randn(4096, 4096).contiguous().realize()
    b = Tensor.randn(4096, 4096).contiguous().realize()
    out = (a * b).sum(axis=1)
    si = out.schedule_linear().src[-1]
    prg = to_program(si.src[0], Device[Device.DEFAULT].renderer)
    opts = prg.src[0].arg.applied_opts
    assert not any(o.op is OptOps.GROUP for o in opts), f"elementwise reduce misclassified as MATVEC: {opts}"

class TestPTXBfloat16(unittest.TestCase):
  def test_ptx_renderer_bf16_keyerror(self):
    from tinygrad.renderer.ptx import PTXRenderer
    from tinygrad.helpers import Target
    from tinygrad.dtype import dtypes, AddrSpace
    from tinygrad.uop.ops import UOp, Ops
    ren = PTXRenderer(Target('CUDA', 'PTX', 'sm_89'))
    ptr = dtypes.bfloat16.ptr(size=1, addrspace=AddrSpace.REG)
    u = UOp(Ops.DEFINE_REG, ptr, (), 0)
    sink = UOp(Ops.SINK, dtypes.void, (u,))
    ren.render([u, sink])  # KeyError: dtypes.bfloat16 at ptx.py:183

if __name__ == '__main__':
  unittest.main()
