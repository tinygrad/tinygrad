import unittest, itertools, torch, numpy as np
from tinygrad import Tensor, Device, dtypes, nn, GlobalCounters
from tinygrad.helpers import VIZ
from tinygrad.renderer.isa import ISARenderer, IselContext
from tinygrad.uop.ops import graph_rewrite, PatternMatcher, UPat, UOp, Ops, ProgramInfo, AddrSpace
from tinygrad.codegen import full_rewrite_to_sink, pm_to_program
from tinygrad.engine.realize import _get_call_to_compile, run_linear
from test.backend.test_ops import prepare_test_op

def _cross_exec(graph:Tensor) -> int:
  device = Device[Device.DEFAULT]
  isa_ren, final_ren = device.renderer, next(r for r in device.renderers if not issubclass(r, ISARenderer))
  final_ren = final_ren(isa_ren.target)

  def transmute(ast:UOp) -> UOp:
    sink = full_rewrite_to_sink(ast, isa_ren)
    # perform instruction selection
    sink = graph_rewrite(sink, isa_ren.pre_isel_matcher, ctx=itertools.count(-1,-1), name="pre instruction selection", bottom_up=True)
    sink = graph_rewrite(sink, isa_ren.isel_matcher, ctx=IselContext(sink), name="instruction selection", bottom_up=True)
    sink = graph_rewrite(sink, PatternMatcher([]), name="view machine code")

    # NOTE: slightly hacky with the negative slot to differentiate from device BUFFERs
    pm_substitute_operands = PatternMatcher([
      (UPat(Ops.PARAM, name="p"), lambda ctx,p: ctx[p.arg.slot] if p.addrspace is AddrSpace.OPR else None)
    ])
    # re-expand CALL graphs
    sink = sink.substitute({c:graph_rewrite(c.src[0], pm_substitute_operands, ctx=c.src[1:]) for c in sink.toposort() if c.op is Ops.CALL})

    # plug through non-assembly backend's render pass
    prg_info = ProgramInfo.from_sink(sink, final_ren.target)
    prg = UOp(Ops.PROGRAM, src=(sink,), arg=prg_info)
    prg = graph_rewrite(prg, pm_to_program, ctx=final_ren, name="linearize/render")
    if VIZ: graph_rewrite(prg, PatternMatcher([]), name="View Program")
    return prg

  # compile kernels and swap calls
  linear = graph.schedule_linear()
  calls = {c: (c.src[0], final_ren) for c in linear.toposort() if c.op is Ops.CALL and _get_call_to_compile(c) is not None}
  prgs = {c: transmute(a[0]) for c,a in calls.items()}
  linear = linear.substitute({c: c.replace(src=(c.src[0].substitute({a[0]: prgs[c]}), *c.src[1:])) for c,a in calls.items()})
  GlobalCounters.reset()
  run_linear(linear, jit=True)
  return GlobalCounters.kernel_count

# TODO: verify post-linearize round trip?
@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, ISARenderer), "cross compilation is for asm backends")
class TestRetarget(unittest.TestCase):
  def test_transfer_gemm(self):
    trt, tgt = prepare_test_op(-2, 2, [(32,32), (32,32)], None)
    truth, out = torch.matmul(*trt), Tensor.matmul(*tgt)
    _cross_exec(out)
    np.testing.assert_allclose(out.numpy(), truth.detach().numpy(), atol=1e-6, rtol=1e-3)

  def test_transfer_mnist_kernel_count(self):
    layers = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 1)]

    Tensor.realize(*[p.replace(Tensor.ones_like(p).contiguous()) for p in nn.state.get_parameters(layers)])
    x = Tensor.rand(1, 1, 28, 28)
    Tensor.realize(x)
    ref, out = x.sequential(layers), x.sequential(layers)
    GlobalCounters.reset()
    truth = ref.numpy()
    native = GlobalCounters.kernel_count
    cross = _cross_exec(out)
    self.assertEqual(native, cross)
    np.testing.assert_allclose(out.numpy(), truth, atol=1e-6, rtol=1e-3)

  def test_transfer_loop(self):
    from test.backend.test_wait_loop import wait_loop_kernel
    def mk(): return Tensor.custom_kernel(Tensor.empty(1, dtype=dtypes.int), fxn=wait_loop_kernel)[0]
    ref, out = mk(), mk()
    GlobalCounters.reset()
    truth = ref.item()
    native = GlobalCounters.kernel_count
    cross = _cross_exec(out)
    self.assertEqual(native, cross)
    self.assertEqual(truth, out.item())

if __name__ == '__main__':
  np.random.seed(2973)
  unittest.main(verbosity=2)
