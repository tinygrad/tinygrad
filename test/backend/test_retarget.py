import unittest, itertools, torch, numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import VIZ
from tinygrad.renderer.isa import ISARenderer, IselContext
from tinygrad.uop.ops import graph_rewrite, PatternMatcher, UPat, UOp, Ops, ProgramInfo
from tinygrad.codegen import full_rewrite_to_sink, pm_to_program
from tinygrad.engine.realize import ExecContext, pm_exec, _get_call_to_compile
from test.backend.test_ops import prepare_test_op

def _cross_exec(graph:Tensor):
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
      (UPat(Ops.PARAM, name="p"), lambda ctx,p: ctx[abs(p.arg.slot)-1] if p.arg.slot < 0 else None),
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

  # execute and assert
  for call in linear.src: pm_exec.rewrite(call.without_after, ExecContext({}))

@unittest.skipUnless(isinstance(Device[Device.DEFAULT].renderer, ISARenderer), "cross compilation is for asm backends")
class TestRetarget(unittest.TestCase):
  def test_transfer_gemm(self):
    trt, tgt = prepare_test_op(-2, 2, [(32,32), (32,32)], None)
    truth, out = torch.matmul(*trt), Tensor.matmul(*tgt)
    _cross_exec(out)
    np.testing.assert_allclose(out.numpy(), truth.detach().numpy(), atol=1e-6, rtol=1e-3)

  def test_transfer_conv2d(self):
    bs, cin, cout, h, w, groups = 4, 3, 2, 2, 3, 1
    trt, tgt = prepare_test_op(-2, 2, [(bs,cin,5,7), (cout,cin//groups,h,w)], None)
    truth, out = torch.nn.functional.conv2d(*trt, groups=groups), Tensor.conv2d(*tgt, groups=groups)
    _cross_exec(out)
    np.testing.assert_allclose(out.numpy(), truth.detach().numpy(), atol=1e-6, rtol=1e-3)
