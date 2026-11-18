import unittest, math

from tinygrad import Tensor, UOp, Device, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.device import Buffer
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.search import _beam_call, _time_program
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops, KernelInfo, AxisType


class TestSearch(unittest.TestCase):
  def test_beam_symbolic_kernel(self):
    size = UOp.variable("size", 1, 8)
    out = Tensor.empty(8, 8, device="CPU")[:size.bind(4)] + 1
    linear, _ = out.linear_with_vars()
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1, CACHELEVEL=0): compiled = compile_linear(linear, beam=1)
    program = next(u for u in compiled.toposort(enter_calls=True) if u.op is Ops.PROGRAM)
    self.assertNotEqual(program.src[0].arg.applied_opts, ())

  @unittest.skipUnless(Device[Device.DEFAULT].renderer.has_local, "needs global dims")
  def test_beam_unused_input(self):
    # the kernel ignores its input at slot 1 and its captured Variable is its arg at slot 2, beam's timing call keeps the slots
    v = UOp.variable("v", 1, 8).bind(3)
    def fxn(o:UOp, _:UOp) -> UOp:
      r = UOp.range(256, 0, AxisType.GLOBAL)
      return o[r].store(v.cast(dtypes.float)).end(r).sink(arg=KernelInfo("ignores_input"))
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1, CACHELEVEL=0):
      out = Tensor.custom_kernel(Tensor.empty(256), Tensor.empty(256), fxn=fxn)[0].realize()
    self.assertEqual(out.tolist(), [3.0]*256)

  def test_beam_scalar_first(self):
    # the scalar at slot 0 isn't the launch device, the buffer at slot 1 is
    v = UOp.variable("v", 1, 10, dtypes.int).bind(3)
    def fxn(s:UOp, o:UOp) -> UOp:
      i = UOp.range(4096, 0, AxisType.WEAK)
      return o[i].store(s).end(i).sink(arg=KernelInfo("scalar_first"))
    def out(): return Tensor.custom_kernel(Tensor(v), Tensor.empty(4096, dtype=dtypes.int), fxn=fxn)[1]
    # beam times the candidates with a value for the scalar
    prg = to_program(out().linear_with_vars()[0].src[0].body, Device[Device.DEFAULT].renderer)
    bufs = {1:UOp.from_buffer(Buffer(Device.DEFAULT, 4096*4).ensure_allocated(), dtypes.int)}
    self.assertTrue(all(math.isfinite(t) for t in _time_program(_beam_call(prg, bufs))))
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1, CACHELEVEL=0): self.assertEqual(out().tolist()[:2], [3, 3])

if __name__ == "__main__":
  unittest.main()
