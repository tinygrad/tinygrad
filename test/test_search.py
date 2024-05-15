import unittest

from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.search import time_linearizer, bufs_from_lin, actions
from tinygrad.device import Device, Buffer
from tinygrad.ops import LoadOps, BufferOps
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.engine.realize import capturing

class TestTimeLinearizer(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT in {"AMD", "NV"}, "Tries to open HSA/CUDA. FIXME #ticketno")
  def test_reasonable_time(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast[0].op not in LoadOps][0]
    out = Buffer(Device.DEFAULT, si.outputs[0].size, si.outputs[0].dtype).allocate()
    memops = {x.arg.idx:x.arg.st.real_size() for x in si.ast[0].lazyops if x.op is BufferOps.LOAD}
    rawbufs = [out] + [Buffer(Device.DEFAULT, memops[i], x.dtype).allocate() for i,x in enumerate(si.inputs, start=len(si.outputs))]
    tm = time_linearizer(Linearizer(*si.ast), rawbufs, allow_test_size=False, cnt=10)
    assert tm > 0 and tm != float('inf')

  @unittest.skipIf(Device.DEFAULT in {"AMD", "NV"}, "Tries to open HSA/CUDA. FIXME #ticketno")
  def test_bufs_from_lin(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast[0].op not in LoadOps][0]
    rawbufs = bufs_from_lin(lin:=Linearizer(*si.ast))
    assert len(rawbufs) == len(lin.membufs)
    assert all(r is not None for r in rawbufs)
    assert all(isinstance(r, Buffer) for r in rawbufs)
    assert all(r.size > 0 for r in rawbufs)

class TestBEAM(unittest.TestCase):
  @unittest.skipIf(Device.DEFAULT in {"AMD", "NV"}, "Tries to open HSA/CUDA. FIXME #ticketno")
  def test_dynamic_beam(self):
    # TODO: make this infra globally usable
    class Capture:
      def __init__(self): self.captured = []
      def add(self, x): self.captured.append(x)
    capturing.append(Capture())
    with Context(BEAM=1): Tensor.zeros(16).contiguous().realize()
    k_beam_1 = capturing[0].captured
    capturing.clear()
    capturing.append(Capture())
    with Context(BEAM=0): Tensor.zeros(16).contiguous().realize()
    k_beam_0 = capturing[0].captured
    capturing.clear()
    assert k_beam_0[-1].prg.p.src != k_beam_1[-1].prg.p.src

  def test_get_linearizer_actions(self):
    from test.test_linearizer import helper_realized_ast
    a = Tensor.rand(4, 3)
    b = Tensor.rand(3)
    realized_ast, _ = helper_realized_ast(a @ b)
    from tinygrad.engine.search import get_linearizer_actions
    lins = get_linearizer_actions(Linearizer(realized_ast), False).values()

    # ensure amt=0 are not duplicated
    if Opt(OptOps.UPCAST, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UPCAST, axis=0, amt=4)]) == 0, "did not de-dup UPCAST"
    if Opt(OptOps.LOCAL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.LOCAL, axis=0, amt=4)]) == 0, "did not de-dup LOCAL"
    if Opt(OptOps.UNROLL, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.UNROLL, axis=0, amt=3)]) == 0, "did not de-dup UNROLL"
    if Opt(OptOps.GROUP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUP, axis=0, amt=3)]) == 0, "did not de-dup GROUP"
    if Opt(OptOps.GROUPTOP, 0, 0) in actions:
      assert len([x for x in lins if x.applied_opts[0] == Opt(OptOps.GROUPTOP, axis=0, amt=3)]) == 0, "did not de-dup GROUPTOP"

if __name__ == '__main__':
  unittest.main()
