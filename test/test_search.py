import unittest

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.engine.schedule import create_schedule
from tinygrad.features.search import time_linearizer, bufs_from_lin
from tinygrad.device import Device, Buffer
from tinygrad.ops import LoadOps, BufferOps
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from tinygrad.engine.realize import capturing

class TestTimeLinearizer(unittest.TestCase):
  def test_reasonable_time(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast[0].op not in LoadOps][0]
    out = Buffer(Device.DEFAULT, si.outputs[0].size, si.outputs[0].dtype).allocate()
    memops = {x.arg.idx:x.arg.st.real_size() for x in si.ast[0].lazyops if x.op is BufferOps.LOAD}
    rawbufs = [out] + [Buffer(Device.DEFAULT, memops[i], x.dtype).allocate() for i,x in enumerate(si.inputs, start=len(si.outputs))]
    tm = time_linearizer(Linearizer(*si.ast), rawbufs, allow_test_size=False, cnt=10)
    assert tm > 0 and tm != float('inf')

  def test_bufs_from_lin(self):
    si = [i for i in create_schedule([Tensor([1,2,3,4]).add(1).lazydata]) if i.ast[0].op not in LoadOps][0]
    rawbufs = bufs_from_lin(lin:=Linearizer(*si.ast))
    assert len(rawbufs) == len(lin.membufs)
    assert all(r is not None for r in rawbufs)
    assert all(isinstance(r, Buffer) for r in rawbufs)
    assert all(r.size > 0 for r in rawbufs)

class TestBEAM(unittest.TestCase):
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
    assert k_beam_0[-1].prg.prg != k_beam_1[-1].prg.prg

if __name__ == '__main__':
  unittest.main()
