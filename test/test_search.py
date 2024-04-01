import unittest

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.engine.schedule import create_schedule
from tinygrad.features.search import time_linearizer, bufs_from_lin
from tinygrad.device import Device, Buffer
from tinygrad.ops import LoadOps, BufferOps
from tinygrad.tensor import Tensor

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


if __name__ == '__main__':
  unittest.main()
