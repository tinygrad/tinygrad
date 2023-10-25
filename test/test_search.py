import unittest
import numpy as np

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import time_linearizer
from tinygrad.ops import Compiled, Device, LoadOps
from tinygrad.tensor import Tensor

class TestTimeLinearizer(unittest.TestCase):
  def setUp(self) -> None:
    if not isinstance(Device[Device.DEFAULT], Compiled): raise unittest.SkipTest("only test for compiled backends")

  def test_reasonable_time(self):
    si = [si for si in Tensor([1,2,3,4]).add(1).lazydata.schedule() if si.ast.op not in LoadOps][0]
    rawbufs = [Device[Device.DEFAULT].buffer(si.out.st.size(), si.out.dtype)] + [Device[Device.DEFAULT].buffer(x.st.size(), x.dtype) for x in si.inputs]
    tm = time_linearizer(Linearizer(si.ast), rawbufs, allow_test_size=False, cnt=10, should_copy=False)
    assert tm > 0 and tm != float('inf')

  def test_no_corrupt_output(self):
    a = Tensor.rand(3, 3).realize()
    a.assign(a+1)

    num_tested = 0
    for si in a.lazydata.schedule():
      if si.ast.op in LoadOps: continue
      rawbufs = [si.out.output_buffer, *[i.realized for i in si.inputs]]
      pre_values = [buf.toCPU().copy() for buf in rawbufs]
      lin = Linearizer(si.ast, Device[Device.DEFAULT].linearizer_opts)

      t = time_linearizer(lin, rawbufs, disable_cache=True)
      assert t > 0
      post_values = [buf.toCPU() for buf in rawbufs]

      for v1, v2 in zip(pre_values, post_values):
        np.testing.assert_allclose(v2, v1)
      num_tested += 1
    assert num_tested > 0