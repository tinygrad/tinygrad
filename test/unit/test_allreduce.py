import unittest
from tinygrad import Tensor, Device
from tinygrad.helpers import Context
from tinygrad.ops import Ops

class TestRingAllReduce(unittest.TestCase):
  def test_schedule_ring(self):
    with Context(RING=2):
      MAX_N = 8
      ds = []
      for i in range(MAX_N):
             # Throws if the device is not available.
        try: Device[s:=f"{Device.DEFAULT}:{i}"] and ds.append(s)
        except Exception: break

      if N:=len(ds) < 2: self.skipTest("You are too GPU poor to run this test")
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      schedules = t.sum(0).schedule_with_vars()[0]
      copies = [si for si in schedules if si.ast.op is Ops.COPY]
      pairs = [(c.bufs[0].device, c.bufs[1].device) for c in copies]
      # N*(N-1) scatter reduce, and N*(N-1) allgather
      self.assertEqual(len(pairs), N*(N-1)*2)
      # copy topology forms a ring
      self.assertEqual(len(set(pairs)), N)

if __name__ == '__main__':
  unittest.main()
