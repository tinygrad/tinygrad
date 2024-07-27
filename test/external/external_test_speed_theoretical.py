import time, unittest
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import getenv

def _test(tcount, fxn, szmax):
  print(f"**** testing {fxn.__name__}")
  allgbs = []
  for sz in range(szmax):
    jfxn = TinyJit(fxn)
    ts = [Tensor.zeros((2**sz)*1024*1024).contiguous().realize() for _ in range(tcount)]
    tms = []
    for _ in range(10):
      ts = [(x+1).realize() for x in ts]
      Device.default.synchronize()
      st = time.perf_counter()
      out_nbytes = jfxn(*ts).nbytes()
      Device.default.synchronize()
      tms.append(time.perf_counter() - st)
    gbs = (out_nbytes+sum(x.nbytes() for x in ts))*1e-9/min(tms)
    print(f"{ts[0].nbytes()/(1024*1024):10.0f} MB, {min(tms)*1e3:6.2f} ms GB/s {gbs:<10.2f} {str(ts[0].shape):20s}")
    allgbs.append(gbs)
  return max(allgbs)

MEMBW = getenv("MEMBW", 10)
class TestTheoreticalSpeed(unittest.TestCase):
  def test_add(self): self.assertGreater(_test(2, Tensor.add, 11), MEMBW)
  def test_exp(self): self.assertGreater(_test(1, Tensor.exp, 11), MEMBW)
  def test_sum(self): self.assertGreater(_test(1, Tensor.sum, 11), MEMBW)

if __name__ == '__main__':
  unittest.main()
