import unittest
from tinygrad.helpers import prod
from tinygrad.ops import Device
from tinygrad.tensor import Tensor
from tinygrad.helpers import GlobalCounters
from tinygrad.jit import CacheCollector

class TestCopy(unittest.TestCase):
  def test_add1(self):
    pts = []
    for i in range(16384, 16384*256, 16384):
      t = Tensor.randn(i).realize()
      CacheCollector.start()
      t.assign(t+1).realize()
      fxn, args, _ = CacheCollector.finish()[0]
      GlobalCounters.reset()
      def run(): return fxn(args, force_wait=True)
      ct = min([run() for _ in range(10)])
      mb = prod(t.shape)*t.dtype.itemsize*2*1e-6
      print(f"{mb*1e3:.2f} kB, {ct*1e3:.2f} ms, {mb/ct:.2f} MB/s")
      pts.append((mb, mb/ct))
    from matplotlib import pyplot as plt
    plt.plot([x[0] for x in pts], [x[1] for x in pts])
    plt.show()

if __name__ == '__main__':
  unittest.main()