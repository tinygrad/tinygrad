import os, time, unittest
import numpy as np
from unittest.mock import patch
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import getenv
from tinygrad.dtype import dtypes

@unittest.skipUnless(getenv("RDMA"), "requires two AMD nodes with BNXT NICs and RDMA=1")
class TestRDMACopy(unittest.TestCase):
  def setUp(self):
    self.devs = os.environ.get("RDMA_DEVS", "AMD,AMD:6").split(",")
    assert Device[self.devs[0]].peer_group != Device[self.devs[1]].peer_group

  def test_copy_replay(self):
    expected = np.arange(16 << 20, dtype=np.uint8)
    x = Tensor(expected, device=self.devs[0]).realize()
    def copy(x): return x.to(self.devs[1]).contiguous().realize()
    np.testing.assert_equal(copy(x).numpy(), expected)
    f = TinyJit(copy)
    for _ in range(2): f(x)
    Device[self.devs[1]].synchronize()
    original = CPUProgram.__call__
    def delayed(prg, *args, **kwargs):
      if prg.dev.device == Device[self.devs[1]].host: time.sleep(0.05)
      return original(prg, *args, **kwargs)
    # Force RNR on nonzero MSN indices: a posted receive must not be required before sending.
    with patch.object(CPUProgram, "__call__", delayed):
      for _ in range(3):
        f(x)
        Device[self.devs[1]].synchronize()
    start = time.perf_counter()
    for _ in range(50):
      y = f(x)
      Device[self.devs[1]].synchronize()
    elapsed = time.perf_counter() - start
    np.testing.assert_equal(y.numpy(), expected)
    print(f"RDMA copy: {50 * expected.nbytes / elapsed / 1e9:.2f} GB/s, 50 x 16 MiB")

  def test_copy_wrap(self): # 300 copies each way of changing contents: the 32 entry rings and the 128 entry cqs wrap, every result is checked
    from tinygrad.runtime import ops_rdma
    n = 3 << 18
    original = CPUProgram.__call__
    with patch.object(ops_rdma, "RDMA_CHUNK", 1 << 18): # three chunks per copy
      for src, dst in (self.devs, self.devs[::-1]):
        xs = [Tensor.zeros(n, dtype=dtypes.uint8, device=src).contiguous().realize() for _ in range(2)] # two input allocations, used in turn
        f = TinyJit(lambda x: x[4096:].to(dst).contiguous().realize()) # an offset view
        def late_receiver(prg, *args, **kwargs): # the send is posted before the receive
          if prg.dev.device == Device[dst].host: time.sleep(0.05)
          return original(prg, *args, **kwargs)
        for i in range(300):
          data = np.random.default_rng(i).integers(0, 256, n, dtype=np.uint8)
          xs[i % 2].assign(Tensor(data, device=src)).realize()
          with patch.object(CPUProgram, "__call__", late_receiver if 34 <= i < 38 else original): y = f(xs[i % 2])
          np.testing.assert_equal(y.numpy(), data[4096:])

  def test_sharded_reduce(self):
    def reduce(x): return (x + 1).sum().realize()
    x = Tensor(np.arange(1024, dtype=np.float32)).shard(self.devs, axis=0).clone().realize()
    expected = reduce(x).item()
    self.assertEqual(expected, 524800)
    f = TinyJit(reduce)
    for _ in range(52): self.assertEqual(f(x).item(), expected)

if __name__ == "__main__": unittest.main()
