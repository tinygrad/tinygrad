import unittest
from tinygrad import Tensor, GlobalCounters, dtypes
from tinygrad.helpers import Context
from tinygrad.engine.realize import run_schedule

class TestArange(unittest.TestCase):
  def _get_flops(self, N):
    GlobalCounters.reset()
    with Context(NOOPT=1):
      Tensor.arange(N).realize()
    return GlobalCounters.global_ops

  def test_complexity(self):
    # add 1 to avoid divide by 0. arange is 0 flops now!
    f1 = self._get_flops(256) + 1
    f2 = self._get_flops(2560) + 1
    print(f"{f1=}, {f2=}")
    assert f2 / f1 < 15, f"bad complexity, flops {f2/f1:.1f}X while inputs 10X"

class TestIndexing(unittest.TestCase):
  def test_arange_2_reduce(self):
    needle = Tensor.zeros(16384, dtype=dtypes.int).contiguous()
    needle[1337] = 1
    needle.realize()
    with Context(NOOPT=1, FUSE_AS_ONE_KERNEL=1):
      GlobalCounters.reset()
      # TODO: it should work without these reshapes
      out = ((Tensor.arange(1,16385).reshape(16384,1)-1)*needle.reshape(16384,1)).sum()
      sched = out.schedule()
      assert len(sched) == 1
      run_schedule(sched)
    assert out.item() == 1337, f"expected 1337, got {out.item()}"

  @unittest.expectedFailure
  def test_manual_index(self):
    #dataset = Tensor.rand(16384, 256).realize()
    #idxs = Tensor([0,3,5,6]).realize()
    print("*** indexing ***")
    with Context(NOOPT=1, FUSE_AS_ONE_KERNEL=1):
      GlobalCounters.reset()
      rng = Tensor.arange(1,16385).reshape(1,16384,1).expand(4, 16384, 256)-1
      sched = rng.schedule()
      run_schedule(sched)

  def test_index(self):
    dataset = Tensor.rand(16384, 256).realize()
    idxs = Tensor([0,3,5,6]).realize()
    print("*** indexing ***")
    with Context(NOOPT=1):
      GlobalCounters.reset()
      X = dataset[idxs]
      assert X.shape == (4,256)
      sched = X.schedule()
      #assert len(sched) == 1
      run_schedule(sched)

if __name__ == "__main__":
  unittest.main()