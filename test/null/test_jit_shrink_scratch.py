import gc, unittest
from tinygrad import Tensor, TinyJit
from tinygrad.engine.jit import shrink_scratch
from tinygrad.helpers import GlobalCounters

class TestShrinkScratch(unittest.TestCase):
  def test_shrink_after_big_jit_gone(self):
    scratch:dict = {}
    def big(x:Tensor) -> Tensor: return ((x.reshape(1, -1).expand(1024, -1) + 1).contiguous() * 2).sum(0).contiguous() + 1
    def small(x:Tensor) -> Tensor: return ((x + 1).contiguous() * 3).contiguous() + 2
    js, jb = TinyJit(small, capture_first=True, scratch=scratch), TinyJit(big, capture_first=True, scratch=scratch)
    for _ in range(3):
      a = Tensor.ones(4096).contiguous().realize()
      js(a), jb(a)
    before = max(v.max_numel() for v in scratch.values())
    self.assertGreater(before, 8 << 20)
    del jb
    gc.collect()
    m0 = GlobalCounters.mem_used
    got = shrink_scratch(scratch)
    self.assertEqual(len(got), 1)
    self.assertLess(max(v.max_numel() for v in scratch.values()), 1 << 20)
    self.assertLess(GlobalCounters.mem_used, m0 - (8 << 20))
    for _ in range(2): js(Tensor.ones(4096).contiguous().realize())  # (the small jit relinked on the smaller arena)
    self.assertEqual(shrink_scratch(scratch), {})  # (nothing more to shrink)

if __name__ == "__main__":
  unittest.main()
