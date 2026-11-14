import unittest
from unittest.mock import Mock
from tinygrad import Tensor, TinyJit
from tinygrad.llm.qwen4exp import _run_again

def f(x:Tensor) -> Tensor:
  y = (x + 1).contiguous().realize()  # (an intermediate: in the shared scratch's arena)
  return (y * 2).sum(0).realize()

class TestWarmupRunAgain(unittest.TestCase):
  def test_a_later_capture_unlinks_and_run_again_links(self):
    # seq_warmup: jits run as they are taken; one taken later with a bigger intermediate grows the shared arena and rebases the others (their
    # links dropped): their first call in serving would link them again. _run_again runs the warm-up's calls once more after the last capture
    sc:dict = {}
    small, big = TinyJit(f, scratch=sc), TinyJit(f, scratch=sc)
    call_small, call_big = Mock(side_effect=lambda: small(Tensor.ones(4, 4).contiguous().realize())), \
                           Mock(side_effect=lambda: big(Tensor.ones(256, 256).contiguous().realize()))
    for _ in range(3): call_small()
    self.assertTrue(small.captured._linears)
    for _ in range(3): call_big()
    self.assertFalse(small.captured._linears)  # (rebased by big's capture: the case the warm-up leaves behind today)
    n_small, n_big = call_small.call_count, call_big.call_count
    self.assertEqual(_run_again({"small": (small, call_small), "big": (big, call_big)}, set()), ["small"])
    self.assertEqual((call_small.call_count, call_big.call_count), (n_small + 1, n_big + 1))  # (every call once: launch arguments too)
    self.assertTrue(small.captured._linears and big.captured._linears)

  def test_skips_late_and_uncaptured(self):
    sc:dict = {}
    late, never = TinyJit(f, scratch=sc), TinyJit(f, scratch=sc)
    for _ in range(3): late(Tensor.ones(4, 4).contiguous().realize())
    call_late, call_never = Mock(), Mock()
    self.assertEqual(_run_again({"late": (late, call_late), "never": (never, call_never)}, {late}), [])
    self.assertEqual((call_late.call_count, call_never.call_count), (0, 0))  # (the stash's jits: link_late's; a jit never captured: none)

if __name__ == "__main__":
  unittest.main()
