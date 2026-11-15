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

  def test_parts_list_jit(self):
    # (PREFILL_PARTS: a prefill chunk's jit is a list of its parts' jits: run again (one unlinked part: unlinked), never a TypeError; g4 Nov 14)
    sc:dict = {}
    p0, p1, big = TinyJit(f, scratch=sc), TinyJit(f, scratch=sc), TinyJit(f, scratch=sc)
    call_parts = Mock(side_effect=lambda: [p(Tensor.ones(4, 4).contiguous().realize()) for p in (p0, p1)])
    call_big = Mock(side_effect=lambda: big(Tensor.ones(256, 256).contiguous().realize()))
    for _ in range(3): call_parts()
    for _ in range(3): call_big()
    self.assertEqual(_run_again({"prefill192": ([p0, p1], call_parts), "big": (big, call_big)}, set()), ["prefill192"])
    self.assertTrue(p0.captured._linears and p1.captured._linears)
    self.assertEqual(_run_again({"prefill192": ([p0, p1], call_parts)}, {p0, p1}), [])  # (every part skipped: not called)
    self.assertEqual(call_parts.call_count, 4)

class TestSeqRunAgain(unittest.TestCase):
  def test_tower_after_the_model(self):
    # (VISION_LATE: the tower's warm-up captured after seq_warmup grows the scratch and unlinks the model's jits: seq_run_again links them)
    from tinygrad.llm.qwen4exp import Transformer
    sc:dict = {}
    step, late, tower = TinyJit(f, scratch=sc), TinyJit(f, scratch=sc), TinyJit(f, scratch=sc)
    call_step, call_late = Mock(side_effect=lambda: step(Tensor.ones(4, 4).contiguous().realize())), Mock()
    for _ in range(3): call_step()
    model = Mock()
    model.__dict__["_warm_calls"] = ({"step24": (step, call_step), "stash": (late, call_late)}, {late})
    for _ in range(3): tower(Tensor.ones(256, 256).contiguous().realize())
    self.assertFalse(step.captured._linears)
    self.assertEqual(Transformer.seq_run_again(model), ["step24"])
    self.assertTrue(step.captured._linears)
    self.assertEqual((call_step.call_count, call_late.call_count), (4, 0))
    self.assertNotIn("_warm_calls", model.__dict__)  # (dropped: the warm-up's calls not kept for the process's life)

class TestVisionRunAgain(unittest.TestCase):
  def test_every_bucket_once(self):
    # (VISION: the tower's jits share the model's scratch: after the model's warm-up a blank picture of every bucket links them again)
    from tinygrad.llm.qwen_vision import VisionTower, bucket
    tower = Mock(patch_dim=1536)
    VisionTower.run_again(tower, 1024)
    self.assertEqual([c.args[0].shape[0] for c in tower.call_args_list], [256, 512, 1024, 2048, 4096])
    self.assertTrue(all(c.args[1] == (16, c.args[0].shape[0] // 16) and c.args[0].shape[0] == bucket(c.args[0].shape[0])
                        for c in tower.call_args_list))

if __name__ == "__main__":
  unittest.main()
