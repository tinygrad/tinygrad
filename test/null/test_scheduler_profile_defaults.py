import inspect, unittest
from tinygrad.llm.serve import LLMServer

class TestSchedulerProfileDefaults(unittest.TestCase):
  def test_defaults_match_live_profiles(self):
    self.assertEqual((LLMServer.MAX_STALL, LLMServer.DECODE_SHARE, LLMServer.AGING, LLMServer.STARVE_SEQ,
                      LLMServer.TURN_WAIT, LLMServer.COLD_MAX, LLMServer.COLD_TOKENS), (0.3, 0.5, 0, 600.0, 10.0, 1, 16384))

  def test_no_slot_scheduler(self):
    # every production server runs schedule_seqs (or schedule_alone): the batch-slot scheduler and its stash are gone
    for name in ("schedule", "run_alone", "stash_load", "mix_take", "slots"): self.assertFalse(hasattr(LLMServer, name), name)
    self.assertNotIn("self.schedule()", inspect.getsource(LLMServer.serve_forever))
