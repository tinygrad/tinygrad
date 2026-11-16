import inspect, unittest
from tinygrad.llm.serve import LLMServer
from tinygrad.llm import serve

class TestSchedulerProfileDefaults(unittest.TestCase):
  def test_defaults_match_live_profiles(self):
    self.assertEqual((LLMServer.MAX_STALL, LLMServer.DECODE_SHARE, LLMServer.AGING, LLMServer.STARVE_SEQ,
                      LLMServer.TURN_WAIT, LLMServer.COLD_MAX, LLMServer.COLD_TOKENS), (0.3, 0.5, 0, 600.0, 10.0, 1, 16384))

  def test_batching_defaults(self):
    self.assertEqual(serve.BATCH_PREFILL_MIN, 3)
    schedule_source = inspect.getsource(LLMServer.schedule)
    self.assertNotIn("bcp_batch", schedule_source)
    self.assertNotIn("batch_continue", schedule_source)
    source = inspect.getsource(LLMServer)
    self.assertIn('not k and len(active) == 1 and not pending and self.waiting.empty()', source)
