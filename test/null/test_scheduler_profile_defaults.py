import unittest
from tinygrad.llm.serve import LLMServer

class TestSchedulerProfileDefaults(unittest.TestCase):
  def test_defaults_match_live_profiles(self):
    self.assertEqual((LLMServer.MAX_STALL, LLMServer.DECODE_SHARE, LLMServer.AGING, LLMServer.STARVE_SEQ,
                      LLMServer.TURN_WAIT, LLMServer.COLD_MAX, LLMServer.COLD_TOKENS), (0.3, 0.5, 0, 600.0, 10.0, 1, 16384))
