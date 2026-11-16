import inspect, typing, unittest
from types import SimpleNamespace
from tinygrad.llm.serve import LLMServer
from tinygrad.llm import serve

class TestSchedulerProfileDefaults(unittest.TestCase):
  def test_defaults_match_live_profiles(self):
    self.assertEqual((LLMServer.MAX_STALL, LLMServer.DECODE_SHARE, LLMServer.AGING, LLMServer.STARVE_SEQ,
                      LLMServer.TURN_WAIT, LLMServer.COLD_MAX, LLMServer.COLD_TOKENS), (0.3, 0.5, 0, 600.0, 10.0, 1, 16384))

  def test_batch_and_bcp_defaults(self):
    self.assertEqual((serve.BATCH_PREFILL_MIN, serve.BCP_MIN), (3, 2))
    schedule_source = inspect.getsource(LLMServer.schedule)
    self.assertIn('room, most = model.batch_ctx() - model.BPF_P, 1536', schedule_source)
    self.assertIn(' > 1.5', schedule_source)
    self.assertIn(' <= 1.5', schedule_source)
    requests: list[typing.Any] = [SimpleNamespace(ids=list(range(size)), generated=[], st=st)
                                  for size, st in ((8, 98.5), (1, 99.0), (3, 99.2), (2, 99.1))]
    srv = typing.cast(typing.Any, object.__new__(LLMServer))
    srv.model = SimpleNamespace(bpf_class=None)
    srv.stash_hit = lambda ids, busy: 0
    self.assertEqual([r.ids for r in srv.bcp_batch(requests, most=2, busy=set())], [requests[0].ids, requests[1].ids])
    for count, wanted in ((0, 0), (serve.BCP_MIN, serve.BCP_MIN), (serve.BATCH_PREFILL_MIN, serve.BATCH_PREFILL_MIN)):
      self.assertEqual(len(srv.bcp_batch(requests, most=count, busy=set())), wanted)
    sized = [SimpleNamespace(ids=[0]*n, generated=[]) for n in (1, 1536, 1537)]
    self.assertEqual([len(r.ids + r.generated) <= 1536 for r in sized], [True, True, False])
    self.assertFalse(100.0 - requests[0].st > 1.5)
    self.assertTrue(100.01 - requests[0].st > 1.5)
    source = inspect.getsource(LLMServer)
    self.assertIn('getenv("BATCH_DECODE", 1)', source)
    self.assertIn('not k and len(active) == 1 and not pending and self.waiting.empty()', source)
    self.assertIn('getenv("BCP", 0)', source)
