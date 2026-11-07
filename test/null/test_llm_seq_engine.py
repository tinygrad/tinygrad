import array, threading, time, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad.llm.engine import SeqEngine
from tinygrad.llm.serve import LLMServer, Request, BatchStats

class FakeModel:
  """the model side of sequences sharing the caches (the seq_* calls of SeqEngine), recorded: a prefill takes `per_token` s a token"""
  SEQ_SIZES, SEQ_MOVE = (1, 4), 2048
  def __init__(self, per_token:float=0.0):
    self.config, self.max_context, self.per_token = SimpleNamespace(spec_drafts=2), 1 << 16, per_token
    self._token_host = array.array('i', [0] * (self.seq_rows() + 4096))
    self.calls:list[tuple] = []
  def seq_rows(self) -> int: return 1 << 16
  def seq_slots(self) -> int: return 4
  def seq_prefill(self, slot, base, tokens, start, hrow, temp=0.0, upto=None) -> int|None:
    end = len(tokens) if upto is None else upto
    self.calls.append(("prefill", slot, start, end))
    time.sleep(self.per_token * (end - start))
    return 7 if upto is None else None
  def seq_first(self, slot, base, pos, temp=0.0) -> int:
    self.calls.append(("first", slot, pos))
    return 8
  def seq_ckpt(self, save, slot, pos): self.calls.append(("ckpt", save, slot, pos))
  def seq_move(self, src, dst, n): self.calls.append(("move", src, dst, n))
  def seq_step(self, seqs):
    k = self.config.spec_drafts
    return [([5] * (k+1), [5] * k) for _ in seqs]

class TestSeqEngine(unittest.TestCase):
  def setUp(self): self.model = FakeModel()
  def engine(self) -> SeqEngine: return SeqEngine(self.model)

  def test_same_prompt_again(self):
    # the same prompt after its reply: its checkpoint restored, the token after it sampled from it (nothing prefilled)
    eng, ids = self.engine(), list(range(100, 3100))
    s, start, restore = eng.take(ids)
    self.assertEqual((start, restore), (0, False))
    eng.prefill(s, ids, start, restore, (0.0, 1.0))
    eng.step([s])
    eng.stop(s, len(s.tokens))
    self.assertEqual(eng.cached(ids), len(ids))
    self.model.calls.clear()
    s2, start, restore = eng.take(ids)
    self.assertIs(s2, s)
    self.assertEqual((start, restore), (len(ids), True))
    self.assertEqual(eng.prefill(s, ids, start, restore, (0.0, 1.0)), 8)
    self.assertEqual(self.model.calls, [("ckpt", False, s.slot, len(ids)), ("first", s.slot, len(ids))])
    self.assertEqual((s.tokens, s.n), (ids + [8], len(ids)))

  def test_continuation(self):
    # a prompt that continues one from its checkpoint (the reply not in it): prefilled from the checkpoint
    eng, ids = self.engine(), list(range(100, 3100))
    s, start, restore = eng.take(ids)
    eng.prefill(s, ids, start, restore, (0.0, 1.0))
    eng.step([s])
    eng.stop(s, len(s.tokens))
    nxt = ids + [1, 2, 3]
    self.assertEqual(eng.cached(nxt), len(ids))
    self.model.calls.clear()
    s2, start, restore = eng.take(nxt)
    self.assertEqual((start, restore), (len(ids), True))
    eng.prefill(s2, nxt, start, restore, (0.0, 1.0))
    self.assertEqual(self.model.calls[:2], [("ckpt", False, s.slot, len(ids)), ("prefill", s.slot, len(ids), len(nxt))])

  def test_slices(self):
    # a prompt prefilled in slices: every one from the end of the one before, the token sampled after the last
    eng, ids = self.engine(), list(range(100, 5100))
    s, start, restore = eng.take(ids)
    eng.prefill_slice(s, ids, start, restore, 2000)
    self.assertEqual((s.n, s.tokens), (2000, ids[:2000]))
    eng.prefill_slice(s, ids, 2000, False, 4000)
    self.assertEqual(eng.prefill(s, ids, 4000, False, (0.0, 1.0)), 7)
    self.assertEqual([c for c in self.model.calls if c[0] == "prefill"], [("prefill", s.slot, 0, 2000), ("prefill", s.slot, 2000, 4000),
                                                                          ("prefill", s.slot, 4000, 5000)])
    self.assertEqual((s.n, s.ckpt), (5000, 5000))

class TestScheduleSeqs(unittest.TestCase):
  """the server's scheduling of sequences (schedule_seqs) on a fake model"""
  def setUp(self):
    self.model = FakeModel(per_token=2e-5)
    self.model.seqs_supported = lambda: True
    srv = self.server = LLMServer.__new__(LLMServer)
    srv.model, srv.engine, srv.stats, srv.inflight, srv.odd_ids = self.model, None, BatchStats(), 0, set()
    srv.tok = Mock(is_end=lambda t: False)
    srv.waiting, srv.recent = __import__("queue").Queue(), __import__("collections").deque()
    srv.PREFILL_SLICE, srv.MAX_STALL = 1000, 0.01
    self.done:list[Request] = []
    finish = srv.finish
    def fin(req, error=None):
      if not req.done: self.done.append(req)
      return finish(req, error)
    srv.finish = fin
    threading.Thread(target=srv.schedule_seqs, daemon=True).start()
    time.sleep(0.05)
  def wait(self, reqs:list[Request], timeout=10.0):
    st = time.perf_counter()
    while not all(r.done for r in reqs) and time.perf_counter() - st < timeout: time.sleep(0.01)
    self.assertTrue(all(r.done for r in reqs))

  @patch("tinygrad.llm.serve.stderr_log")
  def test_short_before_long(self, _):
    # a short prompt that comes while a long one is prefilled (in slices): prefilled before the long one's slices left, its reply goes on
    # while the long one is prefilled
    long, short = Request(list(range(100, 20100)), "", 4, 0.0), Request(list(range(30000, 30050)), "", 4, 0.0)
    self.server.waiting.put(long)
    time.sleep(0.05)
    self.server.waiting.put(short)
    self.wait([long, short])
    self.assertEqual(self.done, [short, long])
    self.assertGreater(long.slices, 5)
    self.assertEqual(len(short.generated), 4)
    # the same prompt again: all of it cached
    again = Request(list(range(100, 20100)), "", 4, 0.0)
    self.server.waiting.put(again)
    self.wait([again])
    self.assertEqual(again.cache_start, len(again.ids))
    self.assertEqual(again.slices, 1)
    # the latencies of the requests in the status
    st = self.server.status()
    self.assertEqual(st["last_10m"]["requests"], 3)
    self.assertGreater(st["last_10m"]["ttft"]["max"], 0)
    self.assertEqual(st["generated_total"], 12)

if __name__ == "__main__":
  unittest.main()
