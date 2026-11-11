import array, threading, time, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad.llm.engine import SeqEngine
from tinygrad.llm.serve import LLMServer, Request, BatchStats

class FakeModel:
  """the model side of sequences sharing the caches (the seq_* calls of SeqEngine), recorded: a prefill takes `per_token` s a token"""
  SEQ_SIZES, SEQ_MOVE, SEQ_MIX_TOKENS = (1, 4), 2048, 32
  def __init__(self, per_token:float=0.0):
    self.config, self.max_context, self.per_token = SimpleNamespace(spec_drafts=2), 1 << 16, per_token
    self._token_host = array.array('i', [0] * (self.seq_rows() + 4096))
    self._seq_kbase:dict[int, int] = {}
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
  def _seq_copy_call(self, save, slot, pos, hrow, keep=True): self.calls.append(("copy", save, slot, pos, hrow, keep))
  def seq_step(self, seqs, forced=None):
    k = self.config.spec_drafts
    self.calls.append(("step", forced))
    out = []
    for i, seq in enumerate(seqs):
      sampled, drafts = [5] * (k+1), [5] * k
      if forced is not None:
        f = forced[i]
        sampled[:len(f)], drafts[:min(k, len(f))] = f, f[:k]
      out.append((sampled, drafts))
    return out

class TestSeqEngine(unittest.TestCase):
  def setUp(self): self.model = FakeModel()
  def engine(self) -> SeqEngine: return SeqEngine(self.model)

  def test_model_alignment(self):
    self.model.REGION_ALIGN = 4096
    self.assertEqual(self.engine().align, 4096)
    self.model.SEQ_BASE_ALIGN = 24
    self.assertEqual(self.engine().align, 24)
    self.assertEqual(self.model.SEQ_MOVE, 2048)

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

  def test_mixed_short_suffix(self):
    # eight suffix tokens: two full 3-row forced steps, then one suffix token and only the first sampled token emitted
    eng, ids = self.engine(), list(range(100, 3100))
    s, start, restore = eng.take(ids)
    eng.prefill(s, ids, start, restore, (0.0, 1.0))
    eng.step([s])
    eng.stop(s, len(s.tokens))
    nxt = ids + [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008]
    s2, start, restore = eng.take(nxt)
    self.assertIs(s2, s)
    self.assertEqual((start, restore), (len(ids), True))
    self.model.calls.clear()
    self.model._seq_kbase[s.slot] = start - 1  # stale SEQ_REPLAY metadata must not make the first mixed step replay
    eng.mix_start(s, nxt, start, restore, (0.0, 1.0))
    self.assertEqual(self.model._seq_kbase[s.slot], start)
    self.assertEqual(s.y[0], 9001)
    outs = []
    while s.mix: outs += eng.step([s])
    self.assertEqual(outs, [[], [], [5]])
    self.assertEqual([c[1][0] for c in self.model.calls if c[0] == "step"], [[9002, 9003, 9004], [9005, 9006, 9007], [9008]])
    self.assertEqual((s.tokens, s.n, s.ckpt), (nxt + [5], len(nxt), len(nxt)))
    self.assertFalse(any(c[0] == "prefill" for c in self.model.calls))
    self.assertIn(("copy", False, s.slot, len(ids), 0, True), self.model.calls)
    self.assertIn(("ckpt", True, s.slot, len(nxt)), self.model.calls)

  def test_mixed_exact_multiple_and_single(self):
    # a suffix exactly filling the draft slots caches the prompt without inventing an early output; one token mixes immediately
    for suffix in ([1, 2, 3, 4, 5, 6, 7], [1]):
      eng, ids = self.engine(), list(range(100, 3100))
      s, start, restore = eng.take(ids)
      eng.prefill(s, ids, start, restore, (0.0, 1.0))
      eng.stop(s, len(s.tokens))
      nxt = ids + suffix
      s, start, restore = eng.take(nxt)
      eng.mix_start(s, nxt, start, restore, (0.0, 1.0))
      outs = []
      while s.mix: outs += eng.step([s])
      if len(suffix) == 7: self.assertEqual(outs, [[], [], [5]])
      else: self.assertEqual(outs, [[5]])
      self.assertEqual((s.tokens, s.n, s.ckpt), (nxt + [5], len(nxt), len(nxt)))

  def test_mixed_every_supported_suffix(self):
    for n in range(1, self.model.SEQ_MIX_TOKENS + 1):
      with self.subTest(n=n):
        eng, ids = self.engine(), list(range(100, 3100))
        s, start, restore = eng.take(ids)
        eng.prefill(s, ids, start, restore, (0.0, 1.0))
        eng.stop(s, len(s.tokens))
        suffix = list(range(9000, 9000+n))
        nxt = ids + suffix
        s, start, restore = eng.take(nxt)
        eng.mix_start(s, nxt, start, restore, (0.0, 1.0))
        self.model.calls.clear()
        outs, forced = [], []
        while s.mix:
          outs += eng.step([s])
          forced += next(c[1][0] for c in reversed(self.model.calls) if c[0] == "step")
        self.assertEqual(outs, [[]] * (len(outs)-1) + [[5]])
        self.assertEqual(forced, suffix[1:])
        self.assertEqual((s.tokens, s.n, s.ckpt), (nxt + [5], len(nxt), len(nxt)))
        self.assertEqual(self.model._token_host[s.base + len(nxt)], 5)

  def test_mixed_does_not_change_ordinary_decode(self):
    eng = self.engine()
    a, b = list(range(100, 3100)), list(range(5000, 8000))
    sa, _, _ = eng.take(a)
    eng.prefill(sa, a, 0, False, (0.0, 1.0))
    eng.stop(sa, len(sa.tokens))
    sb, _, _ = eng.take(b)
    eng.prefill(sb, b, 0, False, (0.0, 1.0))
    nxt = a + [1, 2]
    sa, start, restore = eng.take(nxt)
    eng.mix_start(sa, nxt, start, restore, (0.0, 1.0))
    self.assertEqual(eng.step([sa, sb]), [[5], [5, 5, 5]])
    self.assertEqual(sb.tokens, b + [7, 5, 5, 5])
    with self.assertRaises(AssertionError): eng.mix_start(sa, nxt + list(range(40)), start, restore, (0.0, 1.0))

  def _conv(self, eng:SeqEngine, first:int, n:int=20000):
    ids = list(range(first, first + n))
    s, start, restore = eng.take(ids, room=100)
    eng.prefill(s, ids, start, restore, (0.0, 1.0))
    eng.stop(s, len(s.tokens))
    return s, ids

  def test_held_not_lost(self):
    # three agents' conversations fill the caches (no stash): a new one waits (nothing evicted) while they're held, takes the least recently
    # used one's room when it's forced or they aren't held anymore
    eng = self.engine()
    convs = [self._conv(eng, 100000 * (i + 1)) for i in range(3)]
    new = list(range(900000, 920000))
    self.assertIsNone(eng.take(new, room=100))
    self.assertEqual((len(eng.seqs), eng.evictions), (3, 0))
    # (an agent's next turn: its conversation continued)
    s, start, _ = eng.take(convs[1][1] + [1, 2, 3], room=100)
    self.assertIs(s, convs[1][0])
    self.assertEqual(start, len(convs[1][1]))
    eng.stop(s, len(s.tokens))
    # (forced: one used in the last PROTECT s isn't lost either, one idle longer is)
    self.assertIsNone(eng.take(new, room=100, force=True))
    with patch.object(SeqEngine, "PROTECT", 0.0): got = eng.take(new, room=100, force=True)
    self.assertIsNotNone(got)
    self.assertEqual((eng.evictions, eng.lost), (1, 1))
    self.assertNotIn(convs[0][0], eng.seqs)
    eng.stop(got[0], 0)
    with patch.object(SeqEngine, "STICKY", 0.0):
      self.assertIsNotNone(eng.take(list(range(700000, 720000)), room=100))

  def test_reserve(self):
    # a reply's room: its max_tokens, RESERVE at most (and without one)
    eng, ids = self.engine(), list(range(100, 3100))
    s, _, _ = eng.take(ids, room=100)
    self.assertEqual(s.cap, eng._need(len(ids) + 100))
    eng.stop(s, 0)
    s2, _, _ = eng.take(list(range(5000, 8000)))
    self.assertEqual(s2.cap, eng._need(3000 + eng.RESERVE))

class CacheModel(FakeModel):
  """a FakeModel whose caches are a row of every token: a prefill writes its tokens' rows, a move copies them"""
  def __init__(self):
    super().__init__()
    self.cache = [-1] * (self.seq_rows() + 4096)
  def seq_prefill(self, slot, base, tokens, start, hrow, temp=0.0, upto=None):
    end = len(tokens) if upto is None else upto
    self.cache[base+start:base+end] = tokens[start:end]
    return super().seq_prefill(slot, base, tokens, start, hrow, temp, upto)
  def seq_move(self, src, dst, n):
    self.cache[dst:dst+n] = self.cache[src:src+n]
    super().seq_move(src, dst, n)

class TestSeqRegions(unittest.TestCase):
  def test_moves_keep_tokens(self):
    # conversations growing turn by turn, new ones, idle ones evicted: every resident one's rows are its tokens, the regions never overlap
    import random
    rng, model = random.Random(0), CacheModel()
    model.seq_slots = lambda: 8
    eng = SeqEngine(model)
    convs:list[list[int]] = []
    with patch.object(SeqEngine, "STICKY", 0.0):
      for it in range(300):
        if convs and rng.random() < 0.7: ids = convs.pop(rng.randrange(len(convs))) + [rng.randrange(1000) for _ in range(rng.randrange(1, 3000))]
        else: ids = [it * 100000 + i for i in range(rng.randrange(100, 12000))]
        if len(ids) > 20000: continue
        if (got:=eng.take(ids, rng.choice([0, 100, 3000]))) is None: continue
        s, start, restore = got
        eng.prefill(s, ids, start, restore, (0.0, 1.0))
        eng.stop(s, len(s.tokens))
        convs.append(ids)
        regs = sorted((t.base, t.base + t.cap) for t in eng.seqs)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(regs, regs[1:])) and regs[-1][1] <= eng.rows)
        for t in eng.seqs: self.assertEqual(model.cache[t.base:t.base+t.n], t.tokens[:t.n])
    self.assertGreater(len([c for c in model.calls if c[0] == "move"]), 0)

class TestSeqMaxLen(unittest.TestCase):
  def test_no_room_past_context(self):
    # the caches' rows can be more than the model's context: a sequence's positions aren't (its reply ends at the context)
    model = FakeModel()
    model.seq_rows = lambda: 1 << 18
    eng = SeqEngine(model)
    ids = list(range(100, 100 + model.max_context - 100))
    s, start, restore = eng.take(ids)
    self.assertLessEqual(s.cap, eng._need(model.max_context))
    self.assertTrue(eng.room(s, model.max_context))
    self.assertFalse(eng.room(s, model.max_context + 1))
    self.assertFalse(eng.fits(model.max_context + 1))

class TestSeqTouch(unittest.TestCase):
  def test_waiting_turn_not_evicted(self):
    # a request waiting for the least recently used conversation: looked at (cached, touch), that one isn't the next evicted
    model = FakeModel()
    eng = SeqEngine(model)
    convs = []
    for i in range(3):
      ids = [100000 * (i + 1) + j for j in range(20000)]
      s, start, restore = eng.take(ids, 100)
      eng.prefill(s, ids, start, restore, (0.0, 1.0))
      eng.stop(s, len(s.tokens))
      convs.append((s, ids))
    self.assertEqual(eng.cached(convs[0][1] + [1, 2], touch=True), len(convs[0][1]))
    with patch.object(SeqEngine, "STICKY", 0.0): self.assertIsNotNone(eng.take(list(range(900000, 920000)), 100))
    self.assertIn(convs[0][0], eng.seqs)
    self.assertNotIn(convs[1][0], eng.seqs)

class TestSeqWindow(unittest.TestCase):
  def test_idle_neighbor_not_moved(self):
    # rows full of A (decoding), B (idle), C (decoding): a new prompt that fits B's rows evicts B, nothing moved
    model = CacheModel()
    eng = SeqEngine(model)
    seqs = []
    for i in range(3):
      ids = [100000 * (i + 1) + j for j in range(18000)]
      s, start, restore = eng.take(ids, 100)
      eng.prefill(s, ids, start, restore, (0.0, 1.0))
      seqs.append(s)
    eng.stop(seqs[1], len(seqs[1].tokens))
    model.calls.clear()
    with patch.object(SeqEngine, "STICKY", 0.0): s, _, _ = eng.take(list(range(900000, 915000)), 100)
    self.assertEqual([c for c in model.calls if c[0] == "move"], [])
    self.assertNotIn(seqs[1], eng.seqs)
    self.assertEqual(s.base, seqs[0].base + seqs[0].cap)

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

  @patch("tinygrad.llm.serve.stderr_log")
  def test_short_suffix_mixed_while_other_decodes(self, _):
    ids = list(range(100, 2100))
    seed = Request(ids, "", 2, 0.0)
    self.server.waiting.put(seed)
    self.wait([seed])
    other = Request(list(range(5000, 5700)), "", 1_000_000, 0.0)
    self.server.waiting.put(other)
    st = time.perf_counter()
    while not other.generated and time.perf_counter() - st < 10: time.sleep(0.001)
    self.assertTrue(other.generated)
    follow = Request(ids + [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008], "", 4, 0.0)
    self.server.waiting.put(follow)
    self.wait([follow])
    self.assertEqual(follow.generated, [5] * 4)
    self.assertEqual(follow.cache_start, len(ids))
    self.assertEqual(follow.prefill_time, 0)
    self.assertEqual(follow.slices, 0)
    self.assertFalse(any(c[0] == "prefill" and c[2] == len(ids) for c in self.model.calls))
    forced = [c[1] for c in self.model.calls if c[0] == "step" and c[1] is not None]
    self.assertTrue(any(any(f) for f in forced))
    self.assertEqual(self.server.stats.mixed, 1)
    other.cancelled = True
    self.wait([other])

if __name__ == "__main__":
  unittest.main()
