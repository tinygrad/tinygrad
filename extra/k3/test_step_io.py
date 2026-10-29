#!/usr/bin/env python3
# extra/k3/engine.py's step_io (the host's part of a step) against fake drives: a thread completes every command rung on a queue, the
# gpus' router tags are set up front. checks the slots and weights the gpus get, the cache tables, the guesses and the counts.
# PYTHONPATH=. python3 extra/k3/test_step_io.py
import ctypes, threading, unittest
import numpy as np
from extra.k3.engine import IoT, host_c, BUFS

E, L, NGPU, K, KMIN, ND, ENTRIES, ROWS = 64, 6, 2, 8, 2, 2, 64, 2

class FakeDrives(threading.Thread):
  """every command rung on a queue completed, in order"""
  def __init__(self, sq, db, cq):
    super().__init__(daemon=True)
    self.sq, self.db, self.cq, self.stop = sq, db, cq, False
    self.head, self.tail, self.phase = [0] * len(sq), [0] * len(sq), [1] * len(sq)
  def run(self):
    while not self.stop:
      for q in range(len(self.sq)):
        while self.head[q] != int(self.db[q][0]):
          cid = int(self.sq[q][self.head[q] * 16]) >> 16
          self.head[q] = (self.head[q] + 1) % ENTRIES
          self.cq[q][self.tail[q] * 4 + 3] = cid | self.phase[q] << 16
          if (t:=self.tail[q] + 1) == ENTRIES: t, self.phase[q] = 0, self.phase[q] ^ 1
          self.tail[q] = t

class TestStepIO(unittest.TestCase):
  def setUp(self, slots=200, T=1):
    SLOTS = self.slots = slots
    rng = np.random.default_rng(0)
    self.keep:list[np.ndarray] = []
    def arr(a) -> np.ndarray:
      self.keep.append(a := np.ascontiguousarray(a))
      return a
    nq = 2 * ND
    self.sq, self.cq = [arr(np.zeros(ENTRIES * 16, np.uint32)) for _ in range(nq)], [arr(np.zeros(ENTRIES * 4, np.uint32)) for _ in range(nq)]
    self.db, self.cq_db = [arr(np.zeros(1, np.uint32)) for _ in range(nq)], [arr(np.zeros(1, np.uint32)) for _ in range(nq)]
    # a code's part on each drive: one command each
    ncodes = L * E
    tables:dict[str, np.ndarray] = {"ptr": np.arange(0, 2 * ncodes + 1, 2, dtype=np.int64), "drive": np.tile(np.arange(ND, dtype=np.int32), ncodes),
              "lba": np.zeros(2 * ncodes, np.int64), "nlb": np.zeros(2 * ncodes, np.uint32), "kind": np.zeros(2 * ncodes, np.int8),
              **{k: np.zeros(2 * ncodes, np.int64) for k in ("inpart", "page_base", "list_base")}, "part_pages": np.ones(2 * ncodes, np.int64),
              "pages": np.arange(SLOTS + 1, dtype=np.int64), "lists": np.zeros(1, np.int64),
              "sq": np.array([a.ctypes.data for a in self.sq], np.uint64), "db": np.array([a.ctypes.data for a in self.db], np.uint64),
              "cq": np.array([a.ctypes.data for a in self.cq], np.uint64), "cq_db": np.array([a.ctypes.data for a in self.cq_db], np.uint64),
              **{k: np.zeros(nq, np.int64) for k in ("tails", "db_tails", "heads", "inflight")}, "phases": np.ones(nq, np.int64),
              "owner": np.full(nq * ENTRIES, -1, np.int64)}
    self.bias = [arr(rng.normal(0, 0.1, E).astype(np.float32)) for _ in range(L)]
    self.state:dict[str, np.ndarray] = {
      "slot_of": np.full(L * E, -1, np.int32), "expert_in": np.full(SLOTS, -1, np.int64), "freq": np.zeros(L * E, np.float32),
      "flying": np.zeros(L * E, np.int32), "guess": np.zeros(L * E, np.uint8), "mask": np.zeros(SLOTS, np.uint8),
      "victims": np.zeros(32 * T, np.int64), "bias": np.array([b.ctypes.data for b in self.bias], np.uint64)}
    self.keep += [*tables.values(), *self.state.values()]
    # the gpus' buffers of every layer: tag, ready, ready1, scores (ROWS of E / NGPU a token), slots and weights (slot, token) of both parts
    nr, k1 = E // NGPU, K // 2 * T
    def gpu():
      return {"tag": arr(np.ones(1, np.uint32)), "ready": arr(np.zeros(1, np.uint32)), "ready1": arr(np.zeros(1, np.uint32)),
              "scores": arr(rng.normal(0, 2, (T * ROWS, nr)).astype(np.float32)), "slots": arr(np.zeros(k1, np.int32)),
              "ws": arr(np.zeros(k1 * T, np.float32)), "slots2": arr(np.zeros(K * T - k1, np.int32)),
              "ws2": arr(np.zeros((K * T - k1) * T, np.float32))}
    self.gbuf = [[gpu() for _ in range(NGPU)] for _ in range(L)]
    self.bufs = arr(np.array([[[g[b].ctypes.data for g in self.gbuf[li]] for b in BUFS] for li in range(L)], np.uint64))
    self.io = io = IoT(layers=L, E=E, ngpu=NGPU, k=K, kmin=KMIN, kmax=K, rows=ROWS, slots=SLOTS, tokens=T, k1=k1, nd=ND, entries=ENTRIES,
                       victims_n=32 * T, pf=3, tau=0.03, decay=0.5 ** (1 / 8), bufs=self.bufs.ctypes.data)
    for name, a in {**tables, **self.state}.items(): setattr(io, name, a.ctypes.data)
    self.drives = FakeDrives(self.sq, self.db, self.cq)
    self.drives.start()
  def tearDown(self): self.drives.stop = True

  def test_small_cache(self):  # (the slots go to later layers within a step: only the tables and counts)
    self.tearDown()
    self.setUp(slots=40)
    for _ in range(5):
      self.step()
      self.check_tables()
    self.check_counts()

  def step(self):
    for li in range(L):
      for g in self.gbuf[li]: g["ready"][0] = g["ready1"][0] = 0
    self.assertEqual(host_c().step_io(ctypes.byref(self.io)), 0)

  def check_tables(self):
    slot_of, expert_in = self.state["slot_of"], self.state["expert_in"]
    for sl, c in enumerate(expert_in):
      if c >= 0: self.assertEqual(slot_of[c], sl)
    for c in np.nonzero(slot_of >= 0)[0]: self.assertEqual(expert_in[slot_of[c]], c)

  def check_counts(self):
    io = self.io
    self.assertEqual(io.refs, io.hits + io.late + io.misses)
    self.assertGreater(io.guesses, 0)
    self.assertGreater(io.hits, 0)
    # every guess is of a layer of the step: settled (used or not) by its end
    self.assertEqual(io.used + io.wasted, io.guesses)
    self.assertEqual(int(self.state["guess"].sum()), 0)

  def test_steps(self):
    for _ in range(5):
      self.step()
      self.check_tables()
      for li in range(L):
        so = self.state["slot_of"][li * E:(li + 1) * E]
        for g in self.gbuf[li]:
          self.assertEqual((g["ready"][0], g["ready1"][0]), (1, 1))
          slots, ws = np.concatenate([g["slots"], g["slots2"]]), np.concatenate([g["ws"], g["ws2"]])
          self.assertAlmostEqual(float(ws.sum()), 1.0, places=5)
          # every slot given with a weight holds one of this layer's experts
          self.assertTrue(all(self.state["expert_in"][s] // E == li for s, w in zip(slots, ws) if w > 0))
          self.assertEqual(len(set(slots[ws > 0].tolist())), int((ws > 0).sum()))
          self.assertTrue(all(so[self.state["expert_in"][s] % E] == s for s in slots[ws > 0]))
      # the gpus get the same slots and weights
      for li in range(L): self.assertTrue(all((self.gbuf[li][0][b] == self.gbuf[li][1][b]).all() for b in ("slots", "ws", "slots2", "ws2")))
    self.check_counts()

  def test_tokens(self):  # a step of 2 tokens: every token its own entries (i*T + t), a weight each
    self.tearDown()
    self.setUp(T=2)
    for _ in range(3):
      self.step()
      self.check_tables()
      for li in range(L):
        g = self.gbuf[li][0]
        slots, ws = np.concatenate([g["slots"], g["slots2"]]), np.concatenate([g["ws"], g["ws2"]]).reshape(-1, 2)
        self.assertTrue(np.allclose(ws.sum(0), 1, atol=1e-5))
        self.assertTrue(all(self.state["expert_in"][s] // E == li for s, w in zip(slots, ws.sum(1)) if w > 0))
    self.check_counts()

  def test_fill(self):  # the experts tau skipped for not being cached are read: kept by the next step (same scores)
    self.io.tau, self.io.pf, self.io.fill = 0.2, 0, 4
    self.step()
    refs, fills = self.io.refs, self.io.fills
    self.assertGreater(fills, 0)
    self.step()
    self.check_tables()
    self.assertGreater(self.io.refs - refs, refs)

  def test_kmax(self):  # at most kmax experts a token
    self.io.kmax = 5
    self.step()
    self.check_tables()
    for li in range(L):
      g = self.gbuf[li][0]
      self.assertLessEqual(int((np.concatenate([g["ws"], g["ws2"]]) > 0).sum()), 5)

  def test_dry(self):
    self.io.dry = 1
    self.step()
    self.assertEqual(self.io.misses + self.io.guesses, 0)
    self.assertTrue((self.state["slot_of"] < 0).all())
    for li in range(L): self.assertEqual(self.gbuf[li][0]["slots"].tolist(), list(range(K // 2)))

if __name__ == "__main__":
  unittest.main()
