#!/usr/bin/env python3
# a start's graphs kept for the next start (persist.py) on the toy engine of test_engine.py, its host's part stubbed: a second engine (fresh
# buffers) loads the first's file and steps as it does; a file of other buffers (a mapping perturbed) is caught by its dry runs and dropped
# DEV=CPU PYTHONPATH=. python3 extra/k3/test_persist.py
import tempfile, unittest
from unittest.mock import patch
import numpy as np
from extra.k3 import persist
import extra.k3.test_engine as te

def engine(t:te.TestEngine, slots:int=2 * te.E):  # (a fresh engine on a fresh model of t's trunk gguf (its graphs and scorers kept of it))
  t.eng = t.engine(slots, source=str(t.trunk))
  t.stub()
  return t.eng

def steps(eng) -> list[np.ndarray]:  # (a jitted step at each position, the next token the step's)
  out, tok = [], 3
  for pos in range(4):
    eng.tok_np[:1], eng.given_np[:1] = [tok], [-1]
    eng.jits[1](eng.v_pos.bind(pos))
    out.append(eng.res_np.copy())
    tok = int(eng.res_np[1])
  return out

class TestPersist(unittest.TestCase):
  def setUp(self):
    self.dir = tempfile.mkdtemp()
    self.patch = patch.object(persist, "cache_dir", self.dir)
    self.patch.start()
    self.t = te.TestEngine("test_forward")
    self.t.setUp()
  def tearDown(self): self.patch.stop()

  def test_saved_and_loaded(self):
    eng = engine(self.t)
    self.assertEqual(eng.saved, {})
    eng.warm()
    self.assertTrue(persist.path(eng).exists())
    want = steps(eng)
    eng2 = engine(self.t)  # (its window's graphs the file's, checked, its scorers the first's)
    self.assertEqual(set(eng2.saved["graphs"]), set(eng.graphs()))
    for a, b in zip(eng.scorers, eng2.scorers): np.testing.assert_equal(a.numpy(), b.numpy())
    eng2.warm()
    self.assertNotEqual(eng2.saved, {})
    for a, b in zip(steps(eng2), want): np.testing.assert_equal(a, b)

  def test_save_at_lookup_key(self):
    with patch.object(persist, "key", return_value="before"):
      eng = engine(self.t)
    with patch.object(persist, "key", return_value="packed"):
      eng.warm()
    with patch.object(persist, "key", return_value="before"):
      again = engine(self.t)
    self.assertTrue(again.saved)

  def test_other_config(self):  # another engine's key (here its pool's slots): its own file, not the first's
    eng = engine(self.t)
    eng.warm()
    self.assertEqual(persist.path(again:=engine(self.t)), persist.path(eng))
    self.assertNotEqual(again.saved, {})
    other = engine(self.t, 2 * te.E + 1)
    self.assertNotEqual(persist.path(other), persist.path(eng))
    self.assertEqual(other.saved, {})

  def test_perturbed(self):  # the file loaded on two buffers swapped: its dry runs aren't the saved ones, it's dropped and the graphs captured
    eng = engine(self.t)
    eng.warm()
    want = steps(eng)
    buffers = persist.buffers
    def swapped(e):
      out = buffers(e)
      for suffix in ("", ":uop"):
        a, b = [i for i, (n, _) in out.items() if n in (f"model.blk.0.ffn_gate.weight#0{suffix}", f"model.blk.0.ffn_up.weight#0{suffix}")]
        out[a], out[b] = (out[a][0], out[b][1]), (out[b][0], out[a][1])
      return out
    with patch.object(persist, "buffers", swapped): eng2 = engine(self.t)
    self.assertNotEqual(eng2.saved, {})
    eng2.warm()
    self.assertEqual(eng2.saved, {})
    for a, b in zip(steps(eng2), want): np.testing.assert_equal(a, b)

  def test_window_mismatch(self):  # the window's route not as saved: the file's dropped, every graph captured again (the window's on its state lent
    # again), the steps as the first engine's, a file saved again
    eng = engine(self.t)
    eng.warm()
    want = steps(eng)
    eng2 = engine(self.t)
    fingerprints, calls = eng2.fingerprints, []
    def other() -> dict[str, str]:
      calls.append(fps:=fingerprints())
      return {**fps, "route": "another start's"} if len(calls) == 1 else fps
    eng2.fingerprints = other  # type: ignore[method-assign]
    eng2.warm()
    self.assertEqual((eng2.saved, len(calls)), ({}, 2))
    self.assertTrue(persist.path(eng2).exists())
    for a, b in zip(steps(eng2), want): np.testing.assert_equal(a, b)

if __name__ == "__main__":
  unittest.main()
