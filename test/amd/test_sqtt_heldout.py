#!/usr/bin/env python3
"""Emulator SQTT timing on held-out kernels: random SRAM probes captured on a 7900 XTX, never used to fit the model.

Each seed must keep at least its recorded number of exact deltas; a timing change that helps the fitted
examples but costs held-out accuracy fails here.

Run with: DEV=MOCK+AMD python -m pytest test/amd/test_sqtt_heldout.py
"""
import unittest
from extra.sqtt.heldout.score import CAPTURES, score

# seed -> (exact deltas at least, scored deltas)
FLOOR = {1: (76, 94), 2: (61, 84), 3: (72, 104), 4: (72, 108), 5: (77, 100), 6: (68, 90), 7: (81, 108), 8: (72, 102),
         9: (83, 112), 10: (68, 90), 11: (85, 116)}

class TestSQTTHeldout(unittest.TestCase):
  def test_seeds(self):
    for seed, (want, n) in FLOOR.items():
      with self.subTest(seed=seed):
        hit, got_n, _ = score(CAPTURES / f"seed{seed}.pkl")
        self.assertEqual(got_n, n, f"seed {seed}: scored deltas changed (a wave's packet sequence moved)")
        self.assertGreaterEqual(hit, want, f"seed {seed}: {hit}/{n} exact, recorded {want}")

if __name__ == "__main__":
  unittest.main()
