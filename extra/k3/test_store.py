#!/usr/bin/env python3
# extra/k3/store.py's layout and its probe of the drives' order against fake drives (numpy, no hardware)
# PYTHONPATH=. python3 extra/k3/test_store.py
import json, unittest
from unittest.mock import patch
import numpy as np
import extra.k3.store as store
from extra.k3.store import STRIPE, EXPERTS, pieces, part_off

def raid(drives:list[np.ndarray]) -> np.ndarray: # the disk of the drives: stripe s on drive s % nd
  return np.stack([d.reshape(-1, STRIPE) for d in drives], 1).reshape(-1)

class TestLayout(unittest.TestCase):
  def test_pieces_read_the_disk(self):
    rng = np.random.default_rng(0)
    for nd in (1, 3, 4):
      drives = [rng.integers(0, 256, 5 * STRIPE, np.uint8) for _ in range(nd)]
      disk = raid(drives)
      for off, size in [(0, 4096), (STRIPE - 4096, 8192), (3 * STRIPE + 12288, 7 * STRIPE + 4096), (0, len(disk))]:
        ps = pieces(off, size, nd)
        self.assertTrue(all(0 < n <= STRIPE and (o % STRIPE + n) <= STRIPE for _, o, n in ps))
        np.testing.assert_equal(np.concatenate([drives[d][o:o + n] for d, o, n in ps]), disk[off:off + size])

  def test_part_off(self):
    m = {"experts_off": EXPERTS, "part_bytes": [3 << 20, 1 << 20]}
    self.assertEqual([part_off(m, g, c) for g in (0, 1) for c in (0, 5)], [EXPERTS, EXPERTS + (15 << 20), EXPERTS, EXPERTS + (5 << 20)])

class TestOpenStore(unittest.TestCase):
  def test_finds_the_order(self): # disks of 4 drives on 2 gpus, plugged in some other order: the order that reads the store back
    rng, nd, part, lat, units = np.random.default_rng(1), 4, 3 * STRIPE + 4096 * 25, 2560, 1400 # (gate and up: 2.7 MB)
    m = {"block": [256, 98], "units": [units, units], "lat": lat, "part_bytes": [part, part], "experts_off": EXPERTS, "expert_type": "IQ3_XXS"}
    def disk(g:int) -> np.ndarray: # the manifest (disk 0), then parts of quant bytes whose blocks start with an fp16 scale under 1
      d = rng.integers(0, 256, EXPERTS + 4 * part, np.uint8)
      if g == 0: d[:EXPERTS] = np.frombuffer(json.dumps(m).encode().ljust(EXPERTS, b"\0"), np.uint8)
      for i in range(4):
        rows = d[EXPERTS + i * part:][:2 * units * (lat // 256 * 98)].reshape(-1, 98)
        rows[:, :2] = rng.uniform(-0.5, 0.5, len(rows)).astype(np.float16).view(np.uint8).reshape(-1, 2)
      return np.pad(d, (0, -len(d) % (nd * STRIPE)))
    drives:dict[str, np.ndarray] = {}
    orders = [[2, 0, 3, 1], [1, 3, 0, 2]]
    for g, order in enumerate(orders):
      local = raid_inverse(disk(g), nd)
      for i in range(nd): drives[f"NVME:{g * nd + order[i]}"] = local[i]
    def read(reads, dev): return [drives[d][o:o + n] for d, o, n in reads]
    with patch.object(store, "read", read), patch.object(store, "disks", lambda n: [[f"NVME:{g * nd + i}" for i in range(nd)] for g in range(n)]):
      found, mf = store.open_store(["AMD:0", "AMD:1"])
    self.assertEqual(mf, m)
    self.assertEqual(found, [[f"NVME:{g * nd + i}" for i in order] for g, order in enumerate(orders)])

def raid_inverse(disk:np.ndarray, nd:int) -> list[np.ndarray]: # the drives of a disk
  return [s.reshape(-1) for s in np.moveaxis(disk.reshape(-1, nd, STRIPE), 1, 0)]

if __name__ == "__main__":
  unittest.main()
