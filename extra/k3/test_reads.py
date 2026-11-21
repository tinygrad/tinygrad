#!/usr/bin/env python3
# on a box with a store: the host's reads (host.py, read_codes) of expert codes into slots of each gpu's pool on the drives' host-owned
# queues, against master's static copies of the same parts, and the time of a burst. never writes the drives
# DEV=AMD PYTHONPATH=. python3 extra/k3/test_reads.py (STORE_AT: the manifest's offset on disk 0)
import ctypes, time, unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from extra.k3.store import open_store, read_disk, part_off
from extra.k3.host import IoT, host_c, reads

GPUS, AT, SLOTS = [f"AMD:{g}" for g in range(getenv("GPUS", 4))], getenv("STORE_AT", 0), 32

class TestReads(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.disks, cls.m = open_store(GPUS, AT)
    cls.ncodes, cls.parts = len(cls.m["layers"]) * cls.m["n_experts"], cls.m["part_bytes"]
    cls.pools = [Tensor.zeros(SLOTS * p, dtype=dtypes.uint8, device=g).contiguous().realize() for g, p in zip(GPUS, cls.parts)]
    st = time.perf_counter()
    cls.tables, scalars = reads([(cls.m, cls.disks, [(li, cls.parts[0]) for li in range(len(cls.m["layers"]))])],
                                [p.uop.buffer for p in cls.pools], cls.parts[0])
    print(f"{cls.ncodes} codes: {len(cls.tables['lba'])} commands, tables in {time.perf_counter() - st:.2f} s")
    cls.io = IoT(**scalars)
    for name, a in cls.tables.items(): setattr(cls.io, name, a.ctypes.data)

  def read(self, codes:list[int], slots:list[int]) -> float:
    c, s = np.array(codes, np.int64), np.array(slots, np.int64)
    st = time.perf_counter()
    self.assertEqual(host_c().read_codes(ctypes.byref(self.io), len(c), c.ctypes.data, s.ctypes.data), 0)
    return time.perf_counter() - st

  def check(self, codes:list[int], slots:list[int]):
    for g, (disk, pool, part) in enumerate(zip(self.disks, self.pools, self.parts)):
      got = pool.numpy()
      for c, s in zip(codes, slots):
        np.testing.assert_equal(got[s * part:(s + 1) * part], read_disk(disk, part_off(self.m, g, c), part, GPUS[g]), err_msg=f"{c} in {s}")

  def test_reads(self):
    codes, slots = [0, 1, 777, self.ncodes - 1], [3, 0, SLOTS - 1, 17]
    self.read(codes, slots)
    self.check(codes, slots)

  def test_burst(self): # n random experts into n slots on every gpu at once
    rng = np.random.default_rng(0)
    for n in (1, 4, 16, 32):
      ts = [self.read(rng.choice(self.ncodes, n, replace=False).tolist(), list(range(n))) for _ in range(10)]
      print(f"a burst of {n:2d} experts ({n * sum(self.parts) / 1e6:.0f} MB): median {np.median(ts) * 1e6:.0f} us, "
            f"{n * sum(self.parts) / np.median(ts) / 1e9:.1f} GB/s")
    codes = rng.choice(self.ncodes, SLOTS, replace=False).tolist()
    self.read(codes, list(range(SLOTS)))
    self.check(codes, list(range(SLOTS)))

if __name__ == "__main__":
  unittest.main()
