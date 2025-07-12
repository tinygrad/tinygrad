import unittest
import ctypes
import timeit
import numpy as np
from tinygrad.helpers import CI

def to_mv_old(ptr: int, sz: int) -> memoryview:
  return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")

def to_mv_new(ptr: int, sz: int) -> memoryview:
  return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")

class TestMemoryview(unittest.TestCase):
  @unittest.skipIf(CI, "dangerous for CI, it allocates tons of memory")
  def test_to_mv(self):
    sizes = [
      (16, "16 B"),
      (64, "64 B"),
      (256, "256 B"),
      (1024, "1 KB"),
      (4 * 1024, "4 KB"),
      (16 * 1024, "16 KB"),
      (64 * 1024, "64 KB"),
      (256 * 1024, "256 KB"),
      (1 * 1024 * 1024, "1 MB"),
      (10 * 1024 * 1024, "10 MB"),
      (200 * 1024 * 1024, "200 MB"),
    ]

    for sz, label in sizes:
      buf = np.random.randint(0, 256, sz, dtype=np.uint8)
      ptr = buf.ctypes.data

      mv_old = to_mv_old(ptr, sz)
      mv_new = to_mv_new(ptr, sz)

      self.assertEqual(mv_old.tobytes(), mv_new.tobytes(), f"Mismatch content at size {sz}")

      iters = 100_000
      t_old = timeit.timeit(lambda: to_mv_old(ptr, sz), number=iters)
      t_new = timeit.timeit(lambda: to_mv_new(ptr, sz), number=iters)

      us_old = (t_old / iters) * 1e6
      us_new = (t_new / iters) * 1e6
      speedup = float('inf') if us_new == 0 else us_old / us_new

      print(f"Size {label:>9} | Old: {us_old:8.3f} µs | New: {us_new:8.3f} µs | Speedup: {speedup:5.2f}x")

if __name__ == "__main__":
  unittest.main()
