import pickle, unittest
from collections import Counter
from pathlib import Path

from extra.nv_pma.decode import decode

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

def decode_and_aggregate(raw_dumps: list[bytes]) -> Counter[tuple[int, int]]:
  """Decode all PMA buffers and aggregate by (relative_pc, stall_reason)."""
  all_samples = [s for raw in raw_dumps for s in decode(raw)]
  if not all_samples: return Counter()
  base_pc = min(s.pc_offset for s in all_samples)
  return Counter((s.pc_offset - base_pc, int(s.stall_reason)) for s in all_samples)

def cupti_to_counter(cupti_records: list[dict]) -> Counter[tuple[int, int]]:
  """Convert CUPTI records to Counter[(pcOffset, stallReason)]."""
  counter: Counter[tuple[int, int]] = Counter()
  for r in cupti_records:
    counter[(r['pcOffset'], r['stallReason'])] += r['samples']
  return counter

class TestNVProf(unittest.TestCase):
  def _test_example(self, name: str):
    pkl_file = EXAMPLES_DIR / f"{name}.pkl"
    if not pkl_file.exists():
      self.skipTest(f"Example data not found: {pkl_file}. Run collect.py first.")

    with open(pkl_file, "rb") as f:
      data = pickle.load(f)

    self.assertEqual(data["test_name"], name)
    pma_agg = decode_and_aggregate(data["pma_raw_dumps"])
    cupti_agg = cupti_to_counter(data["cupti_pc_samples"])

    total = sum(cupti_agg.values())
    mismatched = sum(abs(pma_agg.get(k, 0) - v) for k, v in cupti_agg.items())
    mismatched += sum(v for k, v in pma_agg.items() if k not in cupti_agg)
    mismatched //= 2

    print(f"\n=== Test: {name} ===")
    print(f"Total samples: {total}, Mismatched: {mismatched} ({mismatched/total*100 if total else 0:.1f}%)")

    self.assertEqual(pma_agg, cupti_agg, f"PMA: {dict(pma_agg)}\nCUPTI: {dict(cupti_agg)}")

  def test_decode_test_plus(self): self._test_example("test_plus")
  def test_decode_test_reduce_sum(self): self._test_example("test_reduce_sum")
  def test_decode_test_broadcast(self): self._test_example("test_broadcast")
  def test_decode_test_matmul(self): self._test_example("test_matmul")
  def test_decode_test_plus_big(self): self._test_example("test_plus_big")
  def test_decode_test_elementwise_chain(self): self._test_example("test_elementwise_chain")
  def test_decode_test_conv2d(self): self._test_example("test_conv2d")
  def test_decode_test_large_matmul(self): self._test_example("test_large_matmul")

if __name__ == "__main__":
  unittest.main()
