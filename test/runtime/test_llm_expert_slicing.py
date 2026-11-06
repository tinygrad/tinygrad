import unittest
from collections import Counter
from tinygrad.llm.qwen4exp import expert_slicing, expert_slice_table

class TestExpertSlicing(unittest.TestCase):
  def test_balanced_complete_partition(self):
    for n in range(1, 9):
      for slices in range(1, 25):
        for experts in range(1, 25):
          parts = expert_slicing(experts, slices, n)
          if experts*slices % n:
            self.assertIsNone(parts)
            continue
          self.assertIsNotNone(parts)
          self.assertEqual(sorted(u for part in parts for u in part), list(range(experts*slices)))
          counts = []
          for part in parts:
            self.assertEqual(len(part), experts*slices//n)
            counts.append(Counter(u//slices for u in part))
            table = expert_slice_table(part, experts, slices)
            for e, row in enumerate(table):
              self.assertEqual([part[i] for i in row if i >= 0], [u for u in part if u//slices == e])
          for e in range(experts): self.assertLessEqual(max(c[e] for c in counts)-min(c[e] for c in counts), 1)

  def test_legacy_order(self):
    for n, slices, experts in ((6, 20, 513), (6, 3, 512), (4, 6, 512), (6, 12, 512)):
      base, r = divmod(slices, n)
      expected = [[] for _ in range(n)]
      for e in range(experts):
        extra = set(range((e % (n//r))*r, (e % (n//r))*r+r)) if r else set()
        devs = [d for d in range(n) if d in extra for _ in range(base+1)] + [d for d in range(n) if d not in extra for _ in range(base)]
        for j, d in enumerate(devs): expected[d].append(e*slices+j)
      self.assertEqual(expert_slicing(experts, slices, n), expected)

if __name__ == "__main__": unittest.main()
