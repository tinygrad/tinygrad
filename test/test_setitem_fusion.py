import unittest

from tinygrad.tensor import Tensor
from tinygrad import Context

# Try tinygrad's internal counters if available; fall back to log parsing.
def count_kernels(fn) -> int:
  # 1) Preferred: tinygrad GlobalCounters (API may vary by version)
  try:
    from tinygrad.helpers import GlobalCounters
    # Reset if present
    reset = getattr(GlobalCounters, "reset", None)
    if callable(reset): reset()
    # Run workload
    fn()
    # Probe a few plausible attribute names across versions
    for attr in ("kernel_count", "kernels", "global_kernels", "kernels_launched"):
      if hasattr(GlobalCounters, attr):
        return int(getattr(GlobalCounters, attr))
    # Some versions store counters in a dict-like .stats
    stats = getattr(GlobalCounters, "stats", None)
    if isinstance(stats, dict):
      for k in ("kernel_count", "kernels", "global_kernels", "kernels_launched"):
        if k in stats:
          return int(stats[k])
  except Exception as e:
    print(f"GlobalCounters method failed: {e}")
    raise e

class TestSetitem(unittest.TestCase):
  def test_setitem_arange_single_kernel(self):
    with Context(DEBUG=4, RANGEIFY=1, BEAM=2):
      def workload():
        N = 10
        cmp = Tensor.empty(N)
        for i in range(N): cmp[i] = i
        self.assertListEqual(Tensor.arange(N).tolist(), cmp.tolist())

      kcount = count_kernels(workload)
      print(f"Kernel count (pre-fix): {kcount}")
      # Goal after the bounty: this becomes exactly 1
      assert kcount == 1
