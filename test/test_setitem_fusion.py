import os
import io
import re
import pytest
from contextlib import redirect_stdout, redirect_stderr

# Ensure DEBUG is set before tinygrad imports (some logging is configured at import time)
os.environ.setdefault("DEBUG", "4")

from tinygrad.tensor import Tensor

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
  except Exception:
    pass

  # 2) Fallback: capture stdout+stderr and count kernel-ish lines
  old_debug = os.environ.get("DEBUG")
  os.environ["DEBUG"] = "4"  # make sure it's loud during the run

  buf = io.StringIO()
  with redirect_stdout(buf), redirect_stderr(buf):
    fn()
  out = buf.getvalue()

  # Count lines that look like compile/launch/codegen (backend-agnostic heuristic)
  kernelish = [
      l for l in out.splitlines()
      if re.search(r"\b(launch|launched|compile|compiled|kernel|codegen|program)\b", l, re.I)
  ]

  if old_debug is None:
    del os.environ["DEBUG"]
  else:
    os.environ["DEBUG"] = old_debug

  print("\n--- captured debug (first 30 lines) ---")
  for l in kernelish[:30]:
    print(l)
  print(f"--- total kernel-ish lines: {len(kernelish)} ---\n")
  return len(kernelish)

@pytest.mark.xfail(reason="Pre-fix: setitem realizes / launches multiple kernels")
def test_setitem_arange_single_kernel():
  def workload():
    # Make a realized contiguous buffer (avoid broadcasted zero w/ stride=0)
    x = Tensor.empty(100).realize()
    x *= 0   # zero without changing contiguity

    for i in range(10):
      x[i*10:(i+1)*10] = Tensor.arange(10)

    _ = x.numpy()  # force execution

  kcount = count_kernels(workload)
  print(f"Kernel count (pre-fix): {kcount}")
  # Goal after the bounty: this becomes exactly 1
  assert kcount == 1
