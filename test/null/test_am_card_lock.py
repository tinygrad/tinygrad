import unittest, subprocess, sys, time, uuid, os
from unittest.mock import patch, mock_open
from tinygrad.runtime.support.am.amdev import CardLock, in_vm

HOLD = "import sys, time; from tinygrad.runtime.support.am.amdev import CardLock\n" \
       "with CardLock(sys.argv[1]).hold():\n  print('held', flush=True); time.sleep(float(sys.argv[2]))\n"

class TestCardLock(unittest.TestCase):
  def setUp(self): self.bus = f"test-{uuid.uuid4().hex[:8]}"
  def tearDown(self):
    if os.path.exists(p:=f"/tmp/am_{self.bus}.smu.lock"): os.unlink(p)

  def test_reentrant_in_a_process(self):
    lock = CardLock(self.bus)
    with lock.hold():
      with lock.hold(wait=0.1): self.assertEqual(lock.depth, 2)
    self.assertEqual(lock.depth, 0)

  def test_another_process_waits_or_times_out(self):
    p = subprocess.Popen([sys.executable, "-c", HOLD, self.bus, "1.0"], stdout=subprocess.PIPE, env={**os.environ, "PYTHONPATH": "."})
    try:
      self.assertEqual(p.stdout.readline().strip(), b"held")
      st = time.perf_counter()
      with self.assertRaises(TimeoutError):
        with CardLock(self.bus).hold(wait=0.2): pass
      self.assertLess(time.perf_counter() - st, 0.8)
      # (no wait limit: it gets the lock once the other process lets go)
      with CardLock(self.bus).hold(): self.assertGreater(time.perf_counter() - st, 0.5)
    finally: p.wait()

  def test_released_after_the_block(self):
    with CardLock(self.bus).hold(): pass
    out = subprocess.run([sys.executable, "-c", HOLD, self.bus, "0"], capture_output=True, timeout=30, env={**os.environ, "PYTHONPATH": "."})
    self.assertEqual(out.stdout.strip(), b"held")

class TestInVM(unittest.TestCase):
  def test_hypervisor_flag(self):
    for text, want in (("flags\t: fpu vme hypervisor lahf_lm\n", True), ("flags\t: fpu vme lahf_lm\n", False)):
      in_vm.cache_clear()
      with patch("builtins.open", mock_open(read_data=text)): self.assertEqual(in_vm(), want)
    in_vm.cache_clear()

if __name__ == "__main__":
  unittest.main()
