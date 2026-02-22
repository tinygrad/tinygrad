import unittest, contextlib
from tinygrad import Device, Tensor, Context, TinyJit
from tinygrad.viz.serve import load_amd_counters

@contextlib.contextmanager
def save_sqtt():
  Device[Device.DEFAULT].synchronize()
  Device[Device.DEFAULT].profile_events.clear()
  yield (ret:=[])
  load_amd_counters(ret, Device[Device.DEFAULT].profile_events)
  ret[:] = [r for r in ret if r["name"].startswith("Exec")]

@unittest.skipUnless(Device.DEFAULT == "AMD", "only runs on AMD")
class TestSQTTProfiler(unittest.TestCase):
  # TODO: can we enable SQTT profiling in context?
  @classmethod
  def setUpClass(cls):
    if not Device[Device.DEFAULT].sqtt_enabled: raise unittest.SkipTest("device must be in SQTT profiling mode")

  def test_simple(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as sqtt:
      ei = t.schedule()[0].lower()
      ei.run()
    self.assertEqual(len(sqtt), 1)
    self.assertEqual(sqtt[0]["name"], f"Exec {ei.prg.p.function_name}")

  def test_multiple_runs(self):
    t = Tensor.empty(1) + 1
    with save_sqtt() as sqtt:
      ei = t.schedule()[0].lower()
      for _ in range(N:=3):
        ei.run()
    self.assertEqual(len(sqtt), N)
    for i in range(1, N):
      self.assertEqual(sqtt[i]["name"], f"Exec {ei.prg.p.function_name} n{i+1}")

  def test_multiple_kernels(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    sched = t.schedule()
    with save_sqtt() as sqtt:
      for si in sched: si.lower().run()
    self.assertEqual(len(sqtt), len(sched))
    for i,k in enumerate(sched):
      self.assertEqual(sqtt[i]["name"], f"Exec {k.lower().prg.p.function_name}")

  def test_multiple_kernels(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    sched = t.schedule()
    with save_sqtt() as sqtt:
      for si in sched: si.lower().run()
    self.assertEqual(len(sqtt), len(sched))
    for i,k in enumerate(sched):
      self.assertEqual(sqtt[i]["name"], f"Exec {k.lower().prg.p.function_name}")

  def test_multiple_kernels_lower(self):
    t = ((Tensor.empty(1) + 1).contiguous() + 2)
    sched = t.schedule()
    with save_sqtt() as sqtt:
      prgs = [si.lower() for si in sched]
      for p in prgs: p.run()
    self.assertEqual(len(sqtt), len(sched))
    for i,ei in enumerate(prgs):
      self.assertEqual(sqtt[i]["name"], f"Exec {ei.prg.p.function_name}")

  def test_jit(self):
    @TinyJit
    def f(a): return a + 1
    t = Tensor.empty(1)
    with save_sqtt() as sqtt:
      for _ in range(N:=5):
        f(t).realize()
    self.assertEqual(len(sqtt), N)
    kernel_name = sqtt[0]["name"]
    for i,s in enumerate(sqtt[1:], start=1): self.assertEqual(s["name"], f"{kernel_name} n{i+1}")

  # TODO: can we trace SQTT for graphed kernels?
  def test_jit_graph(self):
    #@TinyJit
    def f(a): return ((a + 1).contiguous() + 2).contiguous().sum()
    t = Tensor.empty(32)
    with save_sqtt() as sqtt:
      for _ in range(N:=5):
        f(t).realize()
    # there's only 2 SQTT events per kernel, before they enter the graph
    #self.assertEqual(len(sqtt), 3*2)
    names = [s["name"] for s in sqtt]
    k0, k1, k2 = names[:3]
    for i in range(3, len(sqtt), 3):
      n = (i // 3)+1
      self.assertEqual(names[i], f"{k0} n{n}")
      self.assertEqual(names[i+1], f"{k1} n{n}")
      self.assertEqual(names[i+2], f"{k2} n{n}")

if __name__ == "__main__":
  unittest.main()
