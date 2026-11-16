import os, subprocess, sys, textwrap, threading, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _run(body:str, timeout:float=30) -> tuple[subprocess.CompletedProcess, float]:
  # a python that installs the server's SIGTERM handler, prints READY, sends itself SIGTERM and waits
  code = textwrap.dedent(f"""
    import atexit, os, signal, sys, time
    import tinygrad.llm.cli as cli
    cli.STOP_DEADLINE_S = 2
    signal.signal(signal.SIGTERM, cli._on_sigterm)
    {body}
    print("READY", flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(60)
  """)
  st = time.perf_counter()
  p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout,
                     env={**os.environ, "PYTHONPATH": ROOT, "DEV": "NULL"})
  return p, time.perf_counter() - st

class TestStop(unittest.TestCase):
  def test_sigterm_exits(self):
    p, dt = _run("")
    self.assertIn("READY", p.stdout)
    self.assertEqual(p.returncode, 0)
    self.assertLess(dt, 15)

  def test_sigterm_hung_exit_ends_with_stacks(self):
    # (an exit hook that never returns, as a device's finalize that hangs: the process still ends at the deadline, every thread's stack logged)
    p, dt = _run("atexit.register(lambda: time.sleep(1000))")
    self.assertIn("READY", p.stdout)
    self.assertNotEqual(p.returncode, 0)
    self.assertLess(dt, 20)
    self.assertIn("Timeout", p.stderr)
    self.assertIn("File", p.stderr)

  def test_sigterm_to_the_group_with_compile_workers(self):
    # (the launchers' stop signals the whole process group: the compile workers die with the server; an idle one held the pool's queue lock,
    # and the pool's exit finalizer waited for it forever (g3, 13:37 Nov 16, py-spy: Pool._help_stuff_finish))
    code = textwrap.dedent("""
      import os, signal, time
      import tinygrad.llm.cli as cli
      from tinygrad.engine.worker import get_worker_pool
      cli.STOP_DEADLINE_S = 5
      pool = get_worker_pool()
      assert pool is not None and pool.map(abs, [-1, -2, -3, -4]) == [1, 2, 3, 4]
      import atexit, multiprocessing
      atexit.register(lambda: print(f"WORKERS LEFT {len(multiprocessing.active_children())}", flush=True))
      time.sleep(0.5)
      print("READY", flush=True)
      # (the server's thread is busy when the signal comes: its handler runs late, after the pool replaced its dead workers)
      signal.signal(signal.SIGTERM, lambda *a: (time.sleep(3), cli._on_sigterm(*a)))
      os.killpg(0, signal.SIGTERM)
      time.sleep(60)
    """)
    st = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, start_new_session=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "DEV": "NULL", "PARALLEL": "4"})
    dt = time.perf_counter() - st
    self.assertIn("READY", p.stdout)
    self.assertEqual(p.returncode, 0, p.stderr[-2000:])
    self.assertIn("WORKERS LEFT 0", p.stdout)  # (none replaced behind the kill)
    self.assertLess(dt, 8)

class TestAMIdleClocksStop(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.ops_amd import AMIdleClocks
    self.c = AMIdleClocks
    self.saved = (self.c.devs, self.c.low, self.c.last, self.c.started, self.c.stopped)
  def tearDown(self):
    self.c.devs, self.c.low, self.c.last, self.c.started, self.c.stopped = self.saved

  def test_stop_ends_the_loop_and_wake(self):
    calls = []
    class Dev:  # (a device whose timeline wake would make: wake must not touch it once stopped)
      @property
      def __dict__(self):
        calls.append("dict")
        return {}
    self.c.devs, self.c.low, self.c.stopped = [(None, Dev())], True, False
    t = threading.Thread(target=self.c._loop, args=(0.05,), daemon=True)
    t.start()
    self.c.stop()
    t.join(timeout=2)
    self.assertFalse(t.is_alive())
    self.assertTrue(self.c.stopped)
    self.c.wake()
    self.assertEqual(calls, [])
    self.assertTrue(self.c.lock.acquire(timeout=1))
    self.c.lock.release()

  def test_stop_waits_for_a_message_in_flight(self):
    self.c.stopped = False
    self.c.lock.acquire()
    threading.Timer(0.3, self.c.lock.release).start()
    st = time.perf_counter()
    self.c.stop()
    self.assertGreaterEqual(time.perf_counter() - st, 0.25)

if __name__ == "__main__":
  unittest.main()
