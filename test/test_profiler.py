import unittest, struct, contextlib, statistics, time
from tinygrad import Device, Tensor, dtypes, TinyJit
from tinygrad.helpers import CI, getenv, Context
from tinygrad.device import Buffer, BufferSpec, Compiled, ProfileRangeEvent, ProfileDeviceEvent, ProfileGraphEvent
from tinygrad.runtime.support.hcq import HCQCompiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import get_runner

MOCKGPU = getenv("MOCKGPU")

@contextlib.contextmanager
def helper_collect_profile(*devs):
  Compiled.profile_events = []

  profile_list = []
  with Context(PROFILE=1):
    try: yield profile_list
    finally:
      for dev in devs: dev.synchronize()
      for dev in devs: dev._at_profile_finalize()
      for x in Compiled.profile_events: profile_list.append(x)

def helper_profile_filter_node(profile, **kwargs):
  assert len(profile) > 0, "Empty profile"
  assert 'traceEvents' in profile, "traceEvents should present"
  return [x for x in profile['traceEvents'] if all(x.get(k, None) == v for k,v in kwargs.items())]

def helper_profile_parse_pids(profile):
  pids, tids = {}, {}
  procs = helper_profile_filter_node(profile, name='process_name')
  for proc in procs: pids[proc['pid']] = proc['args']['name']
  threads = helper_profile_filter_node(profile, name='thread_name')
  for th in threads: tids[th['tid']] = th['args']['name']
  return pids, tids

def helper_profile_parse_deps(profile):
  deps = []
  for s in helper_profile_filter_node(profile, ph="s"):
    f = helper_profile_filter_node(profile, ph="f", id=s['id'])[0]

    starts, ends = [], []
    for x in helper_profile_filter_node(profile, ph="X"):
      if s['pid'] == x['pid'] and s['tid'] == x['tid'] and x['ts'] <= s['ts'] <= x['ts'] + x['dur']: starts.append(x)
      if f['pid'] == x['pid'] and f['tid'] == x['tid'] and x['ts'] <= f['ts'] <= x['ts'] + x['dur']: ends.append(x)

    assert len(starts) == 1 and len(ends) == 1, "more than one start and end possible, valid?"
    deps.append((s, f, starts[0], ends[0]))
  return deps

def helper_profile_filter_device(profile, device:str):
  assert any(getattr(x, "device", None) == device and isinstance(x, ProfileDeviceEvent) for x in profile), f"device {device} is not registred"
  dev_events = [x for x in profile if getattr(x, "device", None) == device and isinstance(x, ProfileDeviceEvent)]
  assert len(dev_events) == 1, "only one device registration event is expected"
  return [x for x in profile if getattr(x, "device", None) == device], dev_events[0]

def helper_validate_node(node, duration_s=10, ts_age_s=30, profile=None, pid_name=None, tid_name=None):
  pids, tids = helper_profile_parse_pids(profile)
  assert abs(node['ts'] - time.perf_counter_ns() / 1e3) < ts_age_s * 1e6, "timestimp is not in 30s range"
  assert 0 < node['dur'] < duration_s * 1e6, "duration is not in 10s range"
  assert pid_name is None or pids[node['pid']] == pid_name
  assert tid_name is None or tids[node['tid']] == tid_name

@unittest.skipUnless(issubclass(type(Device[Device.DEFAULT]), HCQCompiled), "HCQ device required to run")
class TestProfiler(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    TestProfiler.d0 = Device[Device.DEFAULT]

    TestProfiler.a = Tensor([0.,1.], device=Device.DEFAULT).realize()
    TestProfiler.b = self.a + 1
    si = create_schedule([self.b.lazydata])[-1]

    TestProfiler.runner = get_runner(TestProfiler.d0.device, si.ast)
    TestProfiler.b.lazydata.buffer.allocate()

    TestProfiler.kernargs_ba_ptr = TestProfiler.runner._prg.fill_kernargs([TestProfiler.b.lazydata.buffer._buf, TestProfiler.a.lazydata.buffer._buf])
    TestProfiler.kernargs_ab_ptr = TestProfiler.runner._prg.fill_kernargs([TestProfiler.a.lazydata.buffer._buf, TestProfiler.b.lazydata.buffer._buf])

  def test_profile_kernel_run(self):
    runner_name = TestProfiler.runner._prg.name
    with helper_collect_profile(TestProfiler.d0) as profile:
      TestProfiler.runner([TestProfiler.b.lazydata.buffer, TestProfiler.a.lazydata.buffer], var_vals={})

    profile, _ = helper_profile_filter_device(profile, TestProfiler.d0.device)
    kernel_runs = [x for x in profile if isinstance(x, ProfileRangeEvent)]
    assert len(kernel_runs) == 1, "one kernel run is expected"
    assert kernel_runs[0].name == runner_name, "kernel name is not correct"
    assert not kernel_runs[0].is_copy, "kernel should not be copy"

  def test_profile_copyin(self):
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))

    profile, _ = helper_profile_filter_device(profile, TestProfiler.d0.device)
    kernel_runs = [x for x in profile if isinstance(x, ProfileRangeEvent)]
    assert len(kernel_runs) == 1, "one kernel run is expected"
    assert kernel_runs[0].is_copy, "kernel should not be copy"

  def test_profile_multiops(self):
    runner_name = TestProfiler.runner._prg.name
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
      TestProfiler.runner([buf1, TestProfiler.a.lazydata.buffer], var_vals={})
      buf1.as_buffer()

    profile, _ = helper_profile_filter_device(profile, TestProfiler.d0.device)
    evs = [x for x in profile if isinstance(x, ProfileRangeEvent)]

    assert len(evs) == 3, "two kernel runs are expected"
    assert evs[0].is_copy, "kernel should be copy"
    assert evs[1].name == runner_name, "kernel name is not correct"
    assert not evs[1].is_copy, "kernel should not be copy"
    assert evs[2].is_copy, "kernel should be copy"

    for i in range(1, 3):
      assert evs[i].st > evs[i-1].en, "timestamp not aranged"

  def test_profile_multidev(self):
    d1 = Device[f"{Device.DEFAULT}:1"]
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()
    buf2 = Buffer(f"{Device.DEFAULT}:1", 2, dtypes.float, options=BufferSpec(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
      buf2.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))

    profile0, _ = helper_profile_filter_device(profile, TestProfiler.d0.device)
    profile1, _ = helper_profile_filter_device(profile, d1.device)

    for p in [profile0, profile1]:
      evs = [x for x in p if isinstance(x, ProfileRangeEvent)]
      assert len(evs) == 1, "one kernel runs are expected"
      assert evs[0].is_copy, "kernel should be copy"

  @unittest.skipIf(MOCKGPU and Device.DEFAULT == "AMD", "AMD mockgpu with indirect buffers does not support queue wait interrupts")
  def test_profile_graph(self):
    d1 = Device[f"{Device.DEFAULT}:1"]

    def f(a):
      x = (a + 1).realize()
      return x, x.to(d1.device).realize()

    a = Tensor.randn(10, 10, device=TestProfiler.d0.device).realize()
    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      jf = TinyJit(f)
      for _ in range(3): jf(a)
      del jf

    graph_evs = [x for x in profile if isinstance(x, ProfileGraphEvent)]

    _, _ = helper_profile_filter_device(profile, TestProfiler.d0.device)
    _, _ = helper_profile_filter_device(profile, d1.device)

    assert len(graph_evs) == 1, "one graph event is expected"
    assert len(graph_evs[0].ents) == 2, "two entities are expected"

  @unittest.skipIf(CI, "skip CI")
  def test_dev_jitter_matrix(self):
    for dev in HCQCompiled.devices: dev.synchronize()
    for dev in HCQCompiled.devices: dev._at_profile_finalize()

    def _sync_d2d(d1:HCQCompiled, d2:HCQCompiled):
      d1.hw_compute_queue_t().signal(d1.timeline_signal, d1.timeline_value).wait(d2.timeline_signal, d2.timeline_value) \
                             .timestamp(d1.timeline_signal).signal(d1.timeline_signal, d1.timeline_value+1).submit(d1)
      d2.hw_compute_queue_t().signal(d2.timeline_signal, d2.timeline_value).wait(d1.timeline_signal, d1.timeline_value) \
                             .timestamp(d2.timeline_signal).signal(d2.timeline_signal, d2.timeline_value+1).submit(d2)
      d1.timeline_value += 2
      d2.timeline_value += 2
      d1.timeline_signal.wait(d1.timeline_value - 1)
      d2.timeline_signal.wait(d2.timeline_value - 1)
      return d2.timeline_signal.timestamp - d1.timeline_signal.timestamp

    # then test it by timing the GPU to GPU times
    jitter_matrix = [[float('nan')] * len(HCQCompiled.devices) for _ in range(len(HCQCompiled.devices))]
    pairs = [(p1, p2) for p1 in enumerate(HCQCompiled.devices) for p2 in enumerate(HCQCompiled.devices) if p1 != p2]
    for (i1, d1), (i2, d2) in pairs:
      cpu_diff = d1.gpu2cpu_compute_time_diff - d2.gpu2cpu_compute_time_diff
      jitter_matrix[i1][i2] = statistics.median(_sync_d2d(d1, d2) - _sync_d2d(d2, d1) for _ in range(20)) / 2 - cpu_diff
      assert jitter_matrix[i1][i2] < 0.1, "jitter should be less than 0.1ms"

if __name__ == "__main__":
  unittest.main()