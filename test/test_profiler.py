import unittest, struct, contextlib, tempfile, pathlib, json, time, atexit, random
from tinygrad import Device, Tensor, dtypes, TinyJit
from tinygrad.helpers import CI, getenv, Context
from tinygrad.device import Buffer, BufferOptions
from tinygrad.runtime.support.hcq import ProfileLogger, HCQCompiled
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import get_runner

MOCKGPU = getenv("MOCKGPU")

@contextlib.contextmanager
def helper_collect_profile(*devs, random_setup_delay=False):
  ProfileLogger.mjson, ProfileLogger.actors = [], {}

  if random_setup_delay:
    devs = list(devs)
    for dev in devs: dev.synchronize()
    random.shuffle(devs)
    for dev in devs:
      dev._prof_setup()
      time.sleep(random.randint(1, 1000) / 1000)
  else:
    for dev in devs: dev._prof_setup()

  profile_dict = {}
  _, tmp = tempfile.mkstemp()
  with Context(PROFILE=1, PROFILEPATH=tmp):
    try: yield profile_dict
    finally:
      for dev in devs:
        dev.synchronize()
        dev._prof_finalize()
        atexit.unregister(dev._prof_finalize)

  for k,v in json.loads(pathlib.Path(tmp).read_text()).items(): profile_dict[k] = v
  pathlib.Path(tmp).unlink()

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

    TestProfiler.runner = get_runner(TestProfiler.d0.dname, si.ast)
    TestProfiler.b.lazydata.buffer.allocate()

    TestProfiler.kernargs_ba_ptr = TestProfiler.runner.clprg.fill_kernargs([TestProfiler.b.lazydata.buffer._buf, TestProfiler.a.lazydata.buffer._buf])
    TestProfiler.kernargs_ab_ptr = TestProfiler.runner.clprg.fill_kernargs([TestProfiler.a.lazydata.buffer._buf, TestProfiler.b.lazydata.buffer._buf])

  def test_profile_kernel_run(self):
    runner_name = TestProfiler.runner.clprg.name
    with helper_collect_profile(TestProfiler.d0) as profile:
      TestProfiler.runner([TestProfiler.b.lazydata.buffer, TestProfiler.a.lazydata.buffer], var_vals={})

    kernel_node = helper_profile_filter_node(profile, name=runner_name)[0]
    helper_validate_node(kernel_node, profile=profile, pid_name=Device.DEFAULT, tid_name="COMPUTE")

  def test_profile_copyin(self):
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))

    copyin_node = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}")[0]
    helper_validate_node(copyin_node, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

  def test_profile_multiops(self):
    runner_name = TestProfiler.runner.clprg.name
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
      TestProfiler.runner([buf1, TestProfiler.a.lazydata.buffer], var_vals={})
      buf1.as_buffer()

    copyin_node = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}")[0]
    helper_validate_node(copyin_node, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

    kernel_node = helper_profile_filter_node(profile, name=runner_name)[0]
    helper_validate_node(kernel_node, profile=profile, pid_name=Device.DEFAULT, tid_name="COMPUTE")

    copyout_node = helper_profile_filter_node(profile, name=f"{Device.DEFAULT} -> CPU")[0]
    helper_validate_node(copyout_node, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

    assert copyin_node['ts'] + copyin_node['dur'] < kernel_node['ts'], "timestamp not aranged"
    assert kernel_node['ts'] + kernel_node['dur'] < copyout_node['ts'], "timestamp not aranged"

  def test_profile_multidev_copyin(self):
    d1 = Device[f"{Device.DEFAULT}:1"]
    buf1 = Buffer(Device.DEFAULT, 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()
    buf2 = Buffer(f"{Device.DEFAULT}:1", 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated()

    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      buf1.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))
      buf2.copyin(memoryview(bytearray(struct.pack("ff", 0, 1))))

    copyin_node_1 = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}")[0]
    helper_validate_node(copyin_node_1, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")

    copyin_node_2 = helper_profile_filter_node(profile, name=f"CPU -> {Device.DEFAULT}:1")[0]
    helper_validate_node(copyin_node_2, profile=profile, pid_name=f"{Device.DEFAULT}:1", tid_name="DMA")

  def test_profile_multidev_transfer(self):
    d1 = Device[f"{Device.DEFAULT}:1"]
    a = Tensor.randn(1 << 20, device=Device.DEFAULT).realize()
    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      y = a.to(f"{Device.DEFAULT}:1")
      y.realize()

    transfer_node_1 = helper_profile_filter_node(profile, name=f"{Device.DEFAULT} -> {Device.DEFAULT}:1")[0]
    helper_validate_node(transfer_node_1, profile=profile, pid_name=Device.DEFAULT, tid_name="DMA")
    assert 80 < transfer_node_1['dur'] < (20000 if CI else 1400), f"Duration is not in the range: {transfer_node_1['dur']}"

  @unittest.skipIf(MOCKGPU and Device.DEFAULT == "AMD", "AMD mockgpu with indirect buffers does not support queue wait interrupts")
  def test_profile_deps(self):
    d1 = Device[f"{Device.DEFAULT}:1"]

    def f(a):
      x = (a + 1).realize()
      return x, x.to(d1.dname).realize()

    a = Tensor.randn(10, 10, device=TestProfiler.d0.dname).realize()
    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      jf = TinyJit(f)
      for _ in range(3): jf(a)
      del jf

    deps = helper_profile_parse_deps(profile)
    assert len(deps) == 1, "one dep is expected, one launch"

    _, _, l, r = deps[0]
    assert l['name'].find("->") == -1, "should be kernel"
    assert r['name'] == f"{Device.DEFAULT} -> {Device.DEFAULT}:1", "should be copy"

  @unittest.skipIf(MOCKGPU and Device.DEFAULT == "AMD", "AMD mockgpu with indirect buffers does not support queue wait interrupts")
  def test_profile_copy_args(self):
    d1 = Device[f"{Device.DEFAULT}:1"]

    def f(a):
      x = (a + 1).realize()
      return x, x.to(d1.dname).realize()

    a = Tensor.randn(10, 10, device=TestProfiler.d0.dname).realize()
    with helper_collect_profile(TestProfiler.d0, d1) as profile:
      jf = TinyJit(f)
      for _ in range(3):
        TestProfiler.d0.raw_prof_records, TestProfiler.d0.sig_prof_records = [], [] # reset to collect only graph logs
        d1.raw_prof_records, d1.sig_prof_records = [], []
        jf(a)
      del jf

    node = helper_profile_filter_node(profile, name=f"{Device.DEFAULT} -> {Device.DEFAULT}:1")[-1]
    assert node['args']['Size'] == "400.00 B"
    assert abs(float(node['args']['GB/S']) - ((10 * 10 * 4) / 1e3) / (node['dur'])) < 0.01

  @unittest.skipIf(CI, "skip CI")
  def test_profile_sync(self):
    mv = memoryview(bytearray(struct.pack("ff", 0, 1)))
    expected_diff = 100000 # sleep in us

    devs = [Device[f"{Device.DEFAULT}:{i}"] for i in range(6)]
    bufs = [Buffer(f"{Device.DEFAULT}:{i}", 2, dtypes.float, options=BufferOptions(nolru=True)).ensure_allocated() for i in range(6)]

    # enqueue ops on different queues to check the timer sync
    cpu_time = []
    with helper_collect_profile(*devs, random_setup_delay=True) as profile:
      for i in range(6):
        x = time.perf_counter_ns()
        time.sleep(expected_diff / 1e6)
        bufs[i].copyin(mv)
        cpu_time.append(((time.perf_counter_ns() - x) / 1000) - expected_diff)

    nodes = [helper_profile_filter_node(profile, name=f"CPU -> {Device.canonicalize(f'{Device.DEFAULT}:{i}')}")[-1] for i in range(6)]
    avg_diff = []
    for i in range(1, 6):
      diff = nodes[i]['ts'] - nodes[i-1]['ts'] - cpu_time[i]
      avg_diff.append(diff - expected_diff)
      assert expected_diff * 0.998 < diff < expected_diff * 1.002, "more that 0.2% diff"

    print(f"total avg delay is {sum(avg_diff) / len(avg_diff)} us")

if __name__ == "__main__":
  unittest.main()
