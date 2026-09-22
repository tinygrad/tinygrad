import os, subprocess, sys, textwrap, unittest

def close_device(d):
  from tinygrad import Device
  from tinygrad.device import Buffer
  from tinygrad.helpers import Context
  from tinygrad.runtime.autogen import kgsl
  import test.mockgpu.mockgpu as mockgpu
  driver = mockgpu.tracked_fds[d.fd.fd].driver
  import gc
  gc.collect()
  d.finalize()
  # Device-owned buffers only; do not search user tensors.
  for name in ('_stack', 'dummy', 'border_color'):
    buf = d.__dict__.get(name)
    if isinstance(buf, Buffer) and buf.is_allocated(): buf.deallocate()
  d.allocator.free_cache()
  # finalize waits for pending work; release Buffer owners first.
  with Context(LRU=0): d.timeline.deallocate()
  # Collect unreachable cycles only; do not force-free buffers.
  gc.collect()
  from tinygrad.tensor import all_tensors
  live = [t for ref in list(all_tensors) if (t := ref()) is not None and t.device == d.device]
  leak = f"objects={driver.objects!r} maps={driver.maps!r} tensors={len(live)} shapes={[t.shape for t in live]}"
  assert not driver.objects and not driver.maps, f"strict teardown leaked {leak}"
  kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(d.fd, drawctxt_id=d.ctx)
  assert not driver.contexts and not driver.gpus and not driver.timestamps and not driver.constraints
  fd = d.fd.fd
  del d.fd
  gc.collect()
  assert fd not in mockgpu.tracked_fds and not driver.fds
  Device._opened_devices.remove(d.device)
  return {'objects': len(driver.objects), 'maps': len(driver.maps), 'contexts': len(driver.contexts),
          'timestamps': len(driver.timestamps), 'constraints': len(driver.constraints), 'fds': len(driver.fds)}

def force_cleanup_for_diagnostics(d):
  import gc
  from tinygrad import Device
  from tinygrad.device import Buffer
  from tinygrad.helpers import Context
  from tinygrad.runtime.autogen import kgsl
  import test.mockgpu.mockgpu as mockgpu
  driver = mockgpu.tracked_fds[d.fd.fd].driver
  d.finalize()
  gc.collect()
  buffers = [obj for obj in gc.get_objects()
             if issubclass(type(obj), Buffer) and obj.device == d.device and obj.is_allocated() and obj is not d.timeline]
  for buf in sorted(buffers, key=lambda buf: buf._base is None): buf.deallocate()
  d.allocator.free_cache()
  # Free synchronizes, so the timeline is the last allocation released.
  with Context(LRU=0): d.timeline.deallocate()
  assert not driver.objects and not driver.maps
  kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(d.fd, drawctxt_id=d.ctx)
  assert not driver.contexts and not driver.gpus and not driver.timestamps and not driver.constraints
  fd = d.fd.fd
  del d.fd
  gc.collect()
  assert fd not in mockgpu.tracked_fds and not driver.fds
  Device._opened_devices.remove(d.device)

class TestQCOMIntegration(unittest.TestCase):
  def test_review_workloads(self):
    code = '''
      from tinygrad import Tensor, dtypes, TinyJit, Device
      for n in (2,3,4,5,127,129,513,1024):
        left = [(i*0x1020304) & 0xffffffff for i in range(n)]
        right = [(0xffffffff-i) for i in range(n)]
        a, b = Tensor(left, dtype=dtypes.uint32), Tensor(right, dtype=dtypes.uint32)
        assert (a+b).tolist() == [(x+y) & 0xffffffff for x,y in zip(left,right)]
      left, right = [0xffffffff,0x12345678,0x80000000], [0xffffffff,0x10001,3]
      a, b = Tensor(left, dtype=dtypes.uint32), Tensor(right, dtype=dtypes.uint32)
      assert (a*b).tolist() == [1,0x68ac5678,0x80000000]
      a, b = Tensor([1.0,-2.0]), Tensor([3.0,4.0])
      assert (a+b).tolist() == [4.0,2.0]
      for n in (128,256):
        values = [(i*0x1020304) & 0xffffffff for i in range(n)]
        assert Tensor(values,dtype=dtypes.uint32).sum().item() == sum(values) & 0xffffffff
      @TinyJit
      def add(a,b): return (a+b).realize()
      for i in range(3):
        a, b = Tensor([i]*128,dtype=dtypes.uint32), Tensor([1]*128,dtype=dtypes.uint32)
        assert add(a,b).tolist() == [i+1]*128
      del a, b
      del add
      from test.mockgpu.qcom.test_integration import close_device
      close_device(Device[Device.DEFAULT])
    '''
    for runtime in ('PYTHON', 'CPU'):
      with self.subTest(runtime=runtime):
        out = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=os.getcwd(),
                             env=os.environ | {'DEV': 'MOCK+QCOM:IR3', 'BEAM': '0', 'NOOPT': '0', 'PYTHONPATH': '.',
                                               'HCQ_RUNTIME_DEV': runtime}, capture_output=True, text=True, timeout=20)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(out.stderr, '')
        self.assertNotIn('failed', out.stdout)

  def test_failed_launch_reports_cause_and_recovers(self):
    code = '''
      from tinygrad import Device, Tensor, dtypes, TinyJit
      from unittest.mock import patch
      d = Device[Device.DEFAULT]
      import test.mockgpu.mockgpu as mockgpu
      driver = mockgpu.tracked_fds[d.fd.fd].driver
      a, b = Tensor([1,2], dtype=dtypes.uint32), Tensor([3,4], dtype=dtypes.uint32)
      timeline = d.timeline.host.view(fmt='Q')
      before = tuple(timeline), driver.timestamps[d.ctx]
      from test.mockgpu.qcom.errors import ErrorCode, ModelExecutionError
      def reject_initial(*_, **__): raise ModelExecutionError('IR3 pc=0x10: rejected test instruction')
      with patch('test.mockgpu.qcom.compute.execute_group', side_effect=reject_initial):
        failed = a+b
        try: failed.realize()
        except OSError as error:
          assert 'IR3 pc=0x10: rejected test instruction' in str(error), str(error)
          assert error.model_code == ErrorCode.EXECUTION
          del error
        else: raise AssertionError('failed submission returned successfully')
        del failed
      assert (tuple(timeline), driver.timestamps[d.ctx]) == before
      assert a.tolist() == [1,2] and b.tolist() == [3,4]
      x, y = Tensor([5,6], dtype=dtypes.uint32), Tensor([7,8], dtype=dtypes.uint32)
      assert (x+y).tolist() == [12,14]
      d.synchronize()
      @TinyJit
      def add(a,b): return (a+b).realize()
      for _ in range(3): assert add(x,y).tolist() == [12,14]
      before = tuple(timeline), driver.timestamps[d.ctx]
      def reject_jit(*_, **__): raise ModelExecutionError('IR3 pc=0x20: rejected JIT instruction')
      with patch('test.mockgpu.qcom.compute.execute_group', side_effect=reject_jit):
        try: add(x,y)
        except OSError as error:
          assert 'IR3 pc=0x20: rejected JIT instruction' in str(error)
          assert error.model_code == ErrorCode.EXECUTION
          del error
        else: raise AssertionError('failed JIT submission returned successfully')
      assert (tuple(timeline), driver.timestamps[d.ctx]) == before
      assert add(x,y).tolist() == [12,14]
      before = tuple(timeline), driver.timestamps[d.ctx]
      unsupported = Tensor([2.0**-149,1.0])+Tensor([1.0,2.0])
      try: unsupported.realize()
      except OSError as error:
        assert 'IR3 pc=' in str(error) and 'non-normal float input unsupported' in str(error), str(error)
        assert error.model_code == ErrorCode.INPUT
        del error
      else: raise AssertionError('unsupported float returned successfully')
      del unsupported
      assert (tuple(timeline), driver.timestamps[d.ctx]) == before
      assert add(x,y).tolist() == [12,14]
      del a, b, x, y, add
      from test.mockgpu.qcom.test_integration import close_device
      close_device(d)
    '''
    for runtime in ('PYTHON', 'CPU'):
      with self.subTest(runtime=runtime):
        out = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=os.getcwd(),
                             env=os.environ | {'DEV': 'MOCK+QCOM:IR3', 'BEAM': '0', 'NOOPT': '0', 'PYTHONPATH': '.',
                                               'HCQ_RUNTIME_DEV': runtime}, capture_output=True, text=True, timeout=15)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(out.stderr, '')
        self.assertNotIn('failed', out.stdout)

  def test_production_add_and_teardown(self):
    code = """
      from tinygrad import Device, Tensor, dtypes
      from test.mockgpu.qcom.test_integration import close_device
      d = Device[Device.DEFAULT]
      import test.mockgpu.mockgpu as mockgpu
      driver = mockgpu.tracked_fds[d.fd.fd].driver
      for size in (1, 128, 256, 1024, 4096):
        for repeat in range(2):
          left = [((i * 0x1020304) ^ (0xffffffff if repeat else 0x80000000)) & 0xffffffff for i in range(size)]
          right = [i * 17 + repeat + 1 for i in range(size)]
          a, b = Tensor(left, dtype=dtypes.uint32), Tensor(right, dtype=dtypes.uint32)
          before = driver.timestamps[d.ctx]
          result = (a + b).numpy().tolist()
          assert result == [(x + y) % (1 << 32) for x, y in zip(left, right)], (size, repeat)
          assert a.numpy().tolist() == left and b.numpy().tolist() == right
          gpu = driver.gpus[d.ctx]
          assert gpu.shader and len(gpu.shader) % 8 == 0
          assert driver.timestamps[d.ctx] > before
          assert gpu.registers[0xb987] == 0x140
          assert gpu.registers[0xb999] == max(1, size // 512), (size, gpu.registers[0xb999])
          assert gpu.registers[0xb991] == (1 if size == 1 else size // 4), (size, gpu.registers[0xb991])
          del a, b
      assert driver.gpus[d.ctx].dispatches == 10
      close_device(d)
    """
    for runtime in ('PYTHON', 'CPU'):
      with self.subTest(runtime=runtime):
        out = subprocess.run([sys.executable, '-c', textwrap.dedent(code)], cwd=os.getcwd(),
                             env=os.environ | {'DEV': 'MOCK+QCOM:IR3', 'BEAM': '0', 'NOOPT': '0', 'PYTHONPATH': '.',
                                               'HCQ_RUNTIME_DEV': runtime}, capture_output=True, text=True, timeout=90)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertEqual(out.stderr, '')

if __name__ == '__main__': unittest.main()
