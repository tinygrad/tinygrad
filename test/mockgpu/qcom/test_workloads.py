import hashlib, json, os, subprocess, sys, unittest
from typing import Any, cast

def digest(value) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def workload(name):
  from tinygrad import Tensor, Device, TinyJit, dtypes
  from test.mockgpu.qcom.test_integration import close_device

  device = cast(Any, Device[Device.DEFAULT])
  import test.mockgpu.mockgpu as mockgpu
  driver = mockgpu.tracked_fds[device.fd.fd].driver
  dispatches: list[dict[str, object]] = []
  driver.instrumentation = dispatches.append
  outputs = []
  try:
    if name.startswith('matmul'):
      rows, inner, cols = (int(x) for x in name.split('_')[1:])
      a = [[float((i*3+k)%7-3) for k in range(inner)] for i in range(rows)]
      b = [[float((k*5+j)%9-4) for j in range(cols)] for k in range(inner)]
      expected = [[sum(a[i][k]*b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]
      x, y = Tensor(a, dtype=dtypes.float32), Tensor(b, dtype=dtypes.float32)
      output = (x @ y).tolist()
      if output != expected: raise AssertionError(f'{name}: output mismatch')
      if x.tolist() != a or y.tolist() != b: raise AssertionError(f'{name}: input mutated')
      outputs.append(output)
      del x, y
    else:
      def chain(x): return ((x*2+1).relu().sum(axis=1)+3).realize()
      jit = TinyJit(chain)
      run = jit if name == 'jit' else chain
      for step in range(5 if name == 'jit' else 1):
        values = [[float((i*3+j+step)%11-5) for j in range(33)] for i in range(17)]
        expected_rows: list[float] = [float(sum(max(v*2+1, 0) for v in row)+3) for row in values]
        x = Tensor(values, dtype=dtypes.float32).realize()
        output = run(x).tolist()
        if output != expected_rows: raise AssertionError(f'{name}: output mismatch at step {step}')
        if x.tolist() != values: raise AssertionError(f'{name}: input mutated at step {step}')
        outputs.append(output)
        del x
      if name == 'jit' and (jit.captured is None or not jit.captured.linear.src): raise AssertionError('jit capture missing')
      del jit, run
  finally:
    if 'x' in locals(): del x
    if 'y' in locals(): del y
    if 'jit' in locals(): del jit
    if 'run' in locals(): del run
    teardown = close_device(device)
    print(json.dumps({'case': name, 'runtime': os.environ.get('HCQ_RUNTIME_DEV', 'unknown'),
                      'dispatch_count': len(dispatches), 'dispatches': dispatches,
                      'output_digest': digest(outputs), 'teardown': teardown}, sort_keys=True), flush=True)

class TestQCOMWorkloads(unittest.TestCase):
  def test_workloads(self):
    for runtime in ('PYTHON', 'CPU'):
      for case in ('matmul_3_5_7', 'matmul_16_16_16', 'matmul_17_33_9', 'chain', 'jit'):
        with self.subTest(runtime=runtime, case=case):
          out = subprocess.run([sys.executable, __file__, case], text=True, capture_output=True, timeout=90,
                               env=os.environ | {'DEV': 'MOCK+QCOM:IR3', 'HCQ_RUNTIME_DEV': runtime,
                                                 'BEAM': '0', 'NOOPT': '0', 'PYTHONPATH': '.'})
          self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
          self.assertEqual(out.stderr, '')
          lines = out.stdout.splitlines()
          self.assertEqual(len(lines), 1, out.stdout)
          result = json.loads(lines[0])
          self.assertEqual((result['case'], result['runtime']), (case, runtime))
          self.assertEqual(result['dispatch_count'], len(result['dispatches']))
          self.assertGreater(result['dispatch_count'], 0)
          self.assertRegex(result['output_digest'], r'^[0-9a-f]{64}$')
          self.assertEqual(result['teardown'], {'objects': 0, 'maps': 0, 'contexts': 0,
                                                'timestamps': 0, 'constraints': 0, 'fds': 0})
          for dispatch in result['dispatches']:
            self.assertEqual(len(dispatch['groups']), 3)
            self.assertEqual(len(dispatch['local']), 3)
            self.assertEqual(len(dispatch['global']), 3)
            self.assertRegex(dispatch['program_sha256'], r'^[0-9a-f]{64}$')

if __name__ == '__main__':
  if len(sys.argv) > 1: workload(sys.argv[1])
  else: unittest.main()
