import math, os, subprocess, sys, unittest

def workload():
  from tinygrad import Tensor, Device
  from test.mockgpu.qcom.test_integration import close_device
  device = Device[Device.DEFAULT]
  try:
    assert (Tensor([2.,6.])/Tensor([4.,3.])).tolist() == [0.5,2.]
    assert Tensor([1.,4.,9.]).sqrt().tolist() == [1.,2.,3.]
    values = [-2.,0.,1.,2.]
    output = Tensor(values).exp().tolist()
    assert isinstance(output, list)
    for got, expected in zip(output, [math.exp(v) for v in values]):
      assert math.isclose(got, expected, rel_tol=2e-6), (got, expected)
    values = [1.,2.,3.]
    weights = [math.exp(v-max(values)) for v in values]
    output = Tensor(values).softmax().tolist()
    assert isinstance(output, list)
    for got, expected in zip(output, [v/sum(weights) for v in weights]):
      assert math.isclose(got, expected, rel_tol=2e-6), (got, expected)
    assert math.isnan(Tensor([-1.]).sqrt().item())
    assert Tensor([math.inf,-math.inf]).exp().tolist() == [math.inf,0.]
  finally: close_device(device)

def backend():
  import numpy as np, pytest
  from tinygrad import Device
  from test.mockgpu.qcom.test_integration import close_device
  np.random.seed(0)
  tests = [f'test/backend/test_ops.py::TestOps::test_{name}' for name in ('div', 'sqrt', 'exp', 'softmax', 'matmul_simple')]
  try: return pytest.main([*tests, '-q', '--tb=short'])
  finally: close_device(Device[Device.DEFAULT])

class TestQCOMMath(unittest.TestCase):
  def test_math(self):
    for runtime in ('PYTHON', 'CPU'):
      with self.subTest(runtime=runtime):
        result = subprocess.run([sys.executable, __file__, '--workload'], text=True, capture_output=True, timeout=60,
                                env=os.environ | {'DEV': 'MOCK+QCOM:IR3', 'HCQ_RUNTIME_DEV': runtime,
                                                  'PYTHONPATH': '.', 'BEAM': '0', 'NOOPT': '0'})
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(result.stderr, '')

if __name__ == '__main__':
  if '--backend' in sys.argv: sys.exit(backend())
  elif '--workload' in sys.argv: workload()
  else: unittest.main()
