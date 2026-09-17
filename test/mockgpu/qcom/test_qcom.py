import os, subprocess, sys, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.helpers import Context, DEV, IMAGE

@unittest.skipUnless(DEV.device == 'QCOM' and DEV.interface == 'MOCK', 'requires the QCOM MockGPU backend')
class TestQCOMBackend(unittest.TestCase):
  @classmethod
  def setUpClass(cls): Device[Device.DEFAULT]
  def test_elementwise_through_command_queue(self):
    from test.mockgpu.mockgpu import drivers
    driver = next(driver for driver in drivers if type(driver).__name__ == 'QCOMDriver')
    before = driver.gpu.dispatches
    for size in (1, 4, 8, 64, 4096):
      with self.subTest(size=size):
        values = [float(i)-2.0 for i in range(size)]
        self.assertEqual((Tensor(values)+1.0).tolist(), [value+1.0 for value in values])
    self.assertGreater(driver.gpu.dispatches, before)

  def test_callback_error_reaches_tensor_caller(self):
    # A failed virtual device is kept in a separate process, just like a failed real-device session.
    program = '''
from tinygrad import Tensor
from unittest.mock import patch
assert (Tensor([1.0])+1).tolist() == [2.0]
from test.mockgpu.mockgpu import drivers
driver = next(driver for driver in drivers if type(driver).__name__ == 'QCOMDriver')
try:
  with patch.object(driver.gpu, 'execute', side_effect=ValueError('A630 execution test failure')):
    (Tensor([2.0])+3).tolist()
except RuntimeError as error:
  chain = []
  while error is not None:
    chain.append(error)
    error = error.__cause__ or error.__context__
  assert any(isinstance(item, ValueError) and str(item) == 'A630 execution test failure' for item in chain), chain
  assert not any('signal wait timed out' in str(item) for item in chain), chain
else:
  raise AssertionError('The device failure was not reported')
'''
    result = subprocess.run([sys.executable, '-c', program], env=os.environ.copy(), capture_output=True, text=True, timeout=15)
    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
    self.assertNotIn('Exception ignored on calling ctypes callback', result.stderr)
    self.assertNotIn('Exception ignored in atexit callback', result.stderr)

  def test_half_precision_arithmetic(self):
    values = [1.0, -2.0, 0.5, 3.0]
    a = Tensor(values, dtype=dtypes.float16)
    b = Tensor([1.0]*4, dtype=dtypes.float16)
    self.assertEqual((a+b).tolist(), [2.0, -1.0, 1.5, 4.0])
    self.assertEqual((a*b).tolist(), values)
    self.assertEqual((-a).tolist(), [-value for value in values])

  def test_half_precision_constants(self):
    values = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float16)
    for factor in (0.3333, 0.7, -1.234):
      with self.subTest(factor=factor):
        result = (Tensor(values, dtype=dtypes.half)*Tensor(factor, dtype=dtypes.half)).numpy()
        np.testing.assert_array_equal(result, values*np.float16(factor))

  def test_native_ioctl_fallback_preserves_pointer(self):
    import ctypes, termios
    from test.mockgpu.mockgpu import mock_ioctl
    reader, writer = os.pipe()
    try:
      os.write(writer, b'abc')
      available = ctypes.c_int()
      self.assertEqual(mock_ioctl(reader, termios.FIONREAD, ctypes.addressof(available)), 0)
      self.assertEqual(available.value, 3)
    finally:
      os.close(reader)
      os.close(writer)

  def test_padded_pooling_keeps_both_neighbors(self):
    values = Tensor([float(i) for i in range(1, 9)]).reshape(1, 1, 1, 8)
    result = values.avg_pool2d(kernel_size=(1, 2), padding=(0, 1), stride=(1, 1)).flatten().tolist()
    self.assertEqual(result, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 4.0])

  def test_saturating_clip(self):
    values = [-2.0, -0.5, 0.0, 0.25, 0.75, 1.0, 1.5, 4.0]
    self.assertEqual(Tensor(values).clip(0, 1).tolist(), [0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0])

  def test_boolean_load_and_conversion(self):
    self.assertEqual(Tensor([True, False]).float().tolist(), [1.0, 0.0])
    self.assertEqual((Tensor([0.0, 1.0, 2.0]) == Tensor([2.0, 1.0, 0.0])).tolist(), [False, True, False])

  def test_external_cpu_buffer(self):
    import ctypes
    values = (ctypes.c_float*4)(1.0, 2.0, 3.0, 4.0)
    tensor = Tensor.from_blob(ctypes.addressof(values), (4,), dtype=dtypes.float)
    self.assertEqual((tensor+1).tolist(), [2.0, 3.0, 4.0, 5.0])
    self.assertEqual(list(values), [1.0, 2.0, 3.0, 4.0])

  def test_default_host_runtime(self):
    environment = os.environ.copy()
    environment.pop('HCQ_RUNTIME_DEV', None)
    # Verify the fresh-process default directly. A tensor execution here recompiles CL,
    # making this unit test depend on compiler-server latency.
    result = subprocess.run([sys.executable, '-c', 'from tinygrad.device import HCQ_RUNTIME_DEV; assert HCQ_RUNTIME_DEV.value == "CPU"'],
                            env=environment, capture_output=True, text=True, timeout=15)
    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

  @unittest.skipUnless(IMAGE, 'requires IMAGE=1 before QCOM initialization')
  def test_image_matmul(self):
    from test.mockgpu.qcom.emu import Image
    original_load = Image.load
    for dtype in (dtypes.float32, dtypes.float16):
      with self.subTest(dtype=dtype), Context(FLOAT16=1):
        formats = set()
        def observe_load(image, *args):
          formats.add(image.half)
          return original_load(image, *args)
        a = (np.arange(16*16, dtype=np.float32).reshape(16,16) % 9)-4
        b = (np.arange(16*16, dtype=np.float32).reshape(16,16) % 7)-3
        with patch.object(Image, 'load', observe_load):
          result = (Tensor(a, dtype=dtype) @ Tensor(b, dtype=dtype)).numpy()
        np.testing.assert_array_equal(result, a @ b)
        self.assertIn(dtype == dtypes.float16, formats)

  @unittest.skipUnless(IMAGE, 'requires IMAGE=1 before QCOM initialization')
  def test_image_padded_convolution(self):
    from test.mockgpu.qcom.emu import Image
    original_load = Image.load
    outside = []
    def observe_load(image, memory, x, y):
      if not (0 <= x < image.width and 0 <= y < image.height): outside.append((x,y))
      return original_load(image, memory, x, y)
    values = np.arange(1, 65, dtype=np.float32).reshape(1,1,8,8)
    padded = np.pad(values[0,0], 1)
    expected = np.array([[padded[y:y+3,x:x+3].sum() for x in range(8)] for y in range(8)], dtype=np.float32)
    with Context(IMAGE=1), patch.object(Image, 'load', observe_load):
      result = Tensor(values).conv2d(Tensor.ones(1,1,3,3), padding=1).numpy()
    np.testing.assert_array_equal(result[0,0], expected)
    self.assertTrue(outside, 'the compiled kernel must exercise texture border handling')

if __name__ == '__main__': unittest.main()
