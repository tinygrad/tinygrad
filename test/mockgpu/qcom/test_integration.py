"""These process-level checks require the real MOCK+QCOM HCQ2 submission path."""
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from tinygrad.helpers import DEV


class TestQCOMIntegration(unittest.TestCase):
  def run_qcom(self, code, renderer=None):
    renderer = renderer or DEV.target('QCOM').renderer or 'IR3'
    with tempfile.TemporaryDirectory() as cache:
      result = subprocess.run([sys.executable, '-c', code], cwd=pathlib.Path(__file__).parents[3],
                              env={**os.environ, 'DEV':f'MOCK+QCOM:{renderer}', 'CACHELEVEL':'0', 'PARALLEL':'0', 'XDG_CACHE_HOME':cache},
                              capture_output=True, text=True, timeout=60)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertNotIn('Exception ignored', result.stderr)
    self.assertNotIn('Traceback', result.stderr)
    return result.stdout

  def test_real_submissions_cover_odd_shapes_and_repeated_programs(self):
    self.run_qcom('''import numpy as np
from tinygrad import Device, Tensor, dtypes
Device["QCOM"]
from test.mockgpu.mockgpu import drivers
driver = drivers[0]
for size in (1,3,17,65,257):
  before = sum(ctx.queued for ctx in driver.contexts.values())
  left = np.arange(size,dtype=np.int32) * 19 - 77
  right = np.arange(size,dtype=np.int32) * -7 + 5
  result = (Tensor(left,dtype=dtypes.int32) + Tensor(right,dtype=dtypes.int32)).numpy()
  np.testing.assert_array_equal(result,left+right)
  assert sum(ctx.queued for ctx in driver.contexts.values()) > before
  assert all(ctx.retired == ctx.queued and not ctx.pending for ctx in driver.contexts.values())
assert driver.error is None
''')

  def test_emitted_instruction_bytes_are_required_for_execution(self):
    output = self.run_qcom('''from tinygrad import Tensor, dtypes
from tinygrad.runtime.support.compiler_mesa import IR3Compiler
original = IR3Compiler.compile
def unsupported_instruction(self, source):
  library = original(self, source)
  image = self.unpack_lib(library)[3]
  return library[:-len(image)] + bytes.fromhex("ffffffffffffffff") + image[8:]
IR3Compiler.compile = unsupported_instruction
try:
  (Tensor([2,5,9],dtype=dtypes.int32) + Tensor([7,4,3],dtype=dtypes.int32)).tolist()
except RuntimeError as error:
  assert "IR3" in str(error), str(error)
  from test.mockgpu.mockgpu import drivers
  assert drivers[0].error is not None
  print("compiled instruction mutation rejected")
else:
  raise AssertionError("modified machine code unexpectedly executed")
''', renderer='IR3')
    self.assertIn('compiled instruction mutation rejected', output)

  def test_mapping_churn_does_not_leave_trackers_or_guest_authority(self):
    self.run_qcom('''from tinygrad import Device
device = Device["QCOM"]
from test.mockgpu.mockgpu import drivers
driver = drivers[0]
before = len(driver.memory.mappings),len(driver.allocations),len(driver.tracked_addresses)
for attempt in range(100):
  buffer = device._gpu_alloc(17)
  address = buffer.va_addr
  device._gpu_free(buffer)
  try:
    driver.memory.transaction().read(address,1)
  except ValueError:
    pass
  else:
    raise AssertionError("freed guest address remained readable")
assert (len(driver.memory.mappings),len(driver.allocations),len(driver.tracked_addresses)) == before
''')


if __name__ == '__main__':
  unittest.main()
