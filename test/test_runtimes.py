import unittest
from tinygrad.ops import Device
from tinygrad.runtime.ops_hip import HIPProgram

class TestHIP(unittest.TestCase):
  def setUp(self):
    if Device.DEFAULT != "HIP": self.skipTest("TestHIP is only for HIP")

  def test_compile_fail(self):
    try:
      prg = HIPProgram("test", "aaaa;", binary=False)
      # should cleanly destruct even if compile fails
      del prg
    except RuntimeError as e:
      assert str(e).startswith('HIP error')

if __name__ == '__main__':
  unittest.main(verbosity=2)
