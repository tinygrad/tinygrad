import unittest
from tinygrad.helpers import Target

class TestTargetParse(unittest.TestCase):
  # https://discord.com/channels/1068976834382925865/1068982781490757652/1485897360054681652
  def test_user_stories(self):
    for x in ["AMD", "AMD:AMDLLVM", "CPU:LLVM:x86_64,avx512", "NV:CUDA:sm_70", "AMD::gfx1100,fma_fold,-sin", "NULL:QCOMCL:a630",
              "MOCK+AMD::gfx950", "PYTHON::sm_89", "USB+AMD:AMDLLVM:gfx1201", "REMOTE:localhost:1337+AMD:AMDLLVM", "PCI:0-2,4+AMD"]:
      self.assertEqual(repr(Target.parse(x)), x)

  def test_plus_in_arch_params(self):
    with self.assertRaises(AssertionError): Target.parse("USB+AMD::gfx1100,+sin")

  def test_invalid_too_many_colons(self):
    with self.assertRaises(AssertionError): Target.parse("AMD:AMDLLVM:gfx1100:extra")

if __name__ == '__main__':
  unittest.main()
