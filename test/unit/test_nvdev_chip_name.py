import unittest

class TestNVChipPrefix(unittest.TestCase):
  def test_known_architectures(self):
    from tinygrad.runtime.support.nv.nvdev import nv_chip_prefix
    self.assertEqual(nv_chip_prefix(0x17), "GA1")
    self.assertEqual(nv_chip_prefix(0x19), "AD1")
    self.assertEqual(nv_chip_prefix(0x1b), "GB2")

  def test_unmapped_architecture_raises_clear_error(self):
    from tinygrad.runtime.support.nv.nvdev import nv_chip_prefix
    with self.assertRaises(RuntimeError) as ctx:
      nv_chip_prefix(0x3f)
    self.assertIn("0x3f", str(ctx.exception))
    self.assertNotIsInstance(ctx.exception, KeyError)

if __name__ == "__main__":
  unittest.main()
