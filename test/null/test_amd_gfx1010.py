import unittest
import tinygrad.runtime.autogen.am as am
from tinygrad.runtime.support.amd import import_soc


class TestGfx1010(unittest.TestCase):
  def test_gfx1010_target_tuple(self):
    trgt = 100100
    target = (trgt // 10000, (trgt // 100) % 100, trgt % 100)
    self.assertEqual(target, (10, 1, 0))
    self.assertEqual("gfx%d%x%x" % target, "gfx1010")
    self.assertTrue((target in ((9, 4, 2), (9, 5, 0))) or target[0] in (10, 11, 12))

  def test_import_soc_rdna1(self):
    self.assertIs(import_soc((10, 1, 0)), am.soc_11)