import os, unittest
from tinygrad.runtime.ops_amd import parse_xccs
from tinygrad.helpers import getenv

class TestParseXCCS(unittest.TestCase):
  def test_default(self):
    os.environ.pop('XCCS', None)
    getenv.cache_clear()
    self.assertEqual(parse_xccs({'num_xcc':8}), 8)

  def test_override(self):
    os.environ['XCCS'] = '4'
    getenv.cache_clear()
    self.assertEqual(parse_xccs({'num_xcc':8}), 4)
    os.environ.pop('XCCS')

  def test_cap_and_invalid(self):
    os.environ['XCCS'] = '12'
    getenv.cache_clear()
    self.assertEqual(parse_xccs({'num_xcc':8}), 8)
    os.environ['XCCS'] = 'abc'
    getenv.cache_clear()
    self.assertEqual(parse_xccs({'num_xcc':8}), 8)
    os.environ['XCCS'] = '-1'
    getenv.cache_clear()
    self.assertEqual(parse_xccs({'num_xcc':8}), 1)
    os.environ.pop('XCCS')

if __name__ == '__main__':
  unittest.main()

