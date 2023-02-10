#!/usr/bin/env python
import unittest
from extra.utils import fetch

class TestUtils(unittest.TestCase):  
  def test_fetch_bad_http(self):
    self.assertRaises(AssertionError, fetch, 'http://httpstat.us/500')
    self.assertRaises(AssertionError, fetch, 'http://httpstat.us/404')
    self.assertRaises(AssertionError, fetch, 'http://httpstat.us/400')
  
  def test_fetch_small(self):
    assert(len(fetch('https://google.com'))>0)

if __name__ == '__main__':
  unittest.main()