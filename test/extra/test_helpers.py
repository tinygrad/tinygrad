#!/usr/bin/env python
import multiprocessing
import unittest
from extra.helpers import cross_process

class TestCrossProcess(unittest.TestCase):
  def test_cross_process(self):
    def _iterate():
      for i in range(3): yield i
    
    ret = cross_process(lambda: _iterate())
    assert len(list(ret)) == 3

if __name__ == '__main__':
  unittest.main()