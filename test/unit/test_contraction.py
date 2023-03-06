#!/usr/bin/env python
import unittest
from tinygrad.lazy import get_contraction

class TestFlopCounter(unittest.TestCase):
  def test_contraction(self):
    r = get_contraction((1,2,3,4), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3]])
    
    r = get_contraction((1,2,3,1,4), (1,2,3,4))
    self.assertEqual(r, [[0], [1], [2], [3, 4]])

    r = get_contraction((1,2,3,1,4,1,1), (2,3,4))
    self.assertEqual(r, [[0, 1], [2], [3, 4, 5, 6]])

    r = get_contraction((1,2,3,4), (1,2,3*4))
    self.assertEqual(r, [[0], [1], [2, 3]])

    r = get_contraction((1,2,3,4), (2,1,3,4))
    self.assertEqual(r, None)

    r = get_contraction((1,2,3,4), (1,2,3,4,1))
    self.assertEqual(r, None)

    r = get_contraction((1,2,3,4), (1,2,6,2))
    self.assertEqual(r, None)



if __name__ == '__main__':
  unittest.main()
