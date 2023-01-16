#!/usr/bin/env python
import unittest
from tinygrad.symbolic import Variable

class TestSymbolic(unittest.TestCase):
  def test_mul_0(self):
    ret = Variable("a", 0, 8)*0
    assert str(ret) == "0"

  def test_mul_1(self):
    ret = Variable("a", 0, 8)*1
    assert str(ret) == "a"

  def test_mul_2(self):
    ret = Variable("a", 0, 8)*2
    assert str(ret) == "(a*2)"

  def test_div_1(self):
    ret = Variable("a", 0, 8)//1
    assert str(ret) == "a"

if __name__ == '__main__':
  unittest.main()

