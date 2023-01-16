#!/usr/bin/env python
import unittest
from tinygrad.symbolic import Variable, SumNode

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

  def test_mod_1(self):
    ret = Variable("a", 0, 8)%1
    assert str(ret) == "0"

  def test_add_min_max(self):
    ret = Variable("a", 0, 8) * 2 + 12
    assert ret.min == 12
    assert ret.max == 16+12

  def test_div_min_max(self):
    ret = Variable("a", 0, 7) // 2
    assert ret.min == 0
    assert ret.max == 3

  def test_sum_div_min_max(self):
    ret = SumNode([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2
    assert ret.min == 0
    assert ret.max == 5

  def test_sum_div_factor(self):
    ret = SumNode([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2
    print(ret)


if __name__ == '__main__':
  unittest.main()

