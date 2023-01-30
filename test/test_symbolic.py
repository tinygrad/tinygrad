#!/usr/bin/env python
import unittest
from tinygrad.shape.symbolic import Variable

def helper_test_variable(v, n, m, s):
  assert v.min == n
  assert v.max == m
  assert str(v) == s

class TestSymbolic(unittest.TestCase):
  def test_mul_0(self):
    helper_test_variable(Variable("a", 0, 8)*0, 0, 0, "0")

  def test_mul_1(self):
    helper_test_variable(Variable("a", 0, 8)*1, 0, 8, "a")

  def test_mul_2(self):
    helper_test_variable(Variable("a", 0, 8)*2, 0, 16, "(a*2)")

  def test_div_1(self):
    helper_test_variable(Variable("a", 0, 8)//1, 0, 8, "a")

  def test_mod_1(self):
    helper_test_variable(Variable("a", 0, 8)%1, 0, 0, "0")

  def test_add_min_max(self):
    helper_test_variable(Variable("a", 0, 8) * 2 + 12, 12, 16+12, "((a*2)+12)")

  def test_div_min_max(self):
    helper_test_variable(Variable("a", 0, 7) // 2, 0, 3, "(a//2)")

  def test_sum_div_min_max(self):
    helper_test_variable(Variable.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2, 0, 5, "((a+b)//2)")

  def test_sum_div_factor(self):
    helper_test_variable(Variable.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2, 0, 20, "((a*2)+(b*2))")

  def test_sum_div_some_factor(self):
    helper_test_variable(Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2, 0, 23, "((b*2)+((a*5)//2))")

  def test_sum_div_no_factor(self):
    helper_test_variable(Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2, 0, 25, "(((a*5)+(b*5))//2)")
  
  def test_mod_factor(self):
    helper_test_variable(Variable.sum([Variable("a", 0, 7)*100, Variable("b", 0, 3)*50]) % 100, 0, 50, "(((a*100)+(b*50))%100)")
  
  def test_sum_0(self):
    helper_test_variable(Variable.sum([Variable("a", 0, 7)]), 0, 7, "a")

  def test_mod_remove(self):
    helper_test_variable(Variable("a", 0, 6)%100, 0, 6, "a")

  def test_gt_remove(self):
    helper_test_variable(Variable("a", 0, 6) >= 25, 0, 0, "0")

  def test_lt_remove(self):
    helper_test_variable(Variable("a", 0, 6) < -3, 0, 0, "0")
    helper_test_variable(Variable("a", 0, 6) < 3, 0, 1, "(a<3)")
    helper_test_variable(Variable("a", 0, 6) < 8, 1, 1, "1")

  def test_and_fold(self):
    helper_test_variable(Variable.ands([Variable.num(0), Variable("a", 0, 1)]), 0, 0, "0")

  def test_and_remove(self):
    helper_test_variable(Variable.ands([Variable.num(1), Variable("a", 0, 1)]), 0, 1, "a")

  def test_mod_factor(self):
    # this is technically wrong, if b is 0 the output will be negative
    helper_test_variable(Variable.sum([Variable.num(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28, -1, 27, "((-1+a)%28)")

  def test_div_factor(self):
    # err, that *2 shouldn't be needed
    helper_test_variable(Variable.sum([Variable.num(-44), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) // 40, -1, 9, "(b+-1)")
  
  def test_mul_div(self):
    helper_test_variable((Variable("a", 0, 10)*4)//4, 0, 10, "a")

  def test_mul_div_factor_mul(self):
    helper_test_variable((Variable("a", 0, 10)*8)//4, 0, 20, "(a*2)")

  def test_mul_div_factor_div(self):
    helper_test_variable((Variable("a", 0, 10)*4)//8, 0, 5, "(a//2)")

  def test_div_remove(self):
    helper_test_variable(Variable.sum([Variable("idx0", 0, 127)*4, Variable("idx2", 0, 3)])//4, 0, 127, "idx0")

if __name__ == '__main__':
  unittest.main()

