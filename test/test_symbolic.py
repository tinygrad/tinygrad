#!/usr/bin/env python
import unittest
from tinygrad.shape.symbolic import Variable

class TestSymbolic(unittest.TestCase):
  def helper_test_variable(self, v, n, m, s):
    self.assertEqual(v.min, n)
    self.assertEqual(v.max, m)
    self.assertEqual(v.render(), s)

  def test_mul_0(self):
    self.helper_test_variable(Variable("a", 0, 8)*0, 0, 0, "0")

  def test_mul_1(self):
    self.helper_test_variable(Variable("a", 0, 8)*1, 0, 8, "a")

  def test_mul_2(self):
    self.helper_test_variable(Variable("a", 0, 8)*2, 0, 16, "(a*2)")

  def test_div_1(self):
    self.helper_test_variable(Variable("a", 0, 8)//1, 0, 8, "a")

  def test_mod_1(self):
    self.helper_test_variable(Variable("a", 0, 8)%1, 0, 0, "0")

  def test_add_min_max(self):
    self.helper_test_variable(Variable("a", 0, 8) * 2 + 12, 12, 16+12, "((a*2)+12)")

  def test_div_min_max(self):
    self.helper_test_variable(Variable("a", 0, 7) // 2, 0, 3, "(a//2)")

  def test_sum_div_min_max(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2, 0, 5, "((a+b)//2)")

  def test_sum_div_factor(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2, 0, 20, "((a*2)+(b*2))")

  def test_sum_div_some_factor(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2, 0, 23, "((b*2)+((a*5)//2))")

  def test_sum_div_no_factor(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2, 0, 25, "(((a*5)+(b*5))//2)")
  
  @unittest.skip("mod max is wrong")
  def test_mod_factor(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*100, Variable("b", 0, 3)*50]) % 100, 0, 50, "(((a*100)+(b*50))%100)")
  
  def test_sum_0(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)]), 0, 7, "a")

  def test_mod_remove(self):
    self.helper_test_variable(Variable("a", 0, 6)%100, 0, 6, "a")

  def test_big_mod(self):
    self.helper_test_variable(Variable("a", -20, 20)%10, -9, 9, "(a%10)")
    self.helper_test_variable(Variable("a", -20, 0)%10, -9, 0, "(a%10)")
    self.helper_test_variable(Variable("a", -20, 1)%10, -9, 1, "(a%10)")
    self.helper_test_variable(Variable("a", 0, 20)%10, 0, 9, "(a%10)")
    self.helper_test_variable(Variable("a", -1, 20)%10, -1, 9, "(a%10)")

  def test_gt_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) >= 25, 0, 0, "0")

  def test_lt_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) < -3, 0, 0, "0")
    self.helper_test_variable(Variable("a", 0, 6) < 3, 0, 1, "(a<3)")
    self.helper_test_variable(Variable("a", 0, 6) < 8, 1, 1, "1")

  def test_and_fold(self):
    self.helper_test_variable(Variable.ands([Variable.num(0), Variable("a", 0, 1)]), 0, 0, "0")

  def test_and_remove(self):
    self.helper_test_variable(Variable.ands([Variable.num(1), Variable("a", 0, 1)]), 0, 1, "a")

  def test_mod_factor_negative(self):
    # this is technically wrong, if b is 0 the output will be negative
    self.helper_test_variable(Variable.sum([Variable.num(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28, -1, 9, "((a+-1)%28)")
    self.helper_test_variable(Variable.sum([Variable.num(-29), Variable("a", 0, 100), Variable("b", 0, 10)*28]) % 28, -1, 27, "((a+-1)%28)")

  def test_sum_combine_num(self):
    self.helper_test_variable(Variable.sum([Variable.num(29), Variable("a", 0, 10), Variable.num(-23)]), 6, 16, "(a+6)")

  def test_div_factor(self):
    # TODO: this isn't right
    self.helper_test_variable(Variable.sum([Variable.num(-44), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) // 40, -1, 9, "(b+-1)")
  
  def test_mul_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//4, 0, 10, "a")

  def test_mul_div_factor_mul(self):
    self.helper_test_variable((Variable("a", 0, 10)*8)//4, 0, 20, "(a*2)")

  def test_mul_div_factor_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//8, 0, 5, "(a//2)")

  def test_div_remove(self):
    self.helper_test_variable(Variable.sum([Variable("idx0", 0, 127)*4, Variable("idx2", 0, 3)])//4, 0, 127, "idx0")

class TestSymbolicNumeric(unittest.TestCase):
  def helper_test_numeric(self, f):
    for i in range(10):
      v = f(Variable.num(i))
      #print(i, f(i), v.min, v.max)
      self.assertEqual(v.min, v.max)
      self.assertEqual(v.min, f(i))

  def test_times_2(self): self.helper_test_numeric(lambda x: x*2)
  def test_times_2_plus_3(self): self.helper_test_numeric(lambda x: x*2 + 3)
  def test_times_2_plus_3_mod_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)%4)
  def test_times_2_plus_3_div_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)//4)
  def test_times_2_plus_3_div_4_mod_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)//4%4)

if __name__ == '__main__':
  unittest.main()

