#!/usr/bin/env python
import unittest
from tinygrad.shape.symbolic import Variable, NumNode, Node

class TestSymbolic(unittest.TestCase):
  def helper_test_variable(self, v, n, m, s):
    self.assertEqual(v.render(), s)
    self.assertEqual(v.min, n)
    self.assertEqual(v.max, m)

  def test_ge(self):
    self.helper_test_variable(Variable("a", 3, 8)>=77, 0, 0, "0")
    self.helper_test_variable(Variable("a", 3, 8)>=9, 0, 0, "0")
    self.helper_test_variable(Variable("a", 3, 8)>=8, 0, 1, "(a>=8)")
    self.helper_test_variable(Variable("a", 3, 8)>=4, 0, 1, "(a>=4)")
    self.helper_test_variable(Variable("a", 3, 8)>=3, 1, 1, "1")
    self.helper_test_variable(Variable("a", 3, 8)>=2, 1, 1, "1")

  def test_lt(self):
    self.helper_test_variable(Variable("a", 3, 8)<77, 1, 1, "1")
    self.helper_test_variable(Variable("a", 3, 8)<9, 1, 1, "1")
    self.helper_test_variable(Variable("a", 3, 8)<8, 0, 1, "(a<8)")
    self.helper_test_variable(Variable("a", 3, 8)<4, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 3, 8)<3, 0, 0, "0")
    self.helper_test_variable(Variable("a", 3, 8)<2, 0, 0, "0")
  
  def test_div_becomes_num(self):
    assert isinstance(Variable("a", 2, 3)//2, NumNode)

  def test_var_becomes_num(self):
    assert isinstance(Variable("a", 2, 2), NumNode)

  def test_equality(self):
    idx1 = Variable("idx1", 0, 3)
    idx2 = Variable("idx2", 0, 3)
    assert idx1 == idx1
    assert idx1 != idx2
    assert idx1*4 == idx1*4
    assert idx1*4 != idx1*3
    assert idx1*4 != idx1+4
    assert idx1*4 != idx2*4
    assert idx1+idx2 == idx1+idx2
    assert idx1+idx2 == idx2+idx1
    assert idx1+idx2 != idx2

  def test_factorize(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a*2+a*3, 0, 8*5, "(a*5)")

  def test_factorize_no_mul(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a+a*3, 0, 8*4, "(a*4)")

  def test_neg(self):
    self.helper_test_variable(-Variable("a", 0, 8), -8, 0, "(a*-1)")

  def test_add_1(self):
    self.helper_test_variable(Variable("a", 0, 8)+1, 1, 9, "(1+a)")

  def test_add_num_1(self):
    self.helper_test_variable(Variable("a", 0, 8)+Variable.num(1), 1, 9, "(1+a)")

  def test_sub_1(self):
    self.helper_test_variable(Variable("a", 0, 8)-1, -1, 7, "(-1+a)")

  def test_sub_num_1(self):
    self.helper_test_variable(Variable("a", 0, 8)-Variable.num(1), -1, 7, "(-1+a)")

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

  def test_div_neg_min_max(self):
    self.helper_test_variable(Variable("a", 0, 7) // -2, -3, 0, "((a//2)*-1)")

  def test_sum_div_min_max(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2, 0, 5, "((a+b)//2)")

  def test_sum_div_factor(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2, 0, 20, "((a*2)+(b*2))")

  def test_sum_div_some_factor(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2, 0, 23, "(((a*5)//2)+(b*2))")

  def test_sum_div_no_factor(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2, 0, 25, "(((a*5)+(b*5))//2)")
  
  def test_mod_factor(self):
    # NOTE: even though the mod max is 50, it can't know this without knowing about the mul
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*100, Variable("b", 0, 3)*50]) % 100, 0, 99, "((b*50)%100)")

  def test_sum_div_const(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*4, Variable.num(3)]) // 4, 0, 7, "a")

  def test_sum_div_const_big(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)*4, Variable.num(3)]) // 16, 0, 1, "(a//4)")

  def test_mod_mul(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)%9, 0, 5, "a")

  def test_mul_mul(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)*9, 0, 5*10*9, "(a*90)")

  def test_div_div(self):
    self.helper_test_variable((Variable("a", 0, 1800)//10)//9, 0, 20, "(a//90)")

  def test_distribute_mul(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 3), Variable("b", 0, 5)])*3, 0, 24, "((a*3)+(b*3))")

  def test_mod_mul_sum(self):
    self.helper_test_variable(Variable.sum([Variable("b", 0, 2), Variable("a", 0, 5)*10])%9, 0, 7, "(a+b)")
  
  def test_sum_0(self):
    self.helper_test_variable(Variable.sum([Variable("a", 0, 7)]), 0, 7, "a")

  def test_mod_remove(self):
    self.helper_test_variable(Variable("a", 0, 6)%100, 0, 6, "a")

  def test_big_mod(self):
    # NOTE: we no longer support negative variables
    #self.helper_test_variable(Variable("a", -20, 20)%10, -9, 9, "(a%10)")
    #self.helper_test_variable(Variable("a", -20, 0)%10, -9, 0, "(a%10)")
    #self.helper_test_variable(Variable("a", -20, 1)%10, -9, 1, "(a%10)")
    self.helper_test_variable(Variable("a", 0, 20)%10, 0, 9, "(a%10)")
    #self.helper_test_variable(Variable("a", -1, 20)%10, -1, 9, "(a%10)")

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
    self.helper_test_variable(Variable.sum([Variable.num(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28, 0, 27, "((27+a)%28)")
    self.helper_test_variable(Variable.sum([Variable.num(-29), Variable("a", 0, 100), Variable("b", 0, 10)*28]) % 28, 0, 27, "((27+a)%28)")

  def test_sum_combine_num(self):
    self.helper_test_variable(Variable.sum([Variable.num(29), Variable("a", 0, 10), Variable.num(-23)]), 6, 16, "(6+a)")

  def test_div_factor(self):
    self.helper_test_variable(Variable.sum([Variable.num(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) // 40, -1, 9, "(-1+b)")
  
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
    # TODO: why are the negative tests broken? (even if we did support negative variables)
    #MIN, MAX = -10, 10
    MIN, MAX = 0, 10
    # one number
    for i in range(MIN, MAX):
      v = f(Variable.num(i))
      #print(i, f(i), v.min, v.max)
      self.assertEqual(v.min, v.max)
      self.assertEqual(v.min, f(i))
    for kmin in range(MIN, MAX):
      for kmax in range(MIN, MAX):
        if kmin > kmax: continue
        v = f(Variable("tmp", kmin, kmax))
        values = [f(rv) for rv in range(kmin, kmax+1)]
        # the min and max may not be exact
        self.assertLessEqual(v.min, min(values))
        self.assertGreaterEqual(v.max, max(values))

  def test_mod_4(self): self.helper_test_numeric(lambda x: (x%4))
  def test_div_4(self): self.helper_test_numeric(lambda x: (x//4))
  def test_plus_1_div_2(self): self.helper_test_numeric(lambda x: (x+1)//2)
  def test_plus_1_mod_2(self): self.helper_test_numeric(lambda x: (x+1)%2)
  def test_times_2(self): self.helper_test_numeric(lambda x: x*2)
  def test_times_2_plus_3(self): self.helper_test_numeric(lambda x: x*2 + 3)
  def test_times_2_plus_3_mod_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)%4)
  def test_times_2_plus_3_div_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)//4)
  def test_times_2_plus_3_div_4_mod_4(self): self.helper_test_numeric(lambda x: ((x*2 + 3)//4)%4)

if __name__ == '__main__':
  unittest.main()

