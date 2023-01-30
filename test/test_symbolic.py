#!/usr/bin/env python
import unittest
from tinygrad.shape.symbolic import Variable

class TestSymbolic(unittest.TestCase):
  def test_mul_0(self):
    ret = Variable("a", 0, 8)*0
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 0)

  def test_mul_1(self):
    ret = Variable("a", 0, 8)*1
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 8)

  def test_mul_2(self):
    ret = Variable("a", 0, 8)*2
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 16)

  def test_div_1(self):
    ret = Variable("a", 0, 8)//1
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 8)

  def test_mod_1(self):
    ret = Variable("a", 0, 8)%1
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 0)

  def test_add_min_max(self):
    ret = Variable("a", 0, 8) * 2 + 12
    self.assertEqual(ret.min, 12)
    self.assertEqual(ret.max, 16+12)

  def test_div_min_max(self):
    ret = Variable("a", 0, 7) // 2
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 3)

  def test_sum_div_min_max(self):
    ret = Variable.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 5)

  def test_sum_div_factor(self):
    ret = Variable.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 20)

  def test_sum_div_some_factor(self):
    ret = Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 23)

  def test_sum_div_no_factor(self):
    ret = Variable.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 25)
  
  def test_mod_factor(self):
    ret = Variable.sum([Variable("a", 0, 7)*100, Variable("b", 0, 3)*50]) % 100
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 50)
  
  def test_sum_0(self):
    ret = Variable.sum([Variable("a", 0, 7)])
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 7)
  
  def test_mod_remove(self):
    ret = Variable("a", 0, 6)%100
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 6)

  def test_gt_remove(self):
    ret = Variable("a", 0, 6) >= 25
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 0)

  def test_lt_remove(self):
    ret = Variable("a", 0, 6) < -3
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 0)
    ret = Variable("a", 0, 6) < 3
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 1)
    ret = Variable("a", 0, 6) < 8
    self.assertEqual(ret.min, 1)
    self.assertEqual(ret.max, 1)

  def test_and_fold(self):
    ret = Variable.ands([Variable.num(0), Variable("a", 0, 1)])
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 0)

  def test_and_remove(self):
    ret = Variable.ands([Variable.num(1), Variable("a", 0, 1)])
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 1)

  def test_mod_factor(self):
    # this is technically wrong, if b is 0 the output will be negative
    ret = Variable.sum([Variable.num(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28
    self.assertEqual(ret.min, -1)
    self.assertEqual(ret.max, 27)

  def test_div_factor(self):
    # err, that *2 shouldn't be needed
    ret = Variable.sum([Variable.num(-44), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) // 40
    # TODO: this isn't right
    self.assertEqual(str(ret), "(b+-1)")
    self.assertEqual(ret.min, -1)
    self.assertEqual(ret.max, 9)
  
  def test_mul_div(self):
    ret = (Variable("a", 0, 10)*4)//4
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 10)

  def test_mul_div_factor_mul(self):
    ret = (Variable("a", 0, 10)*8)//4
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 20)

  def test_mul_div_factor_div(self):
    ret = (Variable("a", 0, 10)*4)//8
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 5)

  def test_div_remove(self):
    ret = Variable.sum([Variable("idx0", 0, 127)*4, Variable("idx2", 0, 3)])//4
    self.assertEqual(ret.min, 0)
    self.assertEqual(ret.max, 127)

if __name__ == '__main__':
  unittest.main()

