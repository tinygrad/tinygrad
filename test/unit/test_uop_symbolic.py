#!/usr/bin/env python
import unittest, pickle
from typing import Tuple

# TODO: fix all the @unittest.expectedFailure

# *** fake symobilc uops ***

from tinygrad.dtype import dtypes, ConstType
from tinygrad.codegen.linearize import linearize_uop
from tinygrad.codegen.uopgraph import full_graph_rewrite, sym
from tinygrad.ops import UOp, Ops, graph_rewrite, sym_infer
from tinygrad import Variable
import functools

def render(self) -> Tuple[str, ConstType, ConstType]:
  # NOTE: we need STORE so the ALU op has children
  glbl = UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), arg=0)
  uops = linearize_uop(full_graph_rewrite(UOp(Ops.STORE, dtypes.void, (glbl.index(UOp.const(dtypes.int, 0)), self)).sink()))
  rewritten_uop = [uop for uop in uops if uop.op is Ops.STORE][0].src[-1]
  return rewritten_uop.render(simplify=False), rewritten_uop.vmin, rewritten_uop.vmax

def NumNode(val): return UOp.const(dtypes.int, val)
class Node:
  @staticmethod
  def sum(ops): return functools.reduce(lambda x,y: x+y, ops)
  @staticmethod
  def ands(ops): return functools.reduce(lambda x,y: x*y, ops)
  def __floordiv__(a,b,unk): return a//b
def SumNode(x): return Node.sum(x)
def MulNode(x, y): return x*y

# *** leave tests the same

class TestSymbolicPickle(unittest.TestCase):
  def _test_pickle_unpickle(self, x): self.assertEqual(x, pickle.loads(pickle.dumps(x)))
  def test_pickle_variable(self): self._test_pickle_unpickle(Variable("a", 3, 8))
  def test_pickle_variable_times_2(self): self._test_pickle_unpickle(Variable("a", 3, 8)*2)

class TestSymbolic(unittest.TestCase):
  def helper_test_variable(self, v, n, m, s):
    rendered, nmin, nmax = render(v)
    if isinstance(s, tuple): self.assertIn(rendered, s)
    else: self.assertEqual(rendered, s)
    self.assertEqual(nmin, n)
    self.assertEqual(nmax, m)

  def test_cmp_simple(self):
    self.helper_test_variable(Variable("a", 3, 8) < 4, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 3, 8) >= 8, 0, 1, "((a<8)!=True)")

  def test_ge(self):
    self.helper_test_variable(Variable("a", 3, 8) >= 77, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 8) >= 9, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 8) >= 8, 0, 1, "((a<8)!=True)")
    self.helper_test_variable(Variable("a", 3, 8) >= 4, 0, 1, "((a<4)!=True)")
    self.helper_test_variable(Variable("a", 3, 8) >= 3, 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 8) >= 2, 1, 1, "True")

  def test_lt(self):
    self.helper_test_variable(Variable("a", 3, 8) < 77, 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 8) < 9, 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 8) < 8, 0, 1, "(a<8)")
    self.helper_test_variable(Variable("a", 3, 8) < 4, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 3, 8) < 3, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 8) < 2, 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 4) < Variable("b", 5, 6), 1, 1, "True")
    self.helper_test_variable(Variable("a", 3, 5) < Variable("b", 5, 6), 0, 1, "(a<b)")
    self.helper_test_variable(Variable("a", 5, 6) < Variable("b", 3, 5), 0, 0, "False")
    self.helper_test_variable(Variable("a", 3, 4) < Variable("a", 3, 4), 0, 0, "False")

  def test_lt_divides(self):
    expr = (Variable("idx", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3)) < 512
    self.helper_test_variable(expr, 0, 1, "(idx<128)")

  def test_lt_divides_and(self):
    expr = Node.ands([(Variable("idx1", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3)) < 512,
                      (Variable("idx2", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3)) < 512])
    self.helper_test_variable(expr, 0, 1, "((idx1<128)&(idx2<128))")

  def test_lt_factors(self):
    expr = (Variable("idx1", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 256)) < 512
    self.helper_test_variable(expr, 0, 1, ("(((idx1*4)+FLOAT4_INDEX)<512)", "((FLOAT4_INDEX+(idx1*4))<512)"))

  def test_div_reduction(self):
    self.helper_test_variable(Variable("a", 2, 3)//2, 1, 1, "1")

  def test_equality(self):
    idx1 = Variable("idx1", 0, 3)
    idx2 = Variable("idx2", 0, 3)
    assert idx1 is idx1
    assert idx1 is not idx2
    assert idx1*4 is idx1*4
    assert idx1*4 is not idx1*3
    assert idx1*4 is not idx1+4
    assert idx1*4 is not idx2*4
    assert idx1+idx2 is idx1+idx2
    # assert idx1+idx2 is idx2+idx1
    assert idx1+idx2 is not idx2
    # assert idx1*idx2 is idx2*idx1

  def test_factorize(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a*2+a*3, 0, 8*5, "(a*5)")

  def test_factorize_no_mul(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a+a*3, 0, 8*4, "(a*4)")

  def test_neg(self):
    self.helper_test_variable(-Variable("a", 0, 8), -8, 0, "(a*-1)")

  def test_add_1(self):
    self.helper_test_variable(Variable("a", 0, 8)+1, 1, 9, "(a+1)")

  def test_add_num_1(self):
    self.helper_test_variable(Variable("a", 0, 8)+NumNode(1), 1, 9, "(a+1)")

  def test_sub_1(self):
    self.helper_test_variable(Variable("a", 0, 8)-1, -1, 7, "(a+-1)")

  def test_sub_num_1(self):
    self.helper_test_variable(Variable("a", 0, 8)-NumNode(1), -1, 7, "(a+-1)")

  def test_add_self(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a+a, 0, 16, "(a*2)")

  def test_sub_self(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a-a, 0, 0, "0")
    self.helper_test_variable(a*3-a, 0, 16, "(a*2)")

  def test_mul_0(self):
    self.helper_test_variable(Variable("a", 0, 8)*0, 0, 0, "0")

  def test_mul_1(self):
    self.helper_test_variable(Variable("a", 0, 8)*1, 0, 8, "a")

  @unittest.expectedFailure
  def test_mul_neg_1(self):
    self.helper_test_variable((Variable("a", 0, 2)*-1)//3, -1, 0, "((((a*-1)+3)//3)+-1)")

  def test_mul_2(self):
    self.helper_test_variable(Variable("a", 0, 8)*2, 0, 16, "(a*2)")

  def test_div_1(self):
    self.helper_test_variable(Variable("a", 0, 8)//1, 0, 8, "a")

  def test_mod_1(self):
    self.helper_test_variable(Variable("a", 0, 8)%1, 0, 0, "0")

  def test_add_min_max(self):
    self.helper_test_variable(Variable("a", 0, 8) * 2 + 12, 12, 16+12, "((a*2)+12)")

  def test_div_remove(self):
    self.helper_test_variable(Variable("a", 0, 7) // 20, 0, 0, "0")

  def test_div_min_max(self):
    self.helper_test_variable(Variable("a", 1, 7) // 2, 0, 3, "(a//2)")
    self.helper_test_variable(Variable("a", 0, 6) // 2, 0, 3, "(a//2)")

  def test_div_neg_min_max(self):
    self.helper_test_variable(Variable("a", 1, 7) // -2, -3, 0, "(a//-2)")
    self.helper_test_variable(Variable("a", 0, 6) // -2, -3, 0, "(a//-2)")

  def test_sum_div_remove(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 20, 0, 0, "0")

  def test_sum_div_min_max(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2, 0, 5, "((a+b)//2)")

  def test_sum_div_mod_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2, 0, 20, "((a*2)+(b*2))")
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) % 2, 0, 0, "0")

  def test_sum_div_some_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2, 0, 23, ("(((a*5)//2)+(b*2))", "((b*2)+((a*5)//2))"))

  def test_sum_div_trim_const(self):
    self.helper_test_variable((Variable("a", 0, 7)*4 + Variable("b", 0, 3)*4 + 7) // 16, 0, 2, "(((a+b)+1)//4)")

  def test_sum_div_some_partial_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 0, 5, "(((a*3)+(b*3))//8)")
    self.helper_test_variable(Node.sum([NumNode(16), Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 1, 6, "((((a*3)+(b*3))//8)+1)")
    self.helper_test_variable((Variable("a", 0, 7)*30+20)//20, 1, 11, "(((a*3)//2)+1)")

  def test_sum_div_no_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2, 0, 25, "(((a*5)+(b*5))//2)")

  def test_mod_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*100, Variable("b", 0, 3)*50]) % 100, 0, 50, "((b%2)*50)")

  def test_mod_to_sub(self):
    self.helper_test_variable((1+Variable("a",1,2))%2, 0, 1, "(a+-1)")

  def test_mod_congruence(self):
    self.helper_test_variable((3+3*Variable("a",0,3))%4, 0, 3, "((a*-1)+3)")
    self.helper_test_variable((17+13*Variable("a",0,3))%18, 2, 17, "((a*-5)+17)")
    self.helper_test_variable((2+9*Variable("a",0,3))%18, 2, 11, "(((a%2)*9)+2)")

  def test_mod_congruence_mul_add(self):
    self.helper_test_variable((6*(Variable("a", 0, 2)+1))%9, 0, 6, "((a*-3)+6)")

  def test_mod_congruence_multiple_vars(self):
    self.helper_test_variable((9+9*Variable("x",0,3)+9*Variable("y",0,3))%10, 3, 9, "(((x*-1)+(y*-1))+9)")
    self.helper_test_variable((7+9*Variable("x",0,2)+9*Variable("y",0,2)+Variable("z",0,2))%10, 3, 9, "(((z+(x*-1))+(y*-1))+7)")
    self.helper_test_variable((10+12*Variable("x",0,2)+Variable("y", 0, 4)%3)%13, 8, 12, "(((x*-1)+(y%3))+10)")

  def test_div_congruence(self):
    self.helper_test_variable((3+3*Variable("a",0,3))//4, 0, 3, "a")
    self.helper_test_variable((18+17*Variable("a",0,2)+17)//18, 1, 3, "(a+1)")

  def test_div_congruence_multiple_vars(self):
    self.helper_test_variable((9+(9+10)*Variable("x",0,3)+(8+10)*Variable("y",0,2))//10, 0, 10, "((x*2)+(y*2))")

  def test_mod_binary_expression(self):
    self.helper_test_variable((3+Variable("a",0,1))%4, 0, 3, "((a*-3)+3)")
    self.helper_test_variable((3+Variable("a",4,5))%4, 0, 3, "((a*-3)+15)")

  def test_sum_div_const(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, NumNode(3)]) // 4, 0, 7, "a")

  def test_sum_div_const_big(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, NumNode(3)]) // 16, 0, 1, "(a//4)")

  def test_sum_lt_fold(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7) * 4, Variable("b", 0, 3)]) < 16, 0, 1, "(a<4)")
    self.helper_test_variable(Node.sum([Variable("a", 0, 7) * 4, Variable("b", 0, 4)]) < 16, 0, 1,
                              ("(((a*4)+b)<16)", "((b+(a*4))<16)"))
    self.helper_test_variable(Node.sum([Variable("uidx", 0, 3), Variable("a", 0, 1529) * 12]) < (4 * 67), 0, 1, "(a<23)")

  def test_mul_mod_large(self):
    self.helper_test_variable((Variable("a", 0, 20)*10)%9, 0, 8, "(a%9)")

  def test_mul_mod_small(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)%9, 0, 5, "a")

  def test_mod_mod(self):
    self.helper_test_variable((Variable("a", 0, 31)%12)%4, 0, 3, "(a%4)")
    self.helper_test_variable(((4*Variable("a", 0, 31)) % 12) % 4, 0, 0, "0")
    self.helper_test_variable(((5*Variable("a", 0, 31)) % 12) % 5, 0, 4, "(((a*5)%12)%5)")
    self.helper_test_variable((Variable("a", 0, 31) % 4) % 12, 0, 3, "(a%4)")

  def test_mul_mul(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)*9, 0, 5*10*9, "(a*90)")

  def test_mul_lt(self):
    self.helper_test_variable(Variable("a", 0, 5)*4 < 13, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 0, 5)*4 < 16, 0, 1, "(a<4)")
    self.helper_test_variable(Variable("a", 0, 5)*(-2) < 0, 0, 1, "((a*-1)<0)")
    self.helper_test_variable(Variable("a", 0, 5)*4 >= 12, 0, 1, "((a<3)!=True)")
    self.helper_test_variable(Variable("a", 0, 5)*4 >= 13, 0, 1, "((a<4)!=True)")

  def test_div_div(self):
    self.helper_test_variable((Variable("a", 0, 1800)//10)//9, 0, 20, "(a//90)")

  def test_div_const_div(self):
    a = Variable("a", 0, 124)
    self.helper_test_variable((a//2+1)//2, 0, 31, "((a+2)//4)")

  def test_distribute_mul(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 3), Variable("b", 0, 5)])*3, 0, 24, "((a*3)+(b*3))")
    self.helper_test_variable((1+Variable("a", 0, 3))*(-2)+12, 4, 10, "((a*-2)+10)")

  def test_mod_mul_sum(self):
    self.helper_test_variable(Node.sum([Variable("b", 0, 2), Variable("a", 0, 5)*10])%9, 0, 7, ("(b+a)", "(a+b)"))

  def test_sum_0(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)]), 0, 7, "a")

  def test_mod_remove(self):
    self.helper_test_variable(Variable("a", 0, 6)%100, 0, 6, "a")

  def test_big_mod(self):
    # NOTE: we no longer support negative variables
    #self.helper_test_variable(Variable("a", -20, 20)%10, -9, 9, "(a%10)")
    #self.helper_test_variable(Variable("a", -20, 0)%10, -9, 0, "(a%10)")
    #self.helper_test_variable(Variable("a", -20, 1)%10, -9, 1, "(a%10)")
    self.helper_test_variable(Variable("a", 0, 20)%10, 0, 9, "(a%10)")
    #self.helper_test_variable(Variable("a", -1, 20)%10, -1, 9, "(a%10)")

  def test_ge_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) >= 25, 0, 0, "False")

  def test_lt_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) < -3, 0, 0, "False")
    self.helper_test_variable(Variable("a", 0, 6) < 3, 0, 1, "(a<3)")
    self.helper_test_variable(Variable("a", 0, 6) < 8, 1, 1, "True")

  def test_lt_sum_remove(self):
    self.helper_test_variable(Variable("a", 0, 6) + 2 < 3, 0, 1, "(a<1)")

  def test_lt_simple_factor(self):
    self.helper_test_variable((Variable("a", 0, 6)*6+Variable("b", 0, 6)*6) < 8, 0, 1, "(((a*3)+(b*3))<4)")

  def test_lt_sum_factor_rhs_partial(self):
    self.helper_test_variable((Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8) < 4, 0, 1,
                              "((((a*3)+(b*2))+(c*4))<2)")

  def test_lt_sum_factor_rhs_all(self):
    self.helper_test_variable((Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8) < 2, 0, 1,
                              "((((a*3)+(b*2))+(c*4))<1)")

  def test_and_fold(self):
    self.helper_test_variable(Node.ands([NumNode(0), Variable("a", 0, 1)]), 0, 0, "0")

  def test_and_remove(self):
    self.helper_test_variable(Node.ands([NumNode(1), Variable("a", 0, 1)]), 0, 1, "a")

  def test_mod_factor_negative(self):
    self.helper_test_variable(Node.sum([NumNode(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28, 0, 27, "((a+27)%28)")
    self.helper_test_variable(Node.sum([NumNode(-29), Variable("a", 0, 100), Variable("b", 0, 10)*28]) % 28, 0, 27, "((a+27)%28)")

  def test_sum_combine_num(self):
    self.helper_test_variable(Node.sum([NumNode(29), Variable("a", 0, 10), NumNode(-23)]), 6, 16, "(a+6)")

  def test_sum_num_hoisted_and_factors_cancel_out(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 1) * -4 + 1, Variable("a", 0, 1) * 4]), 1, 1, "1")

  def test_div_cancel(self):
    self.helper_test_variable(Node.sum([NumNode(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40])//40, -1, 9, "(b+-1)")

  def test_mod_cancel(self):
    self.helper_test_variable(Node.sum([NumNode(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) % 40, 0, 20, "(a*2)")

  def test_mul_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//4, 0, 10, "a")

  def test_add_div(self):
    # careful about the lower bounds and upper bounds
    self.helper_test_variable((Variable("a", 0, 5)-2)//4, -1, 0, "(((a+2)//4)+-1)")
    self.helper_test_variable((Variable("a", 0, 5)-1)//4, -1, 1, "(((a+3)//4)+-1)")
    self.helper_test_variable((Variable("a", 0, 5))//4, 0, 1, "(a//4)")
    self.helper_test_variable((Variable("a", 0, 5)+1)//4, 0, 1, "((a+1)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+2)//4, 0, 1, "((a+2)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+3)//4, 0, 2, "((a+3)//4)")
    self.helper_test_variable((Variable("a", 0, 5)+4)//4, 1, 2, "((a//4)+1)")
    self.helper_test_variable((Variable("a", 0, 5)+5)//4, 1, 2, "(((a+1)//4)+1)")

  def test_mul_div_factor_mul(self):
    self.helper_test_variable((Variable("a", 0, 10)*8)//4, 0, 20, "(a*2)")

  def test_mul_div_factor_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//8, 0, 5, "(a//2)")

  def test_sum_div_partial_remove(self):
    self.helper_test_variable(Node.sum([Variable("idx0", 0, 127)*4, Variable("idx2", 0, 3)])//4, 0, 127, "idx0")

  @unittest.expectedFailure
  def test_div_numerator_negative(self):
    self.helper_test_variable((Variable("idx", 0, 9)*-10)//11, -9, 0, "((((idx*-10)+99)//11)+-9)")

  def test_div_into_mod(self):
    self.helper_test_variable((Variable("idx", 0, 16)*4)%8//4, 0, 1, "(idx%2)")

  # TODO: simplify the expression
  def test_div_neg_cancel(self):
    self.helper_test_variable((-Variable("idx", 0, 100)+199)//-4 + 50, 1, 26, "((((idx*-1)+199)//-4)+50)")
    self.helper_test_variable((-Variable("idx", 0, 100)+200)//-4 + 50, 0, 25, "((((idx*-1)+200)//-4)+50)")
    self.helper_test_variable((-Variable("idx", 0, 100)+201)//-4 + 50, 0, 25, "((((idx*-1)+201)//-4)+50)")
    self.helper_test_variable((-Variable("idx", 0, 100)+202)//-4 + 50, 0, 25, "((((idx*-1)+202)//-4)+50)")

  def test_sum_div_big_const(self):
    gidx0 = Variable("gidx0", 0, 24)
    self.helper_test_variable((gidx0+19)//20, 0, 2, "((gidx0+19)//20)")
    self.helper_test_variable((gidx0+20)//20, 1, 2, "((gidx0//20)+1)")
    self.helper_test_variable((gidx0+21)//20, 1, 2, "(((gidx0+1)//20)+1)")

  def test_sum_div_complex1(self):
    gidx0 = Variable("gidx0", 0, 24)
    gidx1 = Variable("gidx1", 0, 1)
    gidx2 = Variable("gidx2", 0, 255)
    lidx0 = Variable("lidx0", 0, 1)
    lidx1 = Variable("lidx1", 0, 15)
    lidx2 = Variable("lidx2", 0, 3)
    alu0 = gidx2*640+gidx1*160+(gidx0//5)*2+lidx0*320+lidx1*10
    self.helper_test_variable((alu0+lidx2*2+1)//20, 0, 8192,
                              ("((((((gidx0//5)+lidx2)//5)+lidx1)//2)+(((gidx2*32)+(gidx1*8))+(lidx0*16)))",
                               "((((gidx1*8)+(gidx2*32))+(lidx0*16))+((lidx1+((lidx2+(gidx0//5))//5))//2))"))

  def test_sum_div_complex2(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 1)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+1)//10, 0, 3, ("(((gidx0*2)+lidx2)//5)", "((lidx2+(gidx0*2))//5)"))
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//10, 0, 3, ("(((gidx0*2)+lidx2)//5)", "((lidx2+(gidx0*2))//5)"))
    self.helper_test_variable((gidx0*2+lidx2)//10, 0, 1, "(gidx0//5)")

  def test_sum_div_complex3(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//12, 0, 4, ("(((lidx2//2)+gidx0)//3)", "((gidx0+(lidx2//2))//3)"))
    self.helper_test_variable((lidx2*2+gidx0*4+lidx3)//12, 0, 4, ("(((lidx2//2)+gidx0)//3)", "((gidx0+(lidx2//2))//3)"))

  def test_sum_mul_distribute(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0+lidx2+lidx3)*4, 0, 80, "(((gidx0*4)+(lidx2*4))+(lidx3*4))")

  @unittest.expectedFailure
  def test_variable_divmod(self):
    start_pos = Variable("start_pos", 0, 127)
    v = start_pos + 1
    idx0 = Variable("idx0", 0, 2)
    idx1 = Variable("idx1", 0, start_pos)
    self.helper_test_variable((idx0*v+idx1)//v, 0, 2, "(idx0)")
    self.helper_test_variable((idx0*v+idx1)%v, 0, start_pos, "idx1")

  # TODO: simplify the expression
  def test_div_neg_all_range(self):
    gidx = Variable("gidx", 0, 124)
    lidx = Variable("lidx", 0, 7)
    self.helper_test_variable((-gidx*8-lidx+999)//-4 + 250, 1, 250, "(((((gidx*-8)+(lidx*-1))+999)//-4)+250)")
    self.helper_test_variable((-gidx*8-lidx+1000)//-4 + 250, 0, 250, "(((((gidx*-8)+(lidx*-1))+1000)//-4)+250)")
    self.helper_test_variable((-gidx*8-lidx+1001)//-4 + 250, 0, 250, "(((((gidx*-8)+(lidx*-1))+1001)//-4)+250)")
    self.helper_test_variable((-gidx*8-lidx+1002)//-4 + 250, 0, 250, "(((((gidx*-8)+(lidx*-1))+1002)//-4)+250)")

  # NOTE: tests are not correct in symbolic
  def test_div_neg_then_neg(self):
    # taken from arange opts
    lidx0 = Variable("lidx0", 0, 7)
    lidx1 = Variable("lidx1", 0, 7)
    alu2 = -lidx0-lidx1
    self.helper_test_variable((((alu2+14)//(-32))+4), 4, 4, "4")
    self.helper_test_variable(-(((alu2+14)//(-32))+4), -4, -4, "-4")
    self.helper_test_variable((((alu2+134)//(-32))+4), 0, 1, "(((((lidx0*-1)+(lidx1*-1))+134)//-32)+4)")
    self.helper_test_variable((((alu2+142)//(-32))+4), 0, 0, "0")
    self.helper_test_variable((((alu2+150)//(-32))+4), 0, 0, "0")
    self.helper_test_variable((((alu2+158)//(-32))+4), 0, 0, "0")

  def test_div_mod_recombine(self):
    gidx = Variable("gidx", 0, 124)
    self.helper_test_variable(gidx%4+(gidx//4)*4, 0, 124, "gidx")
    self.helper_test_variable((gidx//4)*4+gidx%4, 0, 124, "gidx")

  def test_div_mod_recombine_folded_mod(self):
    a = Variable("a", 0, 2)
    b = Variable("b", 0, 100)
    self.helper_test_variable((31 * a + 1) % 30 + ((31 * a + 1) // 30) * 30, 1, 63, "((a*31)+1)")
    with self.assertRaises(AssertionError):
      self.helper_test_variable((31 * b + 1) % 18 + ((31 * b + 1) // 18) * 18, 1, 3101, "((b*31)+1)")

  def test_div_mod_recombine_with_gcd(self):
    b = Variable("b", 0, 100)
    exp = (16 * b + 2) % 18 + ((16 * b + 2) // 18) * 18
    self.helper_test_variable(exp, 2, 1602, "((b*16)+2)")
    with self.assertRaises(AssertionError):
      self.helper_test_variable((30 * b + 1) % 18 + ((30 * b + 1) // 18) * 18, 1, 3001, "((b*30)+1)")

  def test_arange_unrolled4(self):
    gidx = Variable("gidx", 0, 2559)
    unrolled_div = (gidx+2561)//4+(gidx+2562)//4+(gidx+2560)//4+(gidx+2559)//4
    self.helper_test_variable(unrolled_div, 2559, 5118, "(gidx+2559)")

  def test_arange_unrolled4_small(self):
    gidx = Variable("gidx", 0, 3)
    unrolled_div = (gidx)//4+(gidx+2)//4+(gidx+3)//4+(gidx+1)//4
    self.helper_test_variable(unrolled_div, 0, 3, "gidx")

    gidx = Variable("gidx", 0, 2)
    unrolled_div = (gidx)//4+(gidx+2)//4+(gidx+3)//4+(gidx+1)//4
    self.helper_test_variable(unrolled_div, 0, 2, "gidx")

    gidx = Variable("gidx", 0, 1)
    unrolled_div = (gidx)//4+(gidx+2)//4+(gidx+3)//4+(gidx+1)//4
    self.helper_test_variable(unrolled_div, 0, 1, "gidx")

  def test_arange_unrolled2(self):
    gidx = Variable("gidx", 0, 2559)
    unrolled_div = (gidx+2559)//2+(gidx+2560)//2+3
    self.helper_test_variable(unrolled_div, 2562, 5121, "(gidx+2562)")

  def test_gated_load(self):
    idx = Variable("idx", 0, 24)
    self.helper_test_variable(idx//4, 0, 6, "(idx//4)")
    # TODO: simplify the true branch
    self.helper_test_variable((idx<4).where(idx//4, idx.const_like(-1)), -1, 6, "((idx//4) if (idx<4) else -1)")

  def test_idiv_lt(self):
    idx = Variable("idx", 0, 24)
    self.helper_test_variable((idx//4<3), 0, 1, "(idx<12)")
    self.helper_test_variable((idx//-4<-3), 0, 1, "((idx//-4)<-3)")

  def test_simplex_lt(self):
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    c = Variable("c", 0, 3)
    d = Variable("d", -3, 3)
    self.helper_test_variable((a<1).ne(True), 0, 1, "((a<1)!=True)")
    self.helper_test_variable((a+b<1).ne(True), 0, 1, "(((a+b)<1)!=True)")
    self.helper_test_variable((a*3+b*4<1).ne(True), 0, 1, "(((a+b)<1)!=True)")
    self.helper_test_variable((a*(-3)+b*4<1).ne(True), 0, 1, "((((a*-3)+(b*4))<1)!=True)")  # negative coeff, should not be simplified
    self.helper_test_variable((a*3+d*4<1).ne(True), 0, 1, "((((a*3)+(d*4))<1)!=True)")  # var can be negative, should not be simplified
    self.helper_test_variable((a+b+c*2<1).ne(True), 0, 1, ("((((a+b)+c)<1)!=True)", "(((c+(a+b))<1)!=True)"))
    self.helper_test_variable((a+b*2+c*4<1).ne(True), 0, 1, ("((((a+b)+c)<1)!=True)", "(((c+(a+b))<1)!=True)"))

  def test_where_removal(self):
    cond = Variable("a", 0, 3) < 2
    u1, u0 = cond.ufix(1), cond.ufix(0)
    self.helper_test_variable(cond, 0, 1, "(a<2)")
    self.helper_test_variable(cond.where(u1, u0), 0, 1, "(a<2)")
    self.helper_test_variable(cond.where(u1, u0).where(u1, u0), 0, 1, "(a<2)")

  def test_where_combine(self):
    cond = Variable("x", 0, 3) < 2
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    aa = cond.where(a, a.ufix(0))
    bb = cond.where(b, b.ufix(1))
    self.helper_test_variable(aa, 0, 3, "(a if (x<2) else 0)")
    self.helper_test_variable(bb, 0, 3, "(b if (x<2) else 1)")
    self.helper_test_variable(aa+bb, 0, 6, "((a+b) if (x<2) else 1)")
    self.helper_test_variable(aa.maximum(bb), 0, 3, "(max(a, b) if (x<2) else 1)")

    # not combining because it increased total ALU
    c = Variable("c", 0, 3)
    cc = cond.where(c, c+1)
    self.helper_test_variable(bb+cc, 0, 7, "((b if (x<2) else 1)+(c if (x<2) else (c+1)))")

    # not combining  # TODO: can combine if it can further simplify?
    ab = cond.where(a, b)
    ba = cond.where(b, a)
    self.helper_test_variable(ab+ba, 0, 6, "((a if (x<2) else b)+(b if (x<2) else a))")

    # not combining  # TODO: can combine if one is identity element const
    self.helper_test_variable(aa+ab, 0, 6, "((a if (x<2) else b)+(a if (x<2) else 0))")

  def test_symbolic_div(self):
    # from symbolic arange
    a = Variable("a", 1, 10)
    denominator = ((a*-2)+1)
    numerator = (((((a*2)+-1)*2)+1)*a)
    self.helper_test_variable(denominator, -19, -1, "((a*-2)+1)")
    self.helper_test_variable(numerator, 3, 390, "(a*((a*4)+-1))")
    self.helper_test_variable((numerator//denominator)<=0, 1, 1, "True")

class TestSymbolicNumeric(unittest.TestCase):
  def helper_test_numeric(self, f):
    MIN, MAX = 0, 10
    # one number
    for i in range(MIN, MAX):
      v = graph_rewrite(f(NumNode(i)), sym)
      self.assertEqual(v.vmin, v.vmax)
      self.assertEqual(v.vmin, f(i))
    for kmin in range(MIN, MAX):
      for kmax in range(MIN, MAX):
        if kmin > kmax: continue
        v = f(Variable("tmp", kmin, kmax))
        values = [f(rv) for rv in range(kmin, kmax+1)]
        # the min and max may not be exact
        self.assertLessEqual(v.vmin, min(values))
        self.assertGreaterEqual(v.vmax, max(values))

  def test_mod_4(self): self.helper_test_numeric(lambda x: (x%4))
  def test_div_4(self): self.helper_test_numeric(lambda x: (x//4))
  def test_plus_1_div_2(self): self.helper_test_numeric(lambda x: (x+1)//2)
  def test_plus_1_mod_2(self): self.helper_test_numeric(lambda x: (x+1)%2)
  def test_times_2(self): self.helper_test_numeric(lambda x: x*2)
  def test_times_2_plus_3(self): self.helper_test_numeric(lambda x: x*2 + 3)
  def test_times_2_plus_3_mod_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)%4)
  def test_times_2_plus_3_div_4(self): self.helper_test_numeric(lambda x: (x*2 + 3)//4)
  def test_times_2_plus_3_div_4_mod_4(self): self.helper_test_numeric(lambda x: ((x*2 + 3)//4)%4)

class TestSymbolicVars(unittest.TestCase):
  def test_simple(self):
    z = NumNode(0)
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    assert z.vars() == z.vars() == set()
    print(a.vars())
    assert a.vars() == a.vars() == {a}
    m = MulNode(a, 3)
    assert m.vars() == {a}
    s = SumNode([a, b, c])
    assert s.vars() == {a, b, c}

  def test_compound(self):
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    assert (a + b * c).vars() == {a, b, c}
    assert (a % 3 + b // 5).vars() == {a, b}
    # TODO: fix me
    with self.assertRaises(AssertionError):
      assert (a + b + c - a).vars() == {b, c}

  def test_dedup(self):
    a = Variable("a", 0, 10)
    assert (a * a).vars() == {a}
    assert (a//4 + a//6).vars() == {a}

class TestSymInfer(unittest.TestCase):
  def test_sym_infer(self):
    a = Variable("a", 0, 10)
    b = Variable("b", 0, 10)
    c = Variable("c", 0, 10)
    var_vals = {a: 2, b: 3, c: 4}
    assert sym_infer(5, var_vals) == 5
    assert sym_infer(a, var_vals) == 2
    assert sym_infer(b, var_vals) == 3
    assert sym_infer(a+b, var_vals) == 5
    assert sym_infer(a-b, var_vals) == -1
    assert sym_infer(a+b+c, var_vals) == 9
    assert sym_infer(a*b, var_vals) == 6
    assert sym_infer(a*b+c, var_vals) == 10

"""
@unittest.skip("not supported on uops yet")
class TestSymRender(unittest.TestCase):
  def test_sym_render(self):
    a = Variable("a", 1, 8)
    b = Variable("b", 1, 10)
    assert sym_render(a) == "a"
    assert sym_render(1) == "1"
    assert sym_render(a+1) == "(1+a)"
    assert sym_render(a*b) == "(a*b)"

@unittest.skip("not supported on uops yet")
class TestSymbolicSymbolicOps(unittest.TestCase):
  def test_node_divmod_node(self):
    i = Variable("i", 1, 10)
    idx0 = Variable("idx0", 0, i*3-1)
    assert NumNode(0) // (Variable("i", 1, 10)*128) == 0
    assert NumNode(0) % (Variable("i", 1, 10)*128) == 0
    assert NumNode(127) // (Variable("i", 1, 10)*128) == 0
    assert NumNode(127) % (Variable("i", 1, 10)*128) == 127
    assert 127 // (Variable("i", 1, 10)*128) == 0
    assert 127 % (Variable("i", 1, 10)*128) == 127
    assert NumNode(128) // (Variable("i", 1, 10)*128 + 128) == 0
    assert NumNode(128) % (Variable("i", 1, 10)*128 + 128) == 128
    assert 128 // (Variable("i", 1, 10)*128 + 128) == 0
    assert 128 % (Variable("i", 1, 10)*128 + 128) == 128
    assert 0 // (Variable("i", 1, 10)*128) == 0
    assert 0 % (Variable("i", 1, 10)*128) == 0
    assert idx0 // (i*3) == 0
    assert idx0 % (i*3) == idx0
    assert i // i == 1
    assert i % i == 0
    assert 128 // NumNode(4) == 32
    assert 128 % NumNode(4) == 0
    assert NumNode(128) // NumNode(4) == 32
    assert NumNode(128) % NumNode(4) == 0

  def test_mulnode_divmod_node(self):
    i = Variable("i", 1, 10)
    idx0 = Variable("idx0", 0, 31)
    # assert (idx0*(i*4+4)) // (i+1) == (idx0*4)
    # assert (idx0*(i*4+4)) % (i+1) == 0
    assert (idx0*i) % i == 0

  def test_sumnode_divmod_sumnode(self):
    i = Variable("i", 1, 10)
    # idx0 = Variable("idx0", 0, 7)
    # idx1 = Variable("idx1", 0, 3)
    # idx2 = Variable("idx2", 0, i)
    # assert (idx0*(i*4+4)+idx1*(i+1)+idx2) // (i+1) == idx0*4+idx1
    # assert (idx0*(i*4+4)+idx1*(i+1)+idx2) % (i+1) == idx2
    assert (i+1) // (i*128+128) == 0
    assert (i+1) % (i*128+128) == (i+1)
    # assert (i+1+idx2) // (i+1) == 1
    # assert (i+1+idx2) % (i+1) == idx2
    # assert (idx0*(i*4+4)+i+1+idx2) // (i+1) == idx0*4+1
    # assert (idx0*(i*4+4)+i+1+idx2) % (i+1) == idx2
    # assert (i*128+128)*2 // (i*128+128) == 2
    # assert (i*128+128)*2 % (i*128+128) == 0

  def test_sumnode_div_numnode_no_factoring(self):
    gid = Variable("gid", 0, 1023)
    lid = Variable("lid", 0, 3)
    expr_before_div = NumNode(-1019)-4*lid-gid
    unfactored_expr = Node.__floordiv__(expr_before_div, NumNode(-16), False)
    factored_expr = Node.__floordiv__(expr_before_div, NumNode(-16), True)
    self.assertEqual(unfactored_expr.render(), "(((lid*4)+1019+gid)//16)")
    self.assertEqual(factored_expr.render(), "(((((3+gid)//4)+2+lid)//4)+63)")

  def test_mod_node_max(self):
    i = Variable("i", 1, 128)
    gidx0 = Variable("gidx0", 0, i)
    mod = gidx0 % 8
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 8
    mod = gidx0 % 2
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 2

    gidx0 = Variable("gidx0", 0, i*8+7)
    mod = gidx0 % 8
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 8
    mod = gidx0 % 2
    assert isinstance(mod, ModNode) and mod.a == gidx0 and mod.b == 2

  def test_nested_variable_mod(self):
    i = Variable("i", 1, 5)
    idx0 = Variable("idx0", 0, i)
    with self.assertRaises(AssertionError):
      assert idx0 % 2 == idx0

  def test_num_node_mul_node(self):
    a = Variable("a", 1, 5)
    b = NumNode(2) * a
    assert b == a * 2
    assert isinstance(b, MulNode)
    b = NumNode(1) * a
    assert b == a
    assert isinstance(b, Variable)
    b = NumNode(0) * a
    assert b == 0
    assert isinstance(b, NumNode)

  def test_substitute(self):
    a = Variable("idx0", 1, 3)
    b = a + 1
    c = b.substitute({a: NumNode(1)})
    assert c == NumNode(2)
"""

class TestSymbolicRealWorld(unittest.TestCase):
  def test_resnet_half(self):
    gidx0 = Variable("gidx0", 0, 3)
    gidx1 = Variable("gidx1", 0, 127)
    gidx2 = Variable("gidx2", 0, 7)
    lidx3 = Variable("lidx3", 0, 7)
    lidx4 = Variable("lidx4", 0, 1)
    lidx5 = Variable("lidx5", 0, 15)

    idx:UOp = ((((1+lidx5)%16)*49)+(((262145+lidx5)//16)*802816)+(gidx0*3211264)+(gidx1*784)+(gidx2*8)+(lidx4*100352)+-13151129600+lidx3)
    idx = graph_rewrite(idx, sym)
    #print(idx.render())
    # NOTE: this used to have 13,151,129,600 in the output which is out of int32 range.
    self.assertIn(idx.render(),
      ("((((((((((lidx5+1)//16)*802816)+(((lidx5+1)%16)*49))+(gidx0*3211264))+(gidx1*784))+(gidx2*8))+(lidx4*100352))+lidx3)+2207744)",
       '((lidx3+((((((((lidx5+1)//16)*802816)+(((lidx5+1)%16)*49))+(gidx0*3211264))+(gidx1*784))+(gidx2*8))+(lidx4*100352)))+2207744)',
      ))

class TestBounds(unittest.TestCase):
  def test_unrolled_arange(self):
    # #include <metal_stdlib>
    # using namespace metal;
    # kernel void r_2560_640_4(device int* data0, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    #   int gidx0 = gid.x; /* 2560 */
    #   int alu0 = (gidx0*(-1));
    #   int alu1 = max((int)((-640)),((((alu0+2559)/(-4))*(-1))+(-640)));
    #   int alu2 = max((int)((-640)),((((alu0+2560)/(-4))*(-1))+(-640)));
    #   int alu3 = max((int)((-640)),((((alu0+2561)/(-4))*(-1))+(-640)));
    #   int alu4 = max((int)((-640)),((((alu0+2562)/(-4))*(-1))+(-640)));
    #   *(data0+gidx0) = ((alu1*(-1))+(alu2*(-1))+(alu4*(-1))+(alu3*(-1))+(-1));
    # }
    gidx0 = Variable("gidx0", 0, 2559)
    assert gidx0.vmin == 0 and gidx0.vmax == 2559
    alu0 = gidx0 * -1
    assert alu0.vmin == -2559 and alu0.vmax == 0
    assert (alu0+2559).vmin == 0 and (alu0+2559).vmax == 2559
    assert ((alu0+2559)//-4).vmin == -639 and ((alu0+2559)//-4).vmax == 0
    assert (((alu0+2559)//-4)*(-1)).vmin == 0 and (((alu0+2559)//-4)*(-1)).vmax == 639

class TestInt64Indexing(unittest.TestCase):
    def test_int64_indexing_comprehensive(self):
        """Test comprehensive int64 indexing scenarios"""
        # Helper function to create int64 constants and operations
        def make_int64(val):
            const = UOp.const(dtypes.int32, val)
            return UOp(Ops.CAST, dtypes.int64, (const,))

        # Test basic overflow cases
        idx = make_int64(dtypes.max(dtypes.int32)) + make_int64(1)  # Overflows int32
        buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), arg=dtypes.max(dtypes.int32) + 1)
        load = UOp.load(buf.index(idx))
        simplified = load.simplify()
        assert simplified.src[0].src[1].dtype == dtypes.int64

        # Test arithmetic operations with large numbers
        idx2 = idx * make_int64(2)  # Large multiplication
        load2 = UOp.load(buf.index(idx2))
        simplified2 = load2.simplify()
        assert simplified2.src[0].src[1].dtype == dtypes.int64

        # Test complex indexing expressions
        base_idx = make_int64(dtypes.max(dtypes.int32) // 2)
        offset = make_int64(dtypes.max(dtypes.int32) // 2)
        complex_idx = base_idx + offset  # Should need int64
        load3 = UOp.load(buf.index(complex_idx))
        simplified3 = load3.simplify()
        assert simplified3.src[0].src[1].dtype == dtypes.int64

        # Test negative indices
        neg_idx = make_int64(dtypes.min(dtypes.int32)) - make_int64(1)
        load4 = UOp.load(buf.index(neg_idx))
        simplified4 = load4.simplify()
        assert simplified4.src[0].src[1].dtype == dtypes.int64

        # Test mixed operations
        mixed_idx = (base_idx * make_int64(2)) + make_int64(42)
        load5 = UOp.load(buf.index(mixed_idx))
        simplified5 = load5.simplify()
        assert simplified5.src[0].src[1].dtype == dtypes.int64

        # Test that small indices stay as int32
        small_idx = UOp.const(dtypes.int32, 100) + UOp.const(dtypes.int32, 42)
        load6 = UOp.load(buf.index(small_idx))
        simplified6 = load6.simplify()
        assert simplified6.src[0].src[1].dtype == dtypes.int32

        # Test store operations
        store1 = UOp.store(buf.index(idx), UOp.const(dtypes.float32, 1.0))
        simplified_store = store1.simplify()
        assert simplified_store.src[0].src[1].dtype == dtypes.int64

        # Test chained operations
        chain_idx = ((base_idx + offset) * make_int64(2)) // make_int64(3)
        load7 = UOp.load(buf.index(chain_idx))
        simplified7 = load7.simplify()
        assert simplified7.src[0].src[1].dtype == dtypes.int64

        # Test boundary conditions
        boundary_idx = UOp.const(dtypes.int32, dtypes.max(dtypes.int32))  # Max int32
        load8 = UOp.load(buf.index(boundary_idx))
        simplified8 = load8.simplify()
        assert simplified8.src[0].src[1].dtype == dtypes.int32  # Should still be int32

        # Test ALU operations
        alu_idx = (idx + make_int64(1)) * make_int64(2)
        load9 = UOp.load(buf.index(alu_idx))
        simplified9 = load9.simplify()
        assert simplified9.src[0].src[1].dtype == dtypes.int64

    def test_int64_conversion_from_int32(self):
        """Test conversion from int32 to int64 for large values"""
        # Create a buffer with a large enough size to require int64 indexing
        buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), arg=0x100000000)
        
        # Test cases for edge points
        test_cases = [
            # Edge cases for int32
            (dtypes.min(dtypes.int32), "min int32"),
            (dtypes.max(dtypes.int32), "max int32"),
            (0, "zero"),
            (-1, "negative one"),
            (1, "positive one"),
            
            # Values that should force int64
            (dtypes.max(dtypes.int32) + 1, "overflow max int32"),
            (dtypes.min(dtypes.int32) - 1, "underflow min int32"),
            
            # Mixed operations
            (dtypes.max(dtypes.int32) + dtypes.max(dtypes.int32), "double max"),
            (dtypes.min(dtypes.int32) + dtypes.min(dtypes.int32), "double min"),
            
            # Large positive and negative values
            (0x100000000, "large positive"),
            (-0x100000000, "large negative"),
            
            # Zero-adjacent values in int64 range
            (-1, "negative one"),
            (0, "zero"),
            (1, "positive one"),
        ]
        
        for value, desc in test_cases:
            # Create index with the test value
            idx = UOp.const(dtypes.int32, value)
            idx._min_max = (value, value)
            
            # Create load operation
            load = UOp.load(buf.index(idx))
            
            # Simplify with pattern matcher
            from tinygrad.codegen.uopgraph import load_store_indexing
            simplified = graph_rewrite(load, load_store_indexing)
            
            # Check if dtype is appropriate based on value
            expected_dtype = dtypes.int64 if (
                value > dtypes.max(dtypes.int32) or 
                value < dtypes.min(dtypes.int32)
            ) else dtypes.int32
            
            self.assertEqual(
                simplified.src[0].src[1].dtype,
                expected_dtype,
                f"Failed for {desc}: value {value} should use {expected_dtype}"
            )

    def test_int64_array_bounds(self):
        """Test array bounds checking with int64 indices"""
        from tinygrad.codegen.uopgraph import load_store_indexing
        
        # Test various array sizes that might need int64 indexing
        sizes = [
            dtypes.max(dtypes.int32),  # max int32
            dtypes.max(dtypes.int32) + 1,  # just over max int32
            dtypes.max(dtypes.int32) * 2,  # large power of 2
            dtypes.max(dtypes.uint32),  # max uint32
        ]
        
        for size in sizes:
            buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), arg=size)
            
            # Test indices at various positions
            indices = [
                (0, "start of array"),
                (size-1, "end of array"),
                (size//2, "middle of array"),
                (42, "small positive index"),
                (-1, "negative index"),
            ]
            
            for idx_val, desc in indices:
                idx = UOp.const(dtypes.int32, idx_val)
                idx._min_max = (idx_val, idx_val)
                
                load = UOp.load(buf.index(idx))
                simplified = graph_rewrite(load, load_store_indexing)
                
                # Only use int64 if the index value requires it
                expected_dtype = dtypes.int64 if (
                    idx_val > dtypes.max(dtypes.int32) or 
                    idx_val < dtypes.min(dtypes.int32)
                ) else dtypes.int32
                
                self.assertEqual(
                    simplified.src[0].src[1].dtype,
                    expected_dtype,
                    f"Failed for array size {size}, {desc}: value {idx_val}"
                )

    def test_int64_mixed_operations(self):
        """Test mixing int32 and int64 operations"""
        from tinygrad.codegen.uopgraph import load_store_indexing
        buf = UOp(Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), arg=0x100000000)
        
        def create_op(val, dtype=dtypes.int32):
            op = UOp.const(dtype, val)
            op._min_max = (val, val)
            return op
        
        test_cases = [
            # Mix int32 and int64 in addition
            (lambda: create_op(dtypes.max(dtypes.int32)) + create_op(1), dtypes.int64),
            
            # Multiplication leading to overflow
            (lambda: create_op(0x40000000) * create_op(2), dtypes.int64),
            
            # Division that should stay in int32
            (lambda: create_op(42) // create_op(2), dtypes.int32),
            
            # Negative numbers
            (lambda: create_op(-dtypes.max(dtypes.int32)) - create_op(2), dtypes.int64),
            
            # Zero mixing
            (lambda: create_op(0) + create_op(dtypes.max(dtypes.int32) + 1), dtypes.int64),
            
            # Complex expressions
            (lambda: (create_op(0x40000000) * create_op(2)) + create_op(42), dtypes.int64),
            
            # Operations that should stay in int32
            (lambda: create_op(100) + create_op(42), dtypes.int32),
        ]
        
        for op_creator, expected_dtype in test_cases:
            idx = op_creator()
            load = UOp.load(buf.index(idx))
            simplified = graph_rewrite(load, load_store_indexing)
            
            self.assertEqual(
                simplified.src[0].src[1].dtype,
                expected_dtype,
                f"Failed for operation {op_creator.__name__}: expected {expected_dtype}"
            )

if __name__ == '__main__':
  unittest.main()
