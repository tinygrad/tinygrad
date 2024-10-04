#!/usr/bin/env python
import unittest, pickle
from typing import Tuple
#from tinygrad.shape.symbolic import MulNode, SumNode, Variable, NumNode, LtNode, ModNode, Node, sym_render, sym_infer, create_lt_node, create_ge_node

# TODO: fix all the @unittest.expectedFailure

# *** fake symobilc uops ***

from tinygrad.helpers import DEBUG
from tinygrad.dtype import dtypes, PtrDType, ConstType
from tinygrad.codegen.uopgraph import linearize_uop, full_graph_rewrite
from tinygrad.ops import BinaryOps, UOp, UOps, print_uops
from tinygrad.shape.symbolic import Variable
import functools

def render(self) -> Tuple[str, ConstType, ConstType]:
  # NOTE: we need STORE so the ALU op has children
  glbl = UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=0)
  uops = linearize_uop(full_graph_rewrite(UOp(UOps.STORE, dtypes.void, (glbl, UOp.const(dtypes.int, 0), self)).sink()))
  if DEBUG>=5: print_uops(uops)
  from tinygrad.renderer.cstyle import CStyleLanguage
  class TestRenderer(CStyleLanguage):
    code_for_op = {**CStyleLanguage.code_for_op, BinaryOps.IDIV: lambda a,b,dtype: f"({a}//{b})"}
  rewritten_uop = [uop for uop in uops if uop.op is UOps.STORE][0].src[-1]
  fxn = TestRenderer().render("", uops)
  return fxn.split("data0[0] = ")[1].split(";")[0], rewritten_uop.vmin, rewritten_uop.vmax

def NumNode(val): return UOp.const(dtypes.int, val)
class Node:
  @staticmethod
  def sum(ops): return functools.reduce(lambda x,y: x+y, ops)
  @staticmethod
  def ands(ops): return functools.reduce(lambda x,y: x*y, ops)
  def __floordiv__(a,b,unk): return a//b
def create_lt_node(v, n): return v.lt(n)
def create_ge_node(v, n): return v.ge(n)
def SumNode(x): return Node.sum(x)
def MulNode(x, y): return x*y

# *** leave tests the same

@unittest.skip("not supported on uops yet")
class TestSymbolicPickle(unittest.TestCase):
  def _test_pickle_unpickle(self, x): self.assertEqual(x, pickle.loads(pickle.dumps(x)))
  def test_pickle_variable(self): self._test_pickle_unpickle(Variable("a", 3, 8))
  def test_pickle_variable_times_2(self): self._test_pickle_unpickle(Variable("a", 3, 8)*2)

class TestSymbolic(unittest.TestCase):
  def helper_test_variable(self, v, n, m, s):
    rendered, nmin, nmax = render(v)
    if isinstance(s, set):
      self.assertIn(rendered, s)
    else:
      self.assertEqual(rendered, s)
    self.assertEqual(nmin, n)
    self.assertEqual(nmax, m)

  def test_cmp_simple(self):
    self.helper_test_variable(create_lt_node(Variable("a", 3, 8), 4), 0, 1, "(a<4)")
    self.helper_test_variable(create_ge_node(Variable("a", 3, 8), 8), 0, 1, {"((a*-1)<-7)", "((a*(-1))<(-7))", '((a<8)!=1)'})

  def test_ge(self):
    self.helper_test_variable(create_ge_node(Variable("a", 3, 8), 77), 0, 0, "0")
    self.helper_test_variable(create_ge_node(Variable("a", 3, 8), 9), 0, 0, "0")
    self.helper_test_variable(create_ge_node(Variable("a", 3, 8), 8), 0, 1, {"((a*-1)<-7)", "((a*(-1))<(-7))", '((a<8)!=1)'})
    self.helper_test_variable(create_ge_node(Variable("a", 3, 8), 4), 0, 1, {"((a*-1)<-3)", "((a*(-1))<(-3))", '((a<4)!=1)'})
    self.helper_test_variable(create_ge_node(Variable("a", 3, 8), 3), 1, 1, "1")
    self.helper_test_variable(create_ge_node(Variable("a", 3, 8), 2), 1, 1, "1")

  def test_lt(self):
    self.helper_test_variable(create_lt_node(Variable("a", 3, 8), 77), 1, 1, "1")
    self.helper_test_variable(create_lt_node(Variable("a", 3, 8), 9), 1, 1, "1")
    self.helper_test_variable(create_lt_node(Variable("a", 3, 8), 8), 0, 1, "(a<8)")
    self.helper_test_variable(create_lt_node(Variable("a", 3, 8), 4), 0, 1, "(a<4)")
    self.helper_test_variable(create_lt_node(Variable("a", 3, 8), 3), 0, 0, "0")
    self.helper_test_variable(create_lt_node(Variable("a", 3, 8), 2), 0, 0, "0")

  def test_ge_divides(self):
    expr = create_lt_node(Variable("idx", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3), 512)
    self.helper_test_variable(expr, 0, 1, "(idx<128)")

  def test_ge_divides_and(self):
    expr = Node.ands([create_lt_node(Variable("idx1", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3), 512),
                      create_lt_node(Variable("idx2", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3), 512)])
    self.helper_test_variable(expr, 0, 1, {"((idx1<128) and (idx2<128))", "((idx1<128)&(idx2<128))"})
    # # bool divided by int is not allowed
    # expr = Node.ands([create_lt_node(Variable("idx1", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 3), 512),
    #                   create_lt_node(Variable("idx2", 0, 511)*4 + Variable("FLOAT8_INDEX", 0, 7), 512)])
    # self.helper_test_variable(expr//4, 0, 0, "0")

  def test_lt_factors(self):
    expr = create_lt_node(Variable("idx1", 0, 511)*4 + Variable("FLOAT4_INDEX", 0, 256), 512)
    self.helper_test_variable(expr, 0, 1, {"(((idx1*4)+FLOAT4_INDEX)<512)", "(((FLOAT4_INDEX//4)+idx1)<128)"})

  def test_div_reduction(self):
    self.helper_test_variable(Variable("a", 2, 3)//2, 1, 1, "1")

  @unittest.expectedFailure
  def test_var_becomes_num(self):
    self.helper_test_variable(Variable("a", 2, 2), 2, 2, "2")

  @unittest.expectedFailure
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
    assert idx1*idx2 == idx2*idx1

  #def test_numnode_eq_int(self):
  #  n1 = NumNode(1)
  #  n2 = NumNode(2)
  #  assert n1 == 1
  #  assert n2 == 2
  #  assert n1 != n2
  #  assert hash(n1) == hash(1)
  #  assert hash(n2) == hash(2)

  def test_factorize(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a*2+a*3, 0, 8*5, "(a*5)")

  def test_factorize_no_mul(self):
    a = Variable("a", 0, 8)
    self.helper_test_variable(a+a*3, 0, 8*4, "(a*4)")

  def test_neg(self):
    self.helper_test_variable(-Variable("a", 0, 8), -8, 0, {"(a*-1)", "(a*(-1))"})

  def test_add_1(self):
    self.helper_test_variable(Variable("a", 0, 8)+1, 1, 9, {"(1+a)", "(a+1)"})

  def test_add_num_1(self):
    self.helper_test_variable(Variable("a", 0, 8)+NumNode(1), 1, 9, {"(1+a)", "(a+1)"})

  def test_sub_1(self):
    self.helper_test_variable(Variable("a", 0, 8)-1, -1, 7, {"(-1+a)", "(a+(-1))", "(a+-1)"})

  def test_sub_num_1(self):
    self.helper_test_variable(Variable("a", 0, 8)-NumNode(1), -1, 7, {"(-1+a)", "(a+(-1))", "(a+-1)"})

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
    self.helper_test_variable(Variable("a", 0, 7) // 2, 0, 3, "(a//2)")

  @unittest.expectedFailure
  def test_div_neg_min_max(self):
    self.helper_test_variable(Variable("a", 0, 7) // -2, -4, 0, "((((a*-1)+8)//2)+-4)")
    self.helper_test_variable(Variable("a", 0, 6) // -2, -3, 0, "((((a*-1)+6)//2)+-3)")

  def test_sum_div_remove(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 20, 0, 0, "0")

  def test_sum_div_min_max(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7), Variable("b", 0, 3)]) // 2, 0, 5, "((a+b)//2)")

  def test_sum_div_mod_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) // 2, 0, 20, "((a*2)+(b*2))")
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, Variable("b", 0, 3)*4]) % 2, 0, 0, "0")

  def test_sum_div_some_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*4]) // 2, 0, 23, {"(((a*5)//2)+(b*2))", "((b*2)+((a*5)//2))"})

  def test_sum_div_trim_const(self):
    self.helper_test_variable((Variable("a", 0, 7)*4 + Variable("b", 0, 3)*4 + 7) // 16, 0, 2, {"((1+a+b)//4)", "((a+b+1)//4)"})

  def test_sum_div_some_partial_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 0, 5, "(((a*3)+(b*3))//8)")
    self.helper_test_variable(Node.sum([NumNode(16), Variable("a", 0, 7)*6, Variable("b", 0, 7)*6]) // 16, 1, 6, "((((a*3)+(b*3))//8)+1)")
    self.helper_test_variable((Variable("a", 0, 7)*30+20)//20, 1, 11, "(((a*3)//2)+1)")

  def test_sum_div_no_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*5, Variable("b", 0, 3)*5]) // 2, 0, 25, "(((a*5)+(b*5))//2)")

  def test_mod_factor(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*100, Variable("b", 0, 3)*50]) % 100, 0, 99, "((b*50)%100)")

  def test_mod_to_sub(self):
    # This is mod reduction
    self.helper_test_variable((1+Variable("a",1,2))%2, 0, 1, {"(-1+a)", "(a+(-1))", "(a+-1)"})

  def test_sum_div_const(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, NumNode(3)]) // 4, 0, 7, "a")

  def test_sum_div_const_big(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 7)*4, NumNode(3)]) // 16, 0, 1, "(a//4)")

  def test_sum_lt_fold(self):
    self.helper_test_variable(create_lt_node(Node.sum([Variable("a", 0, 7) * 4, Variable("b", 0, 3)]), 16), 0, 1, "(a<4)")
    self.helper_test_variable(create_lt_node(Node.sum([Variable("a", 0, 7) * 4, Variable("b", 0, 4)]), 16), 0, 1,
                              {"(((a*4)+b)<16)", "(((b//4)+a)<4)"})
    # TODO: fix
    with self.assertRaises(AssertionError):
      self.helper_test_variable(create_lt_node(Node.sum([Variable("uidx", 0, 3), Variable("a", 0, 1529) * 12]), (4 * 67)), 0, 1, "(a<23)")

  def test_mul_mod_large(self):
    self.helper_test_variable((Variable("a", 0, 20)*10)%9, 0, 8, "(a%9)")

  def test_mul_mod_small(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)%9, 0, 5, "a")

  def test_mod_mod(self):
    self.helper_test_variable((Variable("a", 0, 31)%12)%4, 0, 3, "(a%4)")
    self.helper_test_variable(((4*Variable("a", 0, 31)) % 12) % 4, 0, 0, "0")
    self.helper_test_variable(((5*Variable("a", 0, 31)) % 12) % 5, 0, 4, {"(((a*5)%12)%5)", "(((5*a)%12)%5)"})
    self.helper_test_variable((Variable("a", 0, 31) % 4) % 12, 0, 3, "(a%4)")

  def test_mul_mul(self):
    self.helper_test_variable((Variable("a", 0, 5)*10)*9, 0, 5*10*9, "(a*90)")

  def test_mul_lt(self):
    self.helper_test_variable(create_lt_node(Variable("a", 0, 5)*4,13), 0, 1, "(a<4)")
    self.helper_test_variable(create_lt_node(Variable("a", 0, 5)*4,16), 0, 1, "(a<4)")
    self.helper_test_variable(create_lt_node(Variable("a", 0, 5)*(-2),0), 0, 1, {"((a*-1)<0)", "((a*(-1))<0)"})
    self.helper_test_variable(create_ge_node(Variable("a", 0, 5)*4,12), 0, 1, {"((a*-1)<-2)", "((a*(-1))<(-2))", '((a<3)!=1)'})
    self.helper_test_variable(create_ge_node(Variable("a", 0, 5)*4,13), 0, 1, {"((a*-1)<-3)", "((a*(-1))<(-3))", '((a<4)!=1)'})

  def test_div_div(self):
    self.helper_test_variable((Variable("a", 0, 1800)//10)//9, 0, 20, "(a//90)")

  def test_distribute_mul(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 3), Variable("b", 0, 5)])*3, 0, 24, {"((a*3)+(b*3))", "((a+b)*3)"})
    self.helper_test_variable((1+Variable("a", 0, 3))*(-2)+12, 4, 10, {"((a*-2)+10)", "((a*(-2))+10)"})

  def test_mod_mul_sum(self):
    self.helper_test_variable(Node.sum([Variable("b", 0, 2), Variable("a", 0, 5)*10])%9, 0, 7, {"(a+b)", "(b+a)"})

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
    self.helper_test_variable(create_ge_node(Variable("a", 0, 6), 25), 0, 0, "0")

  def test_lt_remove(self):
    self.helper_test_variable(create_lt_node(Variable("a", 0, 6), -3), 0, 0, "0")
    self.helper_test_variable(create_lt_node(Variable("a", 0, 6), 3), 0, 1, "(a<3)")
    self.helper_test_variable(create_lt_node(Variable("a", 0, 6), 8), 1, 1, "1")

  def test_lt_sum_remove(self):
    self.helper_test_variable(create_lt_node(Variable("a", 0, 6) + 2, 3), 0, 1, "(a<1)")

  def test_lt_simple_factor(self):
    self.helper_test_variable(create_lt_node(Variable("a", 0, 6)*6+Variable("b", 0, 6)*6, 8), 0, 1,
                              "(((a*3)+(b*3))<4)")

  def test_lt_sum_factor_rhs_partial(self):
    self.helper_test_variable(create_lt_node(Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8, 4), 0, 1,
                              "(((a*3)+(b*2)+(c*4))<2)")

  def test_lt_sum_factor_rhs_all(self):
    self.helper_test_variable(create_lt_node(Variable("a", 0, 6)*6 + Variable("b", 0, 6)*4 + Variable("c", 0, 6)*8, 2), 0, 1,
                              "(((a*3)+(b*2)+(c*4))<1)")

  def test_and_fold(self):
    self.helper_test_variable(Node.ands([NumNode(0), Variable("a", 0, 1)]), 0, 0, "0")

  def test_and_remove(self):
    self.helper_test_variable(Node.ands([NumNode(1), Variable("a", 0, 1)]), 0, 1, "a")

  def test_mod_factor_negative(self):
    self.helper_test_variable(Node.sum([NumNode(-29), Variable("a", 0, 10), Variable("b", 0, 10)*28]) % 28, 0, 27, {"((27+a)%28)", "((a+27)%28)"})
    self.helper_test_variable(Node.sum([NumNode(-29), Variable("a", 0, 100), Variable("b", 0, 10)*28]) % 28, 0, 27, {"((27+a)%28)", "((a+27)%28)"})

  def test_sum_combine_num(self):
    self.helper_test_variable(Node.sum([NumNode(29), Variable("a", 0, 10), NumNode(-23)]), 6, 16, {"(6+a)", "(a+6)"})

  def test_sum_num_hoisted_and_factors_cancel_out(self):
    self.helper_test_variable(Node.sum([Variable("a", 0, 1) * -4 + 1, Variable("a", 0, 1) * 4]), 1, 1, "1")

  def test_div_cancel(self):
    self.helper_test_variable(Node.sum([NumNode(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40])//40, -1, 9, {"(-1+b)", "(b+(-1))", "(b+-1)"})

  def test_mod_cancel(self):
    self.helper_test_variable(Node.sum([NumNode(-40), Variable("a", 0, 10)*2, Variable("b", 0, 10)*40]) % 40, 0, 20, "(a*2)")

  def test_mul_div(self):
    self.helper_test_variable((Variable("a", 0, 10)*4)//4, 0, 10, "a")

  def test_add_div(self):
    # careful about the lower bounds and upper bounds
    self.helper_test_variable((Variable("a", 0, 5)-2)//4, -1, 0, {"(((a+2)//4)+(-1))", "(((a+2)//4)+-1)"})
    self.helper_test_variable((Variable("a", 0, 5)-1)//4, -1, 1, {"(((a+3)//4)+(-1))", "(((a+3)//4)+-1)"})
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
    self.helper_test_variable((Variable("idx", 0, 16)*4)%8//4, 0, 1, "(((idx*4)%8)//4)")

  # TODO: simplify the expression
  def test_div_neg_cancel(self):
    self.helper_test_variable((-Variable("idx", 0, 100)+199)//-4 + 50, 1, 26, {"((((idx*(-1))+199)//(-4))+50)", "((((idx*-1)+199)//-4)+50)"})
    self.helper_test_variable((-Variable("idx", 0, 100)+200)//-4 + 50, 0, 25, {"((((idx*(-1))+200)//(-4))+50)", "((((idx*-1)+200)//-4)+50)"})
    self.helper_test_variable((-Variable("idx", 0, 100)+201)//-4 + 50, 0, 25, {"((((idx*(-1))+201)//(-4))+50)", "((((idx*-1)+201)//-4)+50)"})

  def test_sum_div_big_const(self):
    gidx0 = Variable("gidx0", 0, 24)
    self.helper_test_variable((gidx0+19)//20, 0, 2, {"((19+gidx0)//20)", "((gidx0+19)//20)"})
    self.helper_test_variable((gidx0+20)//20, 1, 2, "((gidx0//20)+1)")
    self.helper_test_variable((gidx0+21)//20, 1, 2, {"(((1+gidx0)//20)+1)", "(((gidx0+1)//20)+1)"})

  def test_sum_div_complex1(self):
    gidx0 = Variable("gidx0", 0, 24)
    gidx1 = Variable("gidx1", 0, 1)
    gidx2 = Variable("gidx2", 0, 255)
    lidx0 = Variable("lidx0", 0, 1)
    lidx1 = Variable("lidx1", 0, 15)
    lidx2 = Variable("lidx2", 0, 3)
    alu0 = gidx2*640+gidx1*160+(gidx0//5)*2+lidx0*320+lidx1*10
    self.helper_test_variable((alu0+lidx2*2+1)//20, 0, 8192, {"((((((gidx0//5)+lidx2)//5)+lidx1)//2)+(gidx1*8)+(gidx2*32)+(lidx0*16))",
                                                              "((((((gidx0//5)+lidx2)//5)+lidx1)//2)+(gidx2*32)+(gidx1*8)+(lidx0*16))"})

  def test_sum_div_complex2(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 1)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+1)//10, 0, 3, "(((gidx0*2)+lidx2)//5)")
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//10, 0, 3, "(((gidx0*2)+lidx2)//5)")
    self.helper_test_variable((gidx0*2+lidx2)//10, 0, 1, "(gidx0//5)")

  def test_sum_div_complex3(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0*4+lidx2*2+lidx3)//12, 0, 4, "(((lidx2//2)+gidx0)//3)")
    self.helper_test_variable((lidx2*2+gidx0*4+lidx3)//12, 0, 4, "(((lidx2//2)+gidx0)//3)")

  def test_sum_mul_distribute(self):
    gidx0 = Variable("gidx0", 0, 7)
    lidx2 = Variable("lidx2", 0, 12)
    lidx3 = Variable("lidx3", 0, 1)
    self.helper_test_variable((gidx0+lidx2+lidx3)*4, 0, 80, "((gidx0*4)+(lidx2*4)+(lidx3*4))")

  @unittest.expectedFailure
  def test_variable_divmod(self):
    start_pos = Variable("start_pos", 0, 127)
    v = start_pos + 1
    idx0 = Variable("idx0", 0, 2)
    idx1 = Variable("idx1", 0, start_pos)
    self.helper_test_variable((idx0*v+idx1)//v, 0, 2, "(idx0)")
    self.helper_test_variable((idx0*v+idx1)%v, 0, start_pos, "idx1")

  # *** below are uop_symbolic only

  # NOTE: tests are not correct in symbolic
  # TODO: simplify the expression
  def test_div_neg_all_range(self):
    gidx = Variable("gidx", 0, 124)
    lidx = Variable("lidx", 0, 7)
    self.helper_test_variable((-gidx*8-lidx+999)//-4 + 250, 1, 250, "((((gidx*-8)+(lidx*-1)+999)//-4)+250)")
    self.helper_test_variable((-gidx*8-lidx+1000)//-4 + 250, 0, 250, "((((gidx*-8)+(lidx*-1)+1000)//-4)+250)")
    self.helper_test_variable((-gidx*8-lidx+1001)//-4 + 250, 0, 250, "((((gidx*-8)+(lidx*-1)+1001)//-4)+250)")
    self.helper_test_variable((-gidx*8-lidx+1002)//-4 + 250, 0, 250, "((((gidx*-8)+(lidx*-1)+1002)//-4)+250)")

  # NOTE: tests are not correct in symbolic
  def test_div_neg_then_neg(self):
    # taken from arange opts
    lidx0 = Variable("lidx0", 0, 7)
    lidx1 = Variable("lidx1", 0, 7)
    alu2 = -lidx0-lidx1
    self.helper_test_variable((((alu2+14)//(-32))+4), 4, 4, "4")
    self.helper_test_variable(-(((alu2+14)//(-32))+4), -4, -4, {"(-4)", "-4"})
    self.helper_test_variable((((alu2+134)//(-32))+4), 0, 1, {"((((lidx0*(-1))+(lidx1*(-1))+134)//(-32))+4)",
                                                              "((((lidx0*-1)+(lidx1*-1)+134)//-32)+4)"})
    self.helper_test_variable((((alu2+142)//(-32))+4), 0, 0, "0")
    self.helper_test_variable((((alu2+150)//(-32))+4), 0, 0, "0")
    self.helper_test_variable((((alu2+158)//(-32))+4), 0, 0, "0")

  def test_div_mod_recombine(self):
    gidx = Variable("gidx", 0, 124)
    self.helper_test_variable(gidx%4+(gidx//4)*4, 0, 124, "gidx")
    self.helper_test_variable((gidx//4)*4+gidx%4, 0, 124, "gidx")

  def test_arange_unrolled4(self):
    gidx = Variable("gidx", 0, 2559)
    unrolled_div = (gidx+2561)//4+(gidx+2562)//4+(gidx+2560)//4+(gidx+2559)//4
    self.helper_test_variable(unrolled_div, 2559, 5118, "(gidx+2559)")

  def test_arange_unrolled2(self):
    gidx = Variable("gidx", 0, 2559)
    unrolled_div = (gidx+2559)//2+(gidx+2560)//2+3
    self.helper_test_variable(unrolled_div, 2562, 5121, "(gidx+2562)")

  def test_gated_load(self):
    idx = Variable("idx", 0, 24)
    self.helper_test_variable(idx//4, 0, 6, "(idx//4)")
    # TODO: simplify the true branch
    self.helper_test_variable(idx.lt(4).where(idx//4, idx.const_like(-1)), -1, 6, {"((idx<4)?(idx//4):(-1))", "((idx<4)?(idx//4):-1)"})

  def test_idiv_lt(self):
    idx = Variable("idx", 0, 24)
    self.helper_test_variable((idx//4).lt(3), 0, 1, "(idx<12)")
    self.helper_test_variable((idx//-4).lt(-3), 0, 1, {"((idx//(-4))<(-3))", "((idx//-4)<-3)"})

  def test_simplex_lt(self):
    a = Variable("a", 0, 3)
    b = Variable("b", 0, 3)
    c = Variable("c", 0, 3)
    d = Variable("d", -3, 3)
    self.helper_test_variable((a).lt(1).ne(True), 0, 1, "((a<1)!=1)")
    self.helper_test_variable((a+b).lt(1).ne(True), 0, 1, "(((a+b)<1)!=1)")
    self.helper_test_variable((a*3+b*4).lt(1).ne(True), 0, 1, "(((a+b)<1)!=1)")
    self.helper_test_variable((a*(-3)+b*4).lt(1).ne(True), 0, 1, "((((a*-3)+(b*4))<1)!=1)")  # negative coeff, should not be simplified
    self.helper_test_variable((a*3+d*4).lt(1).ne(True), 0, 1, "((((a*3)+(d*4))<1)!=1)")  # var can be negative, should not be simplified
    self.helper_test_variable((a+b+c*2).lt(1).ne(True), 0, 1, "(((a+b+c)<1)!=1)")
    self.helper_test_variable((a+b*2+c*4).lt(1).ne(True), 0, 1, "(((a+b+c)<1)!=1)")

@unittest.skip("not supported on uops yet")
class TestSymbolicNumeric(unittest.TestCase):
  def helper_test_numeric(self, f):
    # TODO: why are the negative tests broken? (even if we did support negative variables)
    #MIN, MAX = -10, 10
    MIN, MAX = 0, 10
    # one number
    for i in range(MIN, MAX):
      v = f(NumNode(i))
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

@unittest.skip("not supported on uops yet")
class TestSymbolicMinMax(unittest.TestCase):
  def test_min_max_known(self):
    a = Variable("a", 1, 8)
    assert max(1, a) == max(a, 1) == a
    assert min(1, a) == min(a, 1) == 1

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

  def test_node_lt_node(self):
    a = Variable("a", 1, 5)
    b = Variable("b", 6, 9)
    c = Variable("c", 1, 10)
    d = Variable("d", 5, 10)
    # if the comparison output is always the same, it folds to num
    assert create_lt_node(a, b) == NumNode(1)
    assert create_lt_node(b, a) == NumNode(0)
    assert create_lt_node(d, a) == NumNode(0)
    assert create_lt_node(a, a) == NumNode(0)
    assert create_lt_node(a, a) == NumNode(0)
    # if it remains as a LtNode, bool is always true and (min, max) == (0, 1)
    a_lt_c = create_lt_node(a, c)
    assert isinstance(a_lt_c, LtNode) and a_lt_c.min == 0 and a_lt_c.max == 1
    assert a_lt_c
    # same when comparing with a constant
    a_lt_3 = create_lt_node(a, 3)
    assert a_lt_3 and a_lt_3.min == 0 and a_lt_3.max == 1

  def test_sumnode_mulnode_lt(self):
    a = Variable("a", 1, 2)
    b = Variable("b", 1, 2)
    c = Variable("c", 1, 2)
    x = SumNode([MulNode(a, b), c])
    with self.assertRaises(AssertionError):
      create_lt_node(x, 3)

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
  @unittest.expectedFailure
  def test_resnet_half(self):
    gidx0 = Variable("gidx0", 0, 3)
    gidx1 = Variable("gidx1", 0, 127)
    gidx2 = Variable("gidx2", 0, 7)
    lidx3 = Variable("lidx3", 0, 7)
    lidx4 = Variable("lidx4", 0, 1)
    lidx5 = Variable("lidx5", 0, 15)

    idx = ((((1+lidx5)%16)*49)+(((262145+lidx5)//16)*802816)+(gidx0*3211264)+(gidx1*784)+(gidx2*8)+(lidx4*100352)+-13151129600+lidx3)
    print(idx.render())
    # NOTE: this used to have 13,151,129,600 in the output which is out of int32 range.
    assert idx.render() == "((((1+lidx5)%16)*49)+(((1+lidx5)//16)*802816)+(gidx0*3211264)+(gidx1*784)+(gidx2*8)+(lidx4*100352)+2207744+lidx3)"

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

if __name__ == '__main__':
  unittest.main()
