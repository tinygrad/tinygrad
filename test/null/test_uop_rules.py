import unittest, ast, copy
from tinygrad.uop.ops import Ops, _dtype_rules, _promo_dtype_ops

class TestUOpRules(unittest.TestCase):
  def test_promo_dtype_ops(self):
    # UOp creation computes the dtype of these ops with two srcs of the same dtype directly, their rule must be promo_dtype(src)
    sp = _dtype_rules.spec
    promo = set()
    for op in Ops:
      body = sp.stmts([sp.exprs(st, op) for st in copy.deepcopy(sp.stmts(sp.fdef.body, op))], op)
      if ast.unparse(body[0]) == "return promo_dtype(src)": promo.add(op)
    self.assertEqual(promo, set(_promo_dtype_ops))

if __name__ == "__main__":
  unittest.main()
