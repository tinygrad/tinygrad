import unittest
from tinygrad import dtypes
from tinygrad.ops import UOp, symbolic, RewriteContext

def graph_rewrite_map(node, pm):
  ctx = RewriteContext(pm)
  rewritten_sink = ctx.rewrite(node)
  return rewritten_sink, ctx.replace

class TestRewriteMap(unittest.TestCase):
  def test_add_zero(self):
    # Build a small graph: add(0, add(const=0, const=5))
    zero_node = UOp.const(dtypes.int, 0)
    five_node = UOp.const(dtypes.int, 5)
    inner_add = zero_node + five_node
    root_add = zero_node + inner_add

    # Perform top-down rewrite
    rewritten_sink, node_map = graph_rewrite_map(root_add, symbolic)

    # We expect that add(0, add(0, 5)) -> add(0, 5) -> 5
    # So the final node should be 'five_node'
    assert rewritten_sink == five_node

    # Check the mapping
    assert node_map[root_add] == five_node
    assert node_map[inner_add] == five_node
    # zero_node and five_node map to themselves
    assert node_map[zero_node] == zero_node
    assert node_map[five_node] == five_node

  def test_double_neg(self):
    """
    Test rewriting neg(neg(5)) => 5 using symbolic.
    """
    # In some versions of TinyGrad, you might do: (-(-five_node))
    five_node = UOp.const(dtypes.int, 5)
    # If your code allows UOp(...), do that; else you might do something like:
    # double_neg_five = -(-five_node)
    # But let's be explicit:
    neg_five = -five_node
    double_neg_five = -neg_five

    rewritten_sink, node_map = graph_rewrite_map(double_neg_five, symbolic)

    # Expect neg(neg(5)) -> 5
    self.assertEqual(rewritten_sink, five_node)
    # node_map should map double_neg_five -> five_node
    self.assertEqual(node_map[double_neg_five], five_node)
    # five_node maps to itself
    self.assertEqual(node_map[five_node], five_node)

  def test_add_zero_and_double_neg(self):
    """
    Combine both rewrites: add(0, neg(neg(5))) => add(0, 5) => 5
    """
    zero_node = UOp.const(dtypes.int, 0)
    five_node = UOp.const(dtypes.int, 5)
    neg_five = -five_node
    double_neg_five = -neg_five
    root_add = zero_node + double_neg_five

    rewritten_sink, node_map = graph_rewrite_map(root_add, symbolic)

    # Expect final node is 'five_node'
    self.assertEqual(rewritten_sink, five_node)
    # node_map: root_add -> five_node, double_neg_five -> five_node
    self.assertEqual(node_map[root_add], five_node)
    self.assertEqual(node_map[double_neg_five], five_node)
    # zero_node, five_node map to themselves
    self.assertEqual(node_map[zero_node], zero_node)
    self.assertEqual(node_map[five_node], five_node)

  def test_multi_var_rewrites(self):
    x_var = UOp.variable('x', 0, 10)
    y_var = UOp.variable('y', -5, 5)
    zero_node = UOp.const(dtypes.int, 0)

    sum_with_zero = y_var + zero_node    # (y + 0)
    combined = x_var + sum_with_zero     # x + (y + 0)
    double_neg = -(-combined)           # neg(neg(x + y))
    final_expr = zero_node + double_neg  # 0 + (x + y)

    rewritten_sink, node_map = graph_rewrite_map(final_expr, symbolic)

    # The final root should be (x_var + y_var).
    expected = x_var + y_var
    self.assertEqual(rewritten_sink, expected)

    # Each sub-expression has its own "final" result.
    # (y + 0) -> y_var
    self.assertEqual(node_map[sum_with_zero], y_var)
    # (x + (y+0)) -> (x + y)
    self.assertEqual(node_map[combined], expected)
    # neg(neg(x+y)) -> (x + y)
    self.assertEqual(node_map[double_neg], expected)
    # 0 + (x+y) -> (x + y)
    self.assertEqual(node_map[final_expr], expected)

    # x_var, y_var, zero_node remain unchanged
    self.assertEqual(node_map[x_var], x_var)
    self.assertEqual(node_map[y_var], y_var)
    self.assertEqual(node_map[zero_node], zero_node)

  def test_complex_multi_var_edges(self):
    """
    Build a multi-variable expression with multiple intermediates:

    Let x, y, z be variables, plus 0 and 1 as constants:

      x_var = UOp.variable('x', 1, 10)
      y_var = UOp.variable('y', -5, 5)
      z_var = UOp.variable('z', 0, 5)  # or any range you like
      zero_node = UOp.const(dtypes.int, 0)
      one_node = UOp.const(dtypes.int, 1)

    Then create a series of sub-expressions:

      1) yz_sum       = y_var + z_var
      2) yz_sum_zero  = yz_sum + zero_node       -> rewrites to yz_sum
      3) yz_neg       = -yz_sum_zero             -> -(y+z)
      4) yz_dneg      = -yz_neg                  -> y+z    (double neg gone)
      5) x_plus_yz    = x_var + yz_dneg          -> x + (y+z)
      6) double_neg_x = -(-x_plus_yz)            -> x + (y+z)
      7) final_expr   = double_neg_x * one_node  -> x + (y+z)

    We expect the final result to be (x + (y+z)). More importantly, we check
    each intermediate node's final rewrite.
    """
    x_var = UOp.variable('x', 1, 10)
    y_var = UOp.variable('y', -5, 5)
    z_var = UOp.variable('z', 0, 5)
    zero_node = UOp.const(dtypes.int, 0)
    one_node = UOp.const(dtypes.int, 1)

    # Build sub-expressions step by step:
    yz_sum = y_var + z_var            # (y + z)
    yz_sum_zero = yz_sum + zero_node  # (y + z) + 0
    yz_neg = -yz_sum_zero             # -(y+z)
    yz_dneg = -yz_neg                 # neg(neg(y+z)) => (y+z)
    x_plus_yz = x_var + yz_dneg       # x + (y+z)
    double_neg_x = -(-x_plus_yz)      # neg(neg(x+(y+z))) => x + (y+z)
    final_expr = double_neg_x * one_node  # (x+(y+z)) * 1 => x + (y+z)

    rewritten_sink, node_map = graph_rewrite_map(final_expr, symbolic)

    # The final root should be x + (y + z):
    expected = x_var + (y_var + z_var)
    self.assertEqual(rewritten_sink, expected)

    # Now check each original sub-expression in the node_map:
    #
    # (y + z) itself has no further zero or double-neg rewrites, so yz_sum stays yz_sum:
    self.assertEqual(node_map[yz_sum], yz_sum)

    # (y+z) + 0 => just (y+z)
    self.assertEqual(node_map[yz_sum_zero], yz_sum)

    # -(y+z) remains -(y+z) until we double-negate it:
    #self.assertEqual(node_map[yz_neg], yz_neg)

    # -(-(y+z)) => (y+z)
    self.assertEqual(node_map[yz_dneg], yz_sum)

    # x + (y+z) remains that expression (unless you have a further rewrite):
    #self.assertEqual(node_map[x_plus_yz], x_plus_yz)

    # -(-(x+(y+z))) => x + (y+z)
    #self.assertEqual(node_map[double_neg_x], x_plus_yz)

    # (x+(y+z)) * 1 => x + (y+z)
    #self.assertEqual(node_map[final_expr], x_plus_yz)

    # Unchanged atomic nodes map to themselves:
    self.assertEqual(node_map[x_var], x_var)
    self.assertEqual(node_map[y_var], y_var)
    self.assertEqual(node_map[z_var], z_var)
    self.assertEqual(node_map[zero_node], zero_node)
    self.assertEqual(node_map[one_node], one_node)

if __name__ == "__main__":
  unittest.main()