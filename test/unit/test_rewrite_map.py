import unittest
from tinygrad import dtypes, Tensor
from tinygrad.ops import UOp, symbolic, graph_rewrite_map, _substitute
from test.unit.test_tensor_uop_representation import is_pattern, realized_pattern, is_pattern_uop

class TestTensorMutates(unittest.TestCase):
  def test_mutate_add(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    ret = a+b
    pa = a.lazydata
    pb = b.lazydata
    pr = ret.lazydata
    ret.schedule()
    self.assertIsNot(pa, a.lazydata)
    self.assertIsNot(pb, b.lazydata)
    self.assertIsNot(pr, ret.lazydata)
    for t in [a,b,ret]: is_pattern(t, realized_pattern)

  def test_reshape_is_same_parent(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = a+b
    d = (a+b).reshape(3,1)
    d.realize()
    is_pattern_uop(d.lazydata.base, realized_pattern)
    is_pattern_uop(c.lazydata.base, realized_pattern)

  def test_reshape_is_same_child(self):
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = a+b
    d = (a+b).reshape(3,1)
    c.realize()
    is_pattern_uop(c.lazydata.base, realized_pattern)
    is_pattern_uop(d.lazydata.base, realized_pattern)

class TestRewriteMap(unittest.TestCase):
  def test_substitute(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    c = UOp.variable('c', 0, 10)
    e = UOp.variable('e', 0, 10)
    ret = (a+b)*c
    sub = {a+b: e}
    sub_map = graph_rewrite_map(ret, _substitute, sub, bottom_up=True)
    self.assertIs(sub_map[a+b], e)
    self.assertIs(sub_map[(a+b)*c], e*c)

  def test_substitute_depth_2(self):
    a = UOp.variable('a', 0, 10)
    b = UOp.variable('b', 0, 10)
    c = UOp.variable('c', 0, 10)
    d = UOp.variable('d', 0, 10)
    e = UOp.variable('e', 0, 10)
    f = UOp.variable('f', 0, 10)
    ret = (a+b)*c+d
    sub = {a+b: e, (a+b)*c: f}
    sub_map = graph_rewrite_map(ret, _substitute, sub, bottom_up=True)
    self.assertIs(sub_map[a+b], e)
    self.assertIs(sub_map[(a+b)*c], f)

  def test_add_zero(self):
    # Build a small graph: add(0, add(const=0, const=5))
    zero_node = UOp.const(dtypes.int, 0)
    five_node = UOp.const(dtypes.int, 5)
    inner_add = zero_node + five_node
    root_add = zero_node + inner_add

    # Perform top-down rewrite
    node_map = graph_rewrite_map(root_add, symbolic)

    # We expect that add(0, add(0, 5)) -> add(0, 5) -> 5
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

    node_map = graph_rewrite_map(double_neg_five, symbolic)

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

    node_map = graph_rewrite_map(root_add, symbolic)

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

    node_map = graph_rewrite_map(final_expr, symbolic)

    # The final root should be (x_var + y_var).
    expected = x_var + y_var

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

      x_var = UOp.variable('x', 1, 10)
      y_var = UOp.variable('y', -5, 5)
      z_var = UOp.variable('z', 0, 5)
      zero_node = UOp.const(dtypes.int, 0)
      one_node = UOp.const(dtypes.int, 1)

      yz_sum       = y_var + z_var
      yz_sum_zero  = yz_sum + zero_node   -> rewrites to yz_sum
      yz_neg       = -yz_sum_zero        -> -(y+z)
      yz_dneg      = -yz_neg             -> y+z    (double neg gone)
      x_plus_yz    = x_var + yz_dneg     -> x + (y+z)
      double_neg_x = -(-x_plus_yz)       -> x + (y+z)
      final_expr   = double_neg_x * one_node -> x + (y+z)

    We expect the final result to be (x + (y+z)).
    Each original node should map to the final node that replaces it,
    which might be structurally equivalent but not the same reference.
    """
    x_var = UOp.variable('x', 1, 10)
    y_var = UOp.variable('y', -5, 5)
    z_var = UOp.variable('z', 0, 5)
    zero_node = UOp.const(dtypes.int, 0)
    one_node = UOp.const(dtypes.int, 1)

    # Build sub-expressions
    yz_sum = y_var + z_var              # (y + z)
    yz_sum_zero = yz_sum + zero_node    # (y + z) + 0
    yz_neg = -yz_sum_zero               # -(y+z)
    yz_dneg = -yz_neg                   # -(-(y+z)) -> (y+z)
    x_plus_yz = x_var + yz_dneg         # x + (y+z)
    double_neg_x = -(-x_plus_yz)        # neg(neg(x+(y+z))) -> x+(y+z)
    final_expr = double_neg_x * one_node  # (x+(y+z)) * 1 -> x+(y+z)

    node_map = graph_rewrite_map(final_expr, symbolic)

    # (y + z) is unchanged
    self.assertEqual(node_map[yz_sum], yz_sum)

    # (y+z) + 0 => (y+z)
    self.assertEqual(node_map[yz_sum_zero], yz_sum)

    # -(y+z) remains -(y+z), but might be a new UOp with updated children
    # Compare structurally to -(y_var + z_var).
    self.assertEqual(node_map[yz_neg], -yz_sum)

    # -(-(y+z)) => (y+z)
    self.assertEqual(node_map[yz_dneg], yz_sum)

    # x + (y+z) => might get recreated if yz_dneg was changed, so compare to x + yz_sum
    self.assertEqual(node_map[x_plus_yz], x_var + yz_sum)

    # -(-(x+(y+z))) => x + (y+z)
    self.assertEqual(node_map[double_neg_x], x_var + yz_sum)

    # (x+(y+z)) * 1 => x+(y+z)
    self.assertEqual(node_map[final_expr], x_var + yz_sum)

    # Unchanged atomic nodes map to themselves
    self.assertEqual(node_map[x_var], x_var)
    self.assertEqual(node_map[y_var], y_var)
    self.assertEqual(node_map[z_var], z_var)
    self.assertEqual(node_map[zero_node], zero_node)
    self.assertEqual(node_map[one_node], one_node)

if __name__ == "__main__":
  unittest.main()
