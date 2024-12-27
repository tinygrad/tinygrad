import unittest
from tinygrad import dtypes
from tinygrad.ops import UOp, symbolic_simple, RewriteContext

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
    rewritten_sink, node_map = graph_rewrite_map(root_add, symbolic_simple)

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
    Test rewriting neg(neg(5)) => 5 using symbolic_simple.
    """
    # In some versions of TinyGrad, you might do: (-(-five_node))
    five_node = UOp.const(dtypes.int, 5)
    # If your code allows UOp(...), do that; else you might do something like:
    # double_neg_five = -(-five_node)
    # But let's be explicit:
    neg_five = -five_node
    double_neg_five = -neg_five

    rewritten_sink, node_map = graph_rewrite_map(double_neg_five, symbolic_simple)

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

    rewritten_sink, node_map = graph_rewrite_map(root_add, symbolic_simple)

    # Expect final node is 'five_node'
    self.assertEqual(rewritten_sink, five_node)
    # node_map: root_add -> five_node, double_neg_five -> five_node
    self.assertEqual(node_map[root_add], five_node)
    self.assertEqual(node_map[double_neg_five], five_node)
    # zero_node, five_node map to themselves
    self.assertEqual(node_map[zero_node], zero_node)
    self.assertEqual(node_map[five_node], five_node)

if __name__ == "__main__":
  unittest.main()