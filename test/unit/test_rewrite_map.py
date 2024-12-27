import unittest
from tinygrad import dtypes
from tinygrad.ops import UOp, symbolic_simple, RewriteContext

class TestRewriteMap(unittest.TestCase):
  def test_add_zero(self):
    # Build a small graph: add(0, add(const=0, const=5))
    zero_node = UOp.const(dtypes.int, 0)
    five_node = UOp.const(dtypes.int, 5)
    inner_add = zero_node + five_node
    root_add = zero_node + inner_add

    # Perform top-down rewrite
    ctx = RewriteContext(symbolic_simple)
    rewritten_sink = ctx.rewrite(root_add)
    node_map = ctx.replace

    # We expect that add(0, add(0, 5)) -> add(0, 5) -> 5
    # So the final node should be 'five_node'
    assert rewritten_sink == five_node

    # Check the mapping
    assert node_map[root_add] == five_node
    assert node_map[inner_add] == five_node
    # zero_node and five_node map to themselves
    assert node_map[zero_node] == zero_node
    assert node_map[five_node] == five_node

if __name__ == "__main__":
  unittest.main()