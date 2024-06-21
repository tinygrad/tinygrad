#%%
import unittest
from tinygrad.codegen.uops import *
from tinygrad.engine.graph import print_tree



class TestRewrite(unittest.TestCase):
  def test_simple_tree(self):
    x = (UOp.var("x") + UOp.var("x")) * (UOp.var("x") + UOp.var("y"))
    print_tree(x)
    assert x.src[0] != x.src[1]

    graph = UOpGraph([x])
    graph.nodes = {}
    x = graph.graph_dedup(x,PatternMatcher([]))
    # assert x.src[0] == x.src[1]
    assert len(graph.nodes)==5


  def test_simple_pattern_matcher(self):
    x = -(-(UOp.var("x")))

    pm = PatternMatcher([
      (-(-UOp.var("x")), lambda x: x)
    ])

    print_tree(x)
    graph = UOpGraph([x])
    graph.nodes = {}
    x = graph.graph_dedup(x,pm)
    print_tree(x)



if __name__ == "__main__":
  TestRewrite().test_simple_pattern_matcher()
