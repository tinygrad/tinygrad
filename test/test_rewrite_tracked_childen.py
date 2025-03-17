import unittest
from tinygrad import Tensor
from tinygrad.ops import PatternMatcher, Ops, UPat, graph_rewrite

class TestRewriteTrackedChildren(unittest.TestCase):
  def test_simple_child(self):
    rewrite = PatternMatcher([
      (UPat(Ops.CONST, arg=2, name="x"), lambda x: x.replace(arg=4))
    ])
    a = Tensor(2)
    view_w_child = a.lazydata.src[0]
    b = Tensor(3)
    c = a + b
    sink = c.lazydata
    print([x().arg for x in view_w_child.children])
    print([x.arg for x in sink.get_children_map()[view_w_child]])
    self.assertSetEqual(set([x.arg for x in sink.get_children_map()[view_w_child]]), set((2,3)))
    # children can either be added to or removed from the map with graph_rewrite
    # added to is easy to detect, just hook the UOp constructor
    # when are children removed?
    #  * if a rewrite rule returns a UOp, the matched node is removed from the graph
    sink = graph_rewrite(sink, rewrite)
    print([x().arg for x in view_w_child.children])
    print([x.arg for x in sink.get_children_map()[view_w_child]])
    self.assertSetEqual(set([x.arg for x in sink.get_children_map()[view_w_child]]), set((3,4)))

if __name__ == '__main__':
  unittest.main()
