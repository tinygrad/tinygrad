# ** setup
import unittest
import os
prev_val = os.getenv("TRACK_MATCH_STATS")
os.environ["TRACK_MATCH_STATS"] = "2"
# ** tests
from tinygrad.ops import contexts
from tinygrad import Tensor
from tinygrad.engine.realize import lower_schedule
from viz.serve import create_graph

def assert_valid_graph(t:Tensor):
  contexts.clear()
  s = t.schedule()
  list(lower_schedule(s))
  for i,ctx in enumerate(contexts):
    ret = create_graph(ctx)
    for j,(x,y) in enumerate(zip(ret.uops, ret.uops[1:])):
      if x.key == y.key:
        raise AssertionError(f"failed to generate the correct diff at rewrite {j} ctx {i}")

class TestViz(unittest.TestCase):
  def test_ctx_diff(self):
    a = Tensor.ones(4, 1).contiguous().realize()
    out = a + a.reshape(1, 4)
    assert_valid_graph(out)

  @unittest.skip("TODO: this graph doesn't change")
  def test_gemm_diff(self):
    x = Tensor.empty(64, 64).realize()
    y = Tensor.empty(64, 64).realize()
    out = x.matmul(y)
    assert_valid_graph(out)

if __name__ == "__main__":
  unittest.main()
  os.environ["TRACK_MATCH_STATS"] = "0"
