import unittest
import os

@unittest.skip("TODO: this graph doesn't change")
class TestViz(unittest.TestCase):
  def setUp(self) -> None:
    os.environ["TRACK_MATCH_STATS"] = "2"
  def tearDown(self) -> None:
    os.environ["TRACK_MATCH_STATS"] = "0"

  def assert_valid_graph(self, t):
    from tinygrad.ops import contexts
    from tinygrad.engine.realize import lower_schedule
    from viz.serve import create_graph
    contexts.clear()
    s = t.schedule()
    list(lower_schedule(s))
    for i,ctx in enumerate(contexts):
      ret = create_graph(ctx)
      for j,(x,y) in enumerate(zip(ret.uops, ret.uops[1:])):
        if x.key == y.key:
          raise AssertionError(f"failed to generate the correct diff at rewrite {j} ctx {i}")

  def test_ctx_diff(self):
    from tinygrad import Tensor
    a = Tensor.ones(4, 1).contiguous().realize()
    out = a + a.reshape(1, 4)
    self.assert_valid_graph(out)

  def test_gemm_diff(self):
    from tinygrad import Tensor
    x = Tensor.empty(64, 64).realize()
    y = Tensor.empty(64, 64).realize()
    out = x.matmul(y)
    self.assert_valid_graph(out)

if __name__ == "__main__":
  unittest.main()
