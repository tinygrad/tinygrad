import unittest
from unittest.mock import patch
from tinygrad import Tensor
from tinygrad.function import function
from tinygrad.uop.ops import UOp, RewriteContext

class TestOrderFreeRewrite(unittest.TestCase):
  # graph_rewrite uses fast_rewrite if the PatternMatchers are order_free. new buffer numbers are in the order of the rewrites,
  # so the patterns that make new buffers must set order_free = False
  def setUp(self):
    self.depth, self.draws, cnt, fast_rewrite = 0, 0, UOp.unique_num, RewriteContext.fast_rewrite
    def counted():
      for x in cnt:
        self.draws += self.depth > 0
        yield x
    def wrapped(rctx, root):
      self.depth += 1
      try: return fast_rewrite(rctx, root)
      finally: self.depth -= 1
    self.patches = [patch.object(UOp, "unique_num", counted()), patch.object(RewriteContext, "fast_rewrite", wrapped)]
    for p in self.patches: p.start()
  def tearDown(self):
    for p in self.patches: p.stop()
    self.assertEqual(self.draws, 0)

  def test_realize(self):
    # the second one is a schedule cache hit
    for _ in range(2): (Tensor.ones(4, 4).contiguous() + 1).sum(0).realize()
  def test_copy(self): (Tensor.ones(4).contiguous().to("NULL:1") + 1).realize()
  def test_shard_reduce(self): Tensor.ones(8, 8).contiguous().shard(("NULL:0", "NULL:1"), axis=0).sum().realize()
  def test_function(self):
    @function
    def f(a:Tensor) -> Tensor: return (a + 1).contiguous() * 2
    f(f(Tensor.ones(4).contiguous())).realize()
  def test_nested_function_backward(self):
    w = Tensor.ones(2, 2).contiguous().realize()
    @function(allow_implicit=True)
    def inner(x:Tensor) -> Tensor: return x + w
    @function(allow_implicit=True)
    def outer(a:Tensor, b:Tensor) -> Tensor: return inner(a.reshape(1,2) + b.reshape(1,2))
    a, b = Tensor.ones(2).contiguous(), Tensor.ones(2).contiguous()
    outer(a, b).sum().backward()
    assert a.grad is not None and b.grad is not None
    Tensor.realize(a.grad, b.grad)

if __name__ == '__main__':
  unittest.main()
