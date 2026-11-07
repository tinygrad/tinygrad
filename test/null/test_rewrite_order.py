import unittest
from tinygrad import Tensor
from tinygrad.function import function
from tinygrad.uop.ops import UOp, RewriteContext

class _CountInOrderFree:
  # counts the UOp.unique_num draws made while graph_rewrite is using a rewriter that doesn't keep the rewrite order
  def __init__(self, cnt): self.cnt, self.depth, self.draws = cnt, 0, 0
  def __next__(self):
    if self.depth: self.draws += 1
    return next(self.cnt)

class TestOrderFreeRewrite(unittest.TestCase):
  # graph_rewrite uses faster rewriters if PatternMatcher.order_free. new buffer numbers are in the order of the rewrites,
  # so the patterns that make new buffers must set order_free = False
  def setUp(self):
    self.counter = _CountInOrderFree(UOp.unique_num)
    self.orig = (UOp.unique_num, RewriteContext.topo_rewrite, RewriteContext.bottom_up_rewrite)
    def wrap(fxn):
      def wrapped(rctx, root):
        self.counter.depth += 1
        try: return fxn(rctx, root)
        finally: self.counter.depth -= 1
      return wrapped
    UOp.unique_num = self.counter  # type: ignore[assignment]
    RewriteContext.topo_rewrite, RewriteContext.bottom_up_rewrite = wrap(self.orig[1]), wrap(self.orig[2])
  def tearDown(self):
    UOp.unique_num, RewriteContext.topo_rewrite, RewriteContext.bottom_up_rewrite = self.orig
    self.assertEqual(self.counter.draws, 0)

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
    Tensor.realize(a.grad, b.grad)

if __name__ == '__main__':
  unittest.main()
