import unittest
from tinygrad import Tensor, dtypes, Variable
from tinygrad.helpers import Context, GlobalCounters, getenv, DEBUG
from tinygrad.uop.ops import UOp

class TestDoubleMatmul(unittest.TestCase):
  def test_double_matmul(self):
    with Context(DEBUG=0):
      a, b, c = [Tensor.randn(16, 16).contiguous().realize() for _ in range(3)]
      ref = a.numpy() @ b.numpy() @ c.numpy()
    with Context(DEBUG=max(2, DEBUG.value)):
      out = (a @ b @ c).numpy()
    self.assertLess(abs(out-ref).max(), 1e-3)

class TestRangeifyAssign(unittest.TestCase):
  def test_assign_permuted(self):
    A = Tensor.empty(4, 4, dtype='int')
    B = Tensor.arange(16).reshape(4,4)
    ret = A.permute(1,0).assign(B)
    lst = ret.tolist()
    lst2 = A.tolist()
    lst3 = B.tolist()
    print(lst)
    print(lst2)
    print(lst3)
    self.assertListEqual(lst, lst3)
    self.assertListEqual(lst2, B.permute(1, 0).tolist())

class TestRangeifyEdgeCase(unittest.TestCase):
  def test_variable_stack_data(self):
    # a bound-Variable STACK used as data gets ranges from its graph position
    v = Variable("v", 0, 10).bind(3)
    t = Tensor(UOp.stack(v, v+1).cast(dtypes.int)) + Tensor.arange(2)
    self.assertListEqual(t.tolist(), [3, 5])

  def test_variable_data_and_shape(self):
    # the same Variable has a data edge through CAST and a structural edge through SHRINK
    v = Variable("shared_v", 1, 10).bind(3)
    t = Tensor.ones(10)[:v] * Tensor(v.cast(dtypes.float))
    self.assertEqual(t.sum().item(), 9)

  def test_matmul_relu_cat(self):
    a = Tensor.ones(100, 512).contiguous().realize()
    c = Tensor.ones(1, 512).contiguous().realize()
    cm = Tensor.ones(512, 512)
    c = c @ cm
    c = c.relu()

    res = Tensor.cat(a, c, dim=0)
    self.assertEqual(res.numpy()[-1, :16].tolist(), [512] * 16)

  def test_multi_gather(self):
    # regression test: local bufferize must have device set for const_like to work
    # NOTE: with uint type, this will become a long and fail on WEBGPU
    forest = Tensor(list(range(8)), dtype='int')
    idx = Tensor([0, 0], dtype='int')
    node_val = forest.gather(0, idx)
    idx2 = idx * 2 + 1
    node_val2 = forest.gather(0, idx2)
    result = (node_val + node_val2).numpy()
    self.assertEqual(result.tolist(), [1, 1])

if getenv("BIG") > 2:
  # llama 8B (8192)
  BS, HEADS, SEQLEN, EMB = 4, 32, 8192, 128
elif getenv("BIG") > 1:
  # llama 8B
  BS, HEADS, SEQLEN, EMB = 4, 32, 2048, 128
elif getenv("BIG") > 0:
  # bigger
  BS, HEADS, SEQLEN, EMB = 4, 32, 128, 128
else:
  BS, HEADS, SEQLEN, EMB = 4, 2, 16, 8

def fa():
  Tensor.manual_seed(1337)
  with Context(DEBUG=0): q,k,v = [Tensor.rand(BS, HEADS, SEQLEN, EMB).contiguous().realize() for _ in range(3)]
  GlobalCounters.reset()
  return q.scaled_dot_product_attention(k, v)

if __name__ == '__main__':
  unittest.main()
