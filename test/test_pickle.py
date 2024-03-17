import unittest, pickle
import numpy as np
from tinygrad import Tensor, TinyJit

class TestPickle(unittest.TestCase):
  def test_pickle_realized_tensor(self):
    t = Tensor.rand(10, 10).realize()
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_unrealized_tensor(self):
    t = Tensor.ones(10, 10)
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_jit(self):
    @TinyJit
    def add(a, b): return a+b
    for _ in range(3): add(Tensor.rand(10, 10), Tensor.rand(10, 10))
    #import dill
    #with dill.detect.trace(): dill.dumps(add)
    st = pickle.dumps(add)
    add_fxn = pickle.loads(st)

    x = Tensor.ones(10, 10).contiguous().realize()
    y = Tensor.ones(10, 10).contiguous().realize()
    print("post jit")
    add_fxn(x, y)

if __name__ == '__main__':
  unittest.main()
