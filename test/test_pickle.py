import unittest, pickle
import numpy as np
from tinygrad import Tensor

class TestPickle(unittest.TestCase):
  def test_pickle_realized_tensor(self):
    t = Tensor.rand(10, 10).realize()
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  def test_pickle_unrealized_tensor(self):
    t = Tensor.ones(10, 10)
    #import dill
    #with dill.detect.trace(): dill.dumps(t)
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

if __name__ == '__main__':
  unittest.main()
