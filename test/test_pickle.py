import unittest, pickle
import numpy as np
from tinygrad import Tensor

class TestPickle(unittest.TestCase):
  def test_pickle_tensor(self):
    t = Tensor.rand(10, 10).realize()
    st = pickle.dumps(t)
    t2:Tensor = pickle.loads(st)
    np.testing.assert_equal(t.numpy(), t2.numpy())

  #def test_pickle_jit(self): pass

if __name__ == '__main__':
  unittest.main()
