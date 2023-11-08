import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.shape.symbolic import Variable

class TestSample(unittest.TestCase):
  def test_sample(self):
    X = Tensor.rand(10000, 10).realize()
    BS = 16
    idxs = np.random.randint(0, X.shape[0], size=(BS)).tolist()
    batch = [Variable(f'idx{i}', 0, X.shape[0]-1).bind(s) for i,s in enumerate(idxs)]
    x = Tensor.cat(*[X.shrink(((batch[i], batch[i]+1), (0,X.shape[1]))) for i in range(BS)])
    print(idxs)
    ret = x.numpy()
    base = X.numpy()[idxs]
    np.testing.assert_equal(ret, base)

if __name__ == '__main__':
  unittest.main()