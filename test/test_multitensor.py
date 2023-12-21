import unittest
from tinygrad import Tensor, Device
import numpy as np

d0, d1 = f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2"
N = 128

# shard_x is "data parallel"
# shard_w is "model parallel"

class TestMultiTensor(unittest.TestCase):
  def _test_matmul_shard_axis(self, shard_x, shard_w):
    X = Tensor.kaiming_uniform(N, N).realize()
    W = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard((d0, d1), shard_x)
    Ws = W.shard((d0, d1), shard_w)
    O = (Xs@Ws)
    np.testing.assert_allclose(X.numpy() @ W.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def _test_double_matmul_shard_axis(self, shard_x, shard_w):
    X = Tensor.kaiming_uniform(N, N).realize()
    W1 = Tensor.kaiming_uniform(N, N).realize()
    W2 = Tensor.kaiming_uniform(N, N).realize()
    Xs = X.shard((d0, d1), shard_x)
    W1s = W1.shard((d0, d1), shard_w)
    W2s = W2.shard((d0, d1), shard_w)
    O = (Xs@W1s)@W2s
    np.testing.assert_allclose((X.numpy() @ W1.numpy()) @ W2.numpy(), O.to(Device.DEFAULT).numpy(), atol=1e-5)

  def test_matmul_shard_X_0(self): return self._test_matmul_shard_axis(0, None)
  def test_matmul_shard_X_1(self): return self._test_matmul_shard_axis(1, None)
  def test_matmul_shard_W_0(self): return self._test_matmul_shard_axis(None, 0)
  def test_matmul_shard_W_1(self): return self._test_matmul_shard_axis(None, 1)

  def test_double_matmul_shard_X_0(self): return self._test_double_matmul_shard_axis(0, None)
  def test_double_matmul_shard_X_1(self): return self._test_double_matmul_shard_axis(1, None)
  def test_double_matmul_shard_W_0(self): return self._test_double_matmul_shard_axis(None, 0)
  def test_double_matmul_shard_W_1(self): return self._test_double_matmul_shard_axis(None, 1)

if __name__ == '__main__':
  unittest.main()