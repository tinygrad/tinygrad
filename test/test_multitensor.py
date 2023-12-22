import unittest
from tinygrad import Tensor, Device
import numpy as np

d0, d1 = f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2"
N = 128

# shard_x is "data parallel"
# shard_w is "model parallel"

class TestMultiTensor(unittest.TestCase):
  def test_numpy(self):
    X = Tensor.ones(256)
    X.shard_((d0, d1), 0)
    np.testing.assert_allclose(X.numpy(), 1)

  def _test_simple_add_axis(self, shard_x, shard_w):
    X = Tensor.ones(256).contiguous().realize()
    W = Tensor.ones(256).contiguous().realize()
    X.shard_((d0, d1), shard_x)
    W.shard_((d0, d1), shard_w)
    O = X + W
    np.testing.assert_allclose(O.numpy(), 2)

  def test_simple_add(self): return self._test_simple_add_axis(None, None)
  def test_simple_add_X(self): return self._test_simple_add_axis(0, None)
  def test_simple_add_W(self): return self._test_simple_add_axis(None, 0)
  def test_simple_add_XW(self): return self._test_simple_add_axis(0, 0)

  def _test_simple_reduce_axis(self, shard_x):
    X = Tensor.ones(256, 256).contiguous().realize()
    X.shard_((d0, d1), shard_x)
    O = X.sum(axis=1)
    np.testing.assert_allclose(O.numpy(), 256)

  def test_simple_reduce(self): return self._test_simple_reduce_axis(None)
  def test_simple_reduce_0(self): return self._test_simple_reduce_axis(0)
  def test_simple_reduce_1(self): return self._test_simple_reduce_axis(1)

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

  def test_matmul_shard_none(self): return self._test_matmul_shard_axis(None, None)
  def test_matmul_shard_X_0(self): return self._test_matmul_shard_axis(0, None)
  def test_matmul_shard_X_1(self): return self._test_matmul_shard_axis(1, None)
  def test_matmul_shard_W_0(self): return self._test_matmul_shard_axis(None, 0)
  def test_matmul_shard_W_1(self): return self._test_matmul_shard_axis(None, 1)

  def test_double_matmul_shard_X_0(self): return self._test_double_matmul_shard_axis(0, None)
  def test_double_matmul_shard_X_1(self): return self._test_double_matmul_shard_axis(1, None)
  def test_double_matmul_shard_W_0(self): return self._test_double_matmul_shard_axis(None, 0)
  def test_double_matmul_shard_W_1(self): return self._test_double_matmul_shard_axis(None, 1)

  def test_data_parallel(self):
    import sys, pathlib
    sys.path.append((pathlib.Path(__file__).parent.parent / "extra" / "models").as_posix())
    from resnet import ResNet18
    from tinygrad.nn.state import get_parameters

    fake_image = Tensor.rand((2, 3, 224, 224))
    m = ResNet18()
    m.load_from_pretrained()
    real_output = m(fake_image).numpy()
    for p in get_parameters(m): p.shard_((d0, d1))
    shard_output = m(fake_image.shard((d0, d1), axis=0)).numpy()
    np.testing.assert_allclose(real_output, shard_output)

if __name__ == '__main__':
  unittest.main()