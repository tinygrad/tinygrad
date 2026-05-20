import numpy as np
import unittest
from tinygrad import Tensor, nn
from tinygrad.helpers import Context

class TestTernaryLinear(unittest.TestCase):
  def test_weights_are_ternary(self):
    with Context(DEV="PYTHON"):
      lin = nn.TernaryLinear(16, 8)
      w = lin.weight.numpy()
      assert set(w.flatten().tolist()).issubset({-1.0, 0.0, 1.0}), f"unexpected values: {np.unique(w)}"

  def test_output_shape(self):
    with Context(DEV="PYTHON"):
      lin = nn.TernaryLinear(4, 3, bias=False)
      x = Tensor.rand(2, 4)
      y = lin(x)
      assert y.shape == (2, 3), f"expected (2, 3), got {y.shape}"

  def test_threshold_zero_no_zeros(self):
    # threshold=0 means every weight becomes ±1 (|w| > 0 for all non-zero uniform samples)
    with Context(DEV="PYTHON"):
      lin = nn.TernaryLinear(32, 16, bias=False, threshold=0.0)
      w = lin.weight.numpy()
      assert 0.0 not in w.flatten().tolist(), "threshold=0 should leave no zero weights"
      assert set(w.flatten().tolist()).issubset({-1.0, 1.0})

  def test_threshold_large_all_zeros(self):
    with Context(DEV="PYTHON"):
      lin = nn.TernaryLinear(8, 4, bias=False, threshold=1e9)
      w = lin.weight.numpy()
      np.testing.assert_array_equal(w, np.zeros_like(w))

  def test_forward_matches_manual(self):
    with Context(DEV="PYTHON"):
      lin = nn.TernaryLinear(4, 2, bias=False, threshold=1e9)
      lin.weight = Tensor([[1., 0., -1., 0.], [0., 1., 0., 1.]])
      x = Tensor([[2., 3., 4., 5.]])
      y = lin(x).numpy()
      # row 0: 1*2 + 0*3 + (-1)*4 + 0*5 = -2
      # row 1: 0*2 + 1*3 + 0*4 + 1*5  = 8
      np.testing.assert_allclose(y, [[-2., 8.]], atol=1e-5)

  def test_bias_shape(self):
    lin = nn.TernaryLinear(6, 3, bias=True)
    assert lin.bias is not None and lin.bias.shape == (3,)

  def test_no_bias(self):
    lin = nn.TernaryLinear(6, 3, bias=False)
    assert lin.bias is None

if __name__ == "__main__":
  unittest.main()
