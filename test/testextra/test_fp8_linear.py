#!/usr/bin/env python
"""Tests for FP8Linear layer - only runs on MI300X/MI350X (gfx94x/gfx95x)."""
import unittest
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Linear
from extra.fp8.fp8_linear import FP8Linear, convert_to_float8_training
from tinygrad.device import is_dtype_supported

BS, T, in_dim, out_dim = 4, 2, 8, 16

@unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3), f"no fp8e4m3 on {Device.DEFAULT}")
class TestFP8Linear(unittest.TestCase):
  def _test_forward(self, shape, in_features, out_features):
    fp8_layer = FP8Linear(in_features, out_features)
    normal_layer = Linear(in_features, out_features)
    weight = Tensor.randn(out_features, in_features, dtype=dtypes.float32)
    bias = Tensor.randn(out_features, dtype=dtypes.float32)
    fp8_layer.weight.assign(weight)
    normal_layer.weight.assign(weight)
    fp8_layer.bias.assign(bias)
    normal_layer.bias.assign(bias)
    x = Tensor.randn(*shape, dtype=dtypes.float32)
    y_fp8, y_normal = fp8_layer(x), normal_layer(x)
    np.testing.assert_allclose(y_fp8.numpy(), y_normal.numpy(), rtol=0.5, atol=5.0)

  def _test_backward(self, shape, in_features, out_features):
    fp8_layer = FP8Linear(in_features, out_features)
    normal_layer = Linear(in_features, out_features)
    weight = Tensor.randn(out_features, in_features, dtype=dtypes.float32)
    bias = Tensor.randn(out_features, dtype=dtypes.float32)
    fp8_layer.weight, normal_layer.weight = weight.detach(), weight.detach()
    fp8_layer.bias, normal_layer.bias = bias.detach(), bias.detach()
    fp8_layer.weight.requires_grad = normal_layer.weight.requires_grad = True
    x_fp8 = Tensor.randn(*shape, dtype=dtypes.float32, requires_grad=True)
    x_normal = x_fp8.detach().requires_grad_(True)
    fp8_layer(x_fp8).sum().backward()
    normal_layer(x_normal).sum().backward()
    np.testing.assert_allclose(x_fp8.grad.numpy(), x_normal.grad.numpy(), rtol=1.0, atol=1.0)
    np.testing.assert_allclose(fp8_layer.weight.grad.numpy(), normal_layer.weight.grad.numpy(), rtol=1.0, atol=1.0)

  def test_forward_2d(self): self._test_forward((BS, in_dim), in_dim, out_dim)
  def test_forward_3d(self): self._test_forward((BS, T, in_dim), in_dim, out_dim)

  def test_backward_2d(self): self._test_backward((BS, in_dim), in_dim, out_dim)
  def test_backward_3d(self): self._test_backward((BS, T, in_dim), in_dim, out_dim)

  def test_filter_fn_and_conversion(self):
    class Model:
      def __init__(self):
        self.fc1 = Linear(4, 2)
        self.fc2 = Linear(2, 1)
      def __call__(self, x):
        return self.fc2(self.fc1(x).relu())
    model = Model()
    x = Tensor.randn(3, 4)
    y_before = model(x).numpy()
    convert_to_float8_training(model, module_filter_fn=lambda _, fqn: "fc1" in fqn)
    self.assertIsInstance(model.fc1, FP8Linear)
    self.assertIsInstance(model.fc2, Linear)
    y_after = model(x).numpy()
    np.testing.assert_allclose(y_after, y_before, rtol=0.5, atol=1.0)

if __name__ == '__main__':
  unittest.main()
