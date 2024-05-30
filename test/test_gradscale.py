import unittest
import numpy as np

from tinygrad import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn import Linear


def _get_test_scales():
  ints = [i for i in range(-10, 11)]
  floats = [float(i) for i in ints]
  fractions = [10 ** i for i in range(-6, 7)]

  return ints + floats + fractions


class TestGradScale(unittest.TestCase):
  def test_int_dtype(self):
    # define integer variable
    x = Tensor([1, 1])
    a = Tensor([5, 2], requires_grad=True)
    b = Tensor(1, requires_grad=True)

    # for gradients on integer variables, scaling with floating values
    # will lead to conversion to int after the multiplication
    for c in _get_test_scales():
      a.grad, b.grad = None, None
      (a.gradscale(c) * x + b.gradscale(c)).sum().backward()

      # unscaled grad of a is 1 and thus equal to c
      np.testing.assert_allclose(a.grad.numpy(), np.array((int(c), int(c))))
      # unscaled of b is 1+1=2 (a has 2 dims) and thus equal to 2*c
      np.testing.assert_allclose(b.grad.numpy(), np.array((int(c * 2),)))

  def test_float_dtype(self):
    # define float variables
    x = Tensor([1.0, 1.0])
    a = Tensor([5.0, 2.0], requires_grad=True)
    b = Tensor(10.0, requires_grad=True)

    # for floating variables, the gradients are simply scaled
    for c in _get_test_scales():
      a.grad, b.grad = None, None
      (a.gradscale(c) * x + b.gradscale(c)).sum().backward()

      # unscaled grad of a is 1 and thus equal to c
      np.testing.assert_allclose(a.grad.numpy(), np.array((c, c)))
      # unscaled of b is 1+1=2 (a has 2 dims) and thus equal to 2*c
      np.testing.assert_allclose(b.grad.numpy(), np.array((c * 2,)))

  def test_identity_scale(self):
    # input
    Tensor.manual_seed(42)
    x = Tensor.rand((100, 20))

    # network
    fc = Linear(20, 3)
    fc.weight.requires_grad = True
    fc.bias.requires_grad = True

    # compute gradient without scaling
    fc(x).sigmoid().sum().backward()
    weight_grad = fc.weight.grad.numpy()
    bias_grad = fc.bias.grad.numpy()

    # compute gradient with scaling of 1
    fc.weight.grad, fc.bias.grad = None, None
    fc(x).sigmoid().gradscale(1).sum().backward()
    weight_grad_scaled = fc.weight.grad.numpy()
    bias_grad_scaled = fc.bias.grad.numpy()

    np.testing.assert_allclose(weight_grad, weight_grad_scaled)
    np.testing.assert_allclose(bias_grad, bias_grad_scaled)

  def test_descendants(self):
    # input
    Tensor.manual_seed(42)
    x = Tensor.rand((100, 20))

    # network
    fc1 = Linear(20, 10)
    fc2 = Linear(10, 5)
    fc3 = Linear(5, 1)

    # set require_grads to `True`
    opt = Adam([fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias])

    # forward without scaling
    h1 = fc1(x).sigmoid()
    h2 = fc2(h1).sigmoid()
    h3 = fc3(h2).sigmoid()
    h3.mean().backward()
    gradients = [e.grad.numpy() for e in opt.params]

    # forward with scaling after third layer
    opt.zero_grad()
    h1 = fc1(x).sigmoid()
    h2 = fc2(h1).sigmoid().gradscale(0.5)
    h3 = fc3(h2).sigmoid()
    h3.mean().backward()
    scaled_gradients = [e.grad.numpy() for e in opt.params]

    # gradients of layer 3 should be equal
    np.testing.assert_allclose(gradients[-1], scaled_gradients[-1])
    np.testing.assert_allclose(gradients[-2], scaled_gradients[-2])

    # gradients of layer 1 and 2 should be half as big due to grad scaling
    np.testing.assert_allclose(gradients[0], scaled_gradients[0] * 2)
    np.testing.assert_allclose(gradients[1], scaled_gradients[1] * 2)
    np.testing.assert_allclose(gradients[2], scaled_gradients[2] * 2)
    np.testing.assert_allclose(gradients[3], scaled_gradients[3] * 2)

if __name__ == '__main__':
  unittest.main()
