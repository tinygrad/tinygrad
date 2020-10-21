import numpy as np
from tinygrad.tensor import Tensor

def jacobian(model, input):
  """
  Compute the (analytical) Jacobian of model w.r.t. input.

      model : A tinygrad model
      input : An input

  returns:

      J : Jacobian
  """
  output = model(input)

  ji = input.data.reshape(-1).shape[-1]
  jo = output.data.reshape(-1).shape[-1]
  J = np.zeros((jo,ji))

  for o in range(jo):
    # tinygrad doesn't support slicing, tiny-hack to select
    # the needed scalar an backpropagate only through it
    o_scalar = Tensor(mask_like(output.data, o, 1.)).mul(output).sum()
    o_scalar.backward()

    for i, grad in enumerate(input.grad.reshape(-1)):
      J[o,i] = grad
  return J

def mask_like(like, mask_inx, mask_value = 1.0):
  mask = np.zeros_like(like).reshape(-1)
  mask[mask_inx] = mask_value
  return mask.reshape(like.shape)

def numerical_jacobian(model, input, eps = 1e-6):
  """
  Compute the Jacobian through Finite-Difference Approximation.
  Somewhat inspired by [1] but not followed closely.

      model : A tinygrad model
      input : An input
      eps : Perturbation step

  returns:

      NJ : an approx. of the Jacobian

  [1]: https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
  """
  output = model(input)

  ji = input.data.reshape(-1).shape[-1]
  jo = output.data.reshape(-1).shape[-1]
  NJ = np.zeros((jo, ji))

  for o in range(jo):
    for i in range(ji):

      eps_perturb = mask_like(input.data, i, mask_value = eps)
      output_perturb_add = model(Tensor(input.data + eps_perturb)).data.reshape(-1)[o]
      output_perturb_sub = model(Tensor(input.data - eps_perturb)).data.reshape(-1)[o]

      grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2*eps)

      NJ[o,i] = grad_approx
  return NJ

def gradcheck(model, input, eps = 1e-06, atol = 1e-5, rtol = 0.001):
  """
  Checks whether the numerical approx. of the Jacobian of model w.r.t input is close to the
  analytical one.

      model : A tinygrad model
      input : An input
      eps : Perturbation step
      atol, rtol: Params for the numpy.allclose test

  returns:
      test_passed : Bool, whether the test passed
      J : Analytical Jacobian
      NJ : Finite-Difference approx. Jacobian
  """
  NJ = numerical_jacobian(model, input, eps)
  J = jacobian(model, input)

  test_passed = np.allclose(J, NJ, atol=atol, rtol=rtol)
  return test_passed, J, NJ
