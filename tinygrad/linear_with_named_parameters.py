from tinygrad.tensor import Tensor
import math

class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    self.weight.requires_grad = True  # Ensure weight requires gradients
    bound = 1 / math.sqrt(in_features)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None
    if self.bias is not None:
      self.bias.requires_grad = True  # Ensure bias requires gradients
  def __call__(self, x: Tensor):
    return x.linear(self.weight.transpose(), self.bias)
  def named_parameters(self):
    params = [('weight', self.weight)]
    if self.bias is not None:
      params.append(('bias', self.bias))
    return params