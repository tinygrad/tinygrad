from tinygrad.tensor import Tensor
from typing import List

def mse_loss(y_pred:List[Tensor], y_true:List[Tensor]):
  """
  Computes the mean squared error between two tensors.

  Args:
    y_pred (`Tensor`): A tensor of predicted values.
    y_true (`Tensor`): A tensor of ground truth values.

  Returns:
    A tensor of the mean squared error between the two tensors.
  """

  loss = (y_pred - y_true)**2
  return loss.mean()
