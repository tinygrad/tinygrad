from tinygrad.tensor import Tensor
from typing import List

def mse_loss(y_pred:List[Tensor], y_true:List[Tensor]):
  loss = (y_pred - y_true)**2
  return loss.mean()

def mae_loss(y_pred:List[Tensor], y_true:List[Tensor]):
  loss = (y_pred - y_true).abs()
  return loss.mean()
