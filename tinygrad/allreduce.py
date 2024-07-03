# allreduce.py
from typing import List, Union
from tinygrad.tensor import Tensor

def allreduce(tensors: List[Union[Tensor, float]]):
  # Ensure that only Tensor objects have their `to` method called
  tensor_values = [tensor.to('cpu') if isinstance(tensor, Tensor) else tensor for tensor in tensors]
  average = sum(tensor_values) / len(tensors)
  for tensor in tensors:
    if isinstance(tensor, Tensor):
      tensor.assign(average.to(tensor.device))
