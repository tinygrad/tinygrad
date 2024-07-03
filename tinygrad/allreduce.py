# allreduce.py
from typing import List
from tinygrad.tensor import Tensor

def allreduce(tensors: List[Union[Tensor, float]]):
  # Simple implementation for now
  average = sum(tensor.to('cpu') if isinstance(tensor, Tensor) else tensor for tensor in tensors) / len(tensors)
  for tensor in tensors:
    if isinstance(tensor, Tensor):
      tensor.assign(average.to(tensor.device))
