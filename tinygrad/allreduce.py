# allreduce.py

from typing import List
from tinygrad.tensor import Tensor

def allreduce(tensors: List[Tensor]):
    # Simple implementation for now
    average = sum(tensor.to('cpu') for tensor in tensors) / len(tensors)
    for tensor in tensors:
        tensor.assign(average.to(tensor.device))
