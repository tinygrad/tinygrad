import torch
import numpy as np
from tinygrad import Tensor

def test_pytorch_behavior():
    print("Testing PyTorch behavior with duplicate values")
    
    # Test case 1: Equal values, but indices matter
    x = torch.tensor([5, 5, 5])
    values, indices = x.topk(3)
    print(f"Equal values [5, 5, 5]:")
    print(f"PyTorch values: {values.numpy()}, indices: {indices.numpy()}")
    
    # Test case 2: Equal values with larger indices
    x = torch.tensor([5, 5, 5, 5, 5])
    values, indices = x.topk(3)
    print(f"Equal values [5, 5, 5, 5, 5]:")
    print(f"PyTorch values: {values.numpy()}, indices: {indices.numpy()}")
    
    # Test case from the error: Should return [1, 0]
    x = torch.tensor([0, 1e-5])
    values, indices = x.topk(2)
    print(f"Small difference [0, 1e-5]:")
    print(f"PyTorch values: {values.numpy()}, indices: {indices.numpy()}")
    
    # The opposite case
    x = torch.tensor([1e-5, 0])
    values, indices = x.topk(2)
    print(f"Small difference [1e-5, 0]:")
    print(f"PyTorch values: {values.numpy()}, indices: {indices.numpy()}")
    
if __name__ == "__main__":
    test_pytorch_behavior() 