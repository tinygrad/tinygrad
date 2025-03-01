import numpy as np
import torch
from tinygrad import Tensor

# Force same random seed
np.random.seed(42)
torch.manual_seed(42)
Tensor.manual_seed(42)

# For testing with equal values
def test_equal_values():
    print("Testing with equal values [5, 5]")
    torch_tensor = torch.tensor([5, 5])
    tiny_tensor = Tensor([5, 5])
    
    torch_values, torch_indices = torch_tensor.topk(2)
    tiny_values, tiny_indices = tiny_tensor.topk(2)
    
    print(f"PyTorch values: {torch_values.numpy()}, indices: {torch_indices.numpy()}")
    print(f"tinygrad values: {tiny_values.numpy()}, indices: {tiny_indices.numpy()}")
    print()

# For testing the specific case from the error
def test_specific_case():
    print("Testing the specific case from the error [x, y] with indices [0, 1] vs [1, 0]")
    # Let's create a tensor where PyTorch and tinygrad might differ
    # Try different values to find a case that produces the different behavior
    for x, y in [(1, 1), (0, 0), (0.5, 0.5), (0.00001, 0), (0, 0.00001)]:
        torch_tensor = torch.tensor([x, y])
        tiny_tensor = Tensor([x, y])
        
        torch_values, torch_indices = torch_tensor.topk(2)
        tiny_values, tiny_indices = tiny_tensor.topk(2)
        
        print(f"Input: [{x}, {y}]")
        print(f"PyTorch values: {torch_values.numpy()}, indices: {torch_indices.numpy()}")
        print(f"tinygrad values: {tiny_values.numpy()}, indices: {tiny_indices.numpy()}")
        print(f"Match: {np.array_equal(torch_indices.numpy(), tiny_indices.numpy())}")
        print()

# Test with random values
def test_random_values():
    print("Testing with random values")
    # Generate random data with the same seed to trigger the test case
    for i in range(5):
        size = (2,)
        np_data = np.random.uniform(low=-2, high=2, size=size).astype(np.float32)
        
        torch_tensor = torch.tensor(np_data)
        tiny_tensor = Tensor(np_data)
        
        torch_values, torch_indices = torch_tensor.topk(2)
        tiny_values, tiny_indices = tiny_tensor.topk(2)
        
        print(f"Input: {np_data}")
        print(f"PyTorch values: {torch_values.numpy()}, indices: {torch_indices.numpy()}")
        print(f"tinygrad values: {tiny_values.numpy()}, indices: {tiny_indices.numpy()}")
        print(f"Match: {np.array_equal(torch_indices.numpy(), tiny_indices.numpy())}")
        print()

if __name__ == "__main__":
    test_equal_values()
    test_specific_case()
    test_random_values() 