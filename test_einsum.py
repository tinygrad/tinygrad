import torch
from tinygrad import Tensor
import numpy as np

# Example usage:
example_input_A = np.random.rand(10, 2, 4, 5, 6, 3)
example_input_B = np.random.rand(10, 1, 4, 1, 6, 3)
epsilon_matrix = [[[0,0,0],[0,0,1],[0,-1,0]], [[0,0,-1],[0,0,0],[1,0,0]], [[0,1,0],[-1,0,0],[0,0,0]]]
epsilon = torch.tensor(epsilon_matrix).float()
epsilon_tiny = Tensor(epsilon_matrix)
epsilon_np = np.array(epsilon_matrix)

A = torch.tensor(example_input_A).float()
B = torch.tensor(example_input_B).float()
A_tiny = Tensor(example_input_A)
B_tiny = Tensor(example_input_B)
A_np = example_input_A
B_np = example_input_B
# Compute the cross product

# self.shape=(10, 1, 4, 1, 6, 3) -> new_shape=(1, 3, 1, 10, 2, 4, 5, 6)

# 4 * 6

# 10, 1, 1, 3, 6, 4
# 10, 1, 1, 3,, 6, 4
# 2 * 5
# print(torch.einsum('ijk,...i,...j->...k', epsilon, A, B))
# print(np.einsum('ijk,...i,...j->...k', epsilon, A_np, B_np))
Tensor.einsum_new('ijk,...i,...j->...k', epsilon_tiny, A_tiny, B_tiny)

# t -> 10
# l -> 2

# from tinygrad import Tensor
# import numpy as np
# import torch

# # a = Tensor(np.random.rand(1,2,3,4,5,6))
# # b = torch.tensor(np.random.rand(1, 2, 3, 4, 5, 6))


# # Assuming 'matrix' is a 2D square tensor
# matrix = torch.tensor([[1, 2], [3, 4]])
# # Calculate the trace using einsum
# trace = torch.einsum('ij->i', matrix)

# print(trace)


# some_tensor = Tensor([[1, 2], [3, 4]])
# Tensor.einsum('ij->i', some_tensor)
