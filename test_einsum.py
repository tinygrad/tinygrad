import torch
from tinygrad import Tensor
import numpy as np

# # Example usage:
# example_input_A = np.random.rand(10, 2, 4, 5, 6, 3)
# example_input_B = np.random.rand(10, 1, 4, 1, 6, 3)
# epsilon_matrix = [[[0,0,0],[0,0,1],[0,-1,0]], [[0,0,-1],[0,0,0],[1,0,0]], [[0,1,0],[-1,0,0],[0,0,0]]]
# epsilon = torch.tensor(epsilon_matrix).float()
# epsilon_tiny = Tensor(epsilon_matrix)
# epsilon_np = np.array(epsilon_matrix)

# A = torch.tensor(example_input_A).float()
# B = torch.tensor(example_input_B).float()
# A_tiny = Tensor(example_input_A)
# B_tiny = Tensor(example_input_B)
# A_np = example_input_A
# B_np = example_input_B
# Compute the cross product


# example_input_a = np.random.rand(4, 2, 5, 10, 6, 3)
# example_input_b = np.random.rand(4, 1, 5, 10, 6, 3)

# temp_a = torch.tensor(example_input_a)
# temp_b = torch.tensor(example_input_b)
# A_tiny = Tensor(example_input_a)
# B_tiny = Tensor(example_input_b)

# print(torch.einsum('...i,...k->...', temp_a, temp_b).shape)
# print(A_tiny.shape[0])
# print(Tensor.einsum('...i,...j->...', A_tiny, B_tiny).numpy().shape)


#----
example_input_a = np.random.rand(1, 2, 3)

temp_a = torch.tensor(example_input_a)
A_tiny = Tensor(example_input_a)

print(torch.einsum("jki", temp_a).numpy())
print(Tensor.einsum("jki", A_tiny).numpy())

# X = [1, 2, 3]
# Y = [100, 50, 75, 200, 500]

# [x for _, x in sorted(zip(Y, X))]

# letter_val = [('i', 3), ('j', 1), ('k', 2)]
# x = Tensor.rand((1, 2, 3))

# [x_shape for _, x_shape in sorted(zip(letter_val, x.shape), key=lambda a: a[0][1])]



# a = {"i": 3, "j": 10, "k": 1}


# list(range(len(a)))
