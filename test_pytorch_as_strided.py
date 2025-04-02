import torch
import numpy as np

# Create a 3x3 tensor like in the test
x = torch.randn(3, 3)
print("Original tensor:")
print(x)

# Apply as_strided with the same parameters as the failing test
y = x.as_strided((2, 2), (1, 2))
print("\nAfter as_strided((2, 2), (1, 2)):")
print(y)

# Visualize what's happening
print("\nMapping:")
for i in range(2):
    for j in range(2):
        orig_idx = i*1 + j*2  # Apply the stride pattern
        orig_i, orig_j = orig_idx // 3, orig_idx % 3
        if orig_idx < 9:  # Check if it's within bounds
            print(f"y[{i},{j}] = x_flat[{orig_idx}] = x[{orig_i},{orig_j}] = {x[orig_i, orig_j].item()}") 