#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import torch
from extra.torch_backend import backend

# Create a simple tensor on tiny device
x = torch.tensor([1., 2., 3., 4., 5.], device='tiny', dtype=torch.float32)
print(f"Input device: {x.device}")
print(f"Input dtype: {x.dtype}")

# Try diag without autograd
with torch.no_grad():
  result = torch.diag(x)
  print(f"Result shape: {result.shape}")
  print(f"Result device: {result.device}")
  # Convert to CPU to print values
  print(f"Result (on CPU): {result.cpu()}")
  print("\nSUCCESS! torch.diag is working!")
