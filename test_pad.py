from tinygrad.tensor import Tensor
#import torch

print(f"{'*'*8} simple padding test {'*'*8}")
p = Tensor([[1,2], [3,4]])

old_padding = tuple(((1,1), (1,1)))
pp = (1,1, [2,2])
print(p.clean_padding(pp))
padded = p.pad(pp)

print(padded.numpy())

print(f"{'*'*8} pad2d test {'*'*8}")
import tinygrad.tensor as tn
import tinygrad.nn as nn

# Create input tensor
input_tensor = tn.Tensor([[[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]]]])

# Create convolutional layer
conv_layer = nn.Conv2d(1, 1, 3)

# Set the convolutional layer weights manually for testing purposes
conv_layer.weight.data = tn.Tensor([[[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0],
                                 [7.0, 8.0, 9.0]]]])

# Perform convolution
output = conv_layer(input_tensor)

# Print the output tensor
print(output)
#"""