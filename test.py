#!/usr/bin/env python3
from tinygrad import Tensor

# Create input tensor (batch_size=1, channels=3, height=8, width=8)
input_tensor = Tensor.randn(1, 3, 8, 8)
print("Input tensor shape:", input_tensor.shape)

# Create weight tensor for conv2d (out_channels=16, in_channels=3, kernel_height=3, kernel_width=3)
weight_tensor = Tensor.randn(16, 3, 3, 3)
print("Weight tensor shape:", weight_tensor.shape)

# Perform conv2d with kernel size 3x3, stride 1, dilation 1
output = input_tensor.conv2d(weight_tensor, stride=1, dilation=1)
print("Output tensor shape:", output.shape)
print("Output tensor:")
print(output.numpy())
