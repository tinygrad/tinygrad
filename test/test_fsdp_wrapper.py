import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygrad.tensor import Tensor
from linear_with_named_parameters import Linear
from fsdp_wrapper import FSDP
from tinygrad.device import Device
print("Available devices:", Device._devices)
class ExampleModule:
  def __init__(self):
    self.linear = Linear(10, 10)
  def __call__(self, x):
    return self.linear(x)
  def named_parameters(self):
    return self.linear.named_parameters()
# Initialize the module and FSDP
module = ExampleModule()
devices = ["CPU", "CPU"]  # Use CPU devices
print("Using devices:", devices)
fsdp = FSDP(module, devices)
# Create input tensor that requires gradients
input_tensor = Tensor.randn(10, 10)
input_tensor.requires_grad = True
# Forward pass with FSDP
output = fsdp.forward(input_tensor)
print("Output shape:", output.shape)
print("Output device:", output.device)
# Compute loss
loss = output.sum()  # Ensure backward is called on a scalar
print("Loss:", loss.numpy())  # Convert to numpy for display
# Backward pass with FSDP
fsdp.backward(loss)
# Print input gradient
if input_tensor.grad is not None:
  print("Input gradient shape:", input_tensor.grad.shape)
  print("Input gradient (first few values):", input_tensor.grad.numpy().flatten()[:5])
else:
  print("Input gradient is None")
# Print parameter gradients
for name, param in module.named_parameters():
  if param.grad is not None:
    print(f"Gradient shape for {name}:", param.grad.shape)
    print(f"Gradient for {name} (first few values):", param.grad.numpy().flatten()[:5])
  else:
    print(f"No gradient for {name}")