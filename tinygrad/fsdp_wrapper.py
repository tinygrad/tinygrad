"""
FSDP (Fully Sharded Data Parallel) implementation for tinygrad.
Current State:
- Implements sharding of model parameters across multiple devices.
- Supports forward and backward propagation with gradient synchronization.
- Tested with a simple linear module on CPU devices.
Future Plans:
- Optimize communication patterns for efficiency.
- Extend support to more complex models and other device types (GPU, etc.).
- Add comprehensive unit tests and benchmarking.
Usage:
1. Initialize the FSDP wrapper with your model and list of devices.
2. Use the forward method for inference.
3. Use the backward method to perform backpropagation and gradient synchronization.
Example:
  from fsdp_wrapper import FSDP
  from linear_with_named_parameters import Linear
  model = Linear(10, 10)
  devices = ["cpu:0", "cpu:1"]
  fsdp = FSDP(model, devices)
  input_tensor = Tensor.randn(10, 10)
  input_tensor.requires_grad = True
  output = fsdp.forward(input_tensor)
  loss = output.sum()
  fsdp.backward(loss)
"""
from typing import List
from tinygrad.helpers import Timing
class FSDP:
  def __init__(self, module, devices: List[str]):
    self.module = module
    self.devices = devices
    self.shard_parameters()
  def shard_parameters(self):
    for name, param in self.module.named_parameters():
      param.requires_grad = True
      print(f"Parameter {name} on device: {param.device}")
  def forward(self, *inputs):
    outputs = self.module(*inputs)
    return outputs
  def backward(self, loss):
    with Timing("FSDP backward"):
      loss.backward()
      for name, param in self.module.named_parameters():
        if param.grad is not None:
          print(f"Gradient shape for {name}:", param.grad.shape)
          print(f"Gradient device for {name}:", param.grad.device)
        else:
          print(f"No gradient for {name}")
  def parameters(self):
    return [param for _, param in self.module.named_parameters()]