import torch
from tinygrad.tensor import Tensor
from tinygrad.nn import CrossEntropyLoss

# torch
input_torch  = torch.randn(3,5, requires_grad=True)
target_torch = torch.randn(3,5).softmax(dim=1)

# torch CrossEntropy function
print("torch function")
torch.set_printoptions(precision=8)
print(input_torch.dtype)

loss   = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.5)
output = loss(input_torch, target_torch)
print(output)

loss   = torch.nn.CrossEntropyLoss(reduction='mean')
output = loss(input_torch, target_torch)
print(output)

loss   = torch.nn.CrossEntropyLoss(reduction='sum')
output = loss(input_torch, target_torch)
print(output)

# tinygrad
input_tinygrad  = Tensor(input_torch.detach().numpy(), requires_grad=True)
target_tinygrad = Tensor(target_torch.numpy())

print("tinygrad function")
print(input_tinygrad.dtype)

loss_tiny = CrossEntropyLoss(reduction='none', label_smoothing=0.5)
output = loss_tiny(input_tinygrad, target_tinygrad)
print(output.numpy())

loss_tiny = CrossEntropyLoss(reduction='mean')
output = loss_tiny(input_tinygrad, target_tinygrad)
print(output.numpy())

loss_tiny = CrossEntropyLoss(reduction='sum')
output = loss_tiny(input_tinygrad, target_tinygrad)
print(output.numpy())
