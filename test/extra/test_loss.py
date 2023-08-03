import torch
from tinygrad.tensor import Tensor

# torch
input_torch  = torch.randn(3,5, requires_grad=True)
target_torch = torch.randn(3,5).softmax(dim=1)
print(input_torch.dtype)
print(input_torch)
print(target_torch)
print("torch log softmax")
sm = torch.nn.LogSoftmax(dim=1)
print(sm(input_torch))
print(sm(input_torch).mul(target_torch))
print(-sm(input_torch).mul(target_torch).sum(dim=1))
# torch CrossEntropy function
loss   = torch.nn.CrossEntropyLoss(reduction='none')
output = loss(input_torch, target_torch)
print(output)

# tinygrad
input_tinygrad  = Tensor(input_torch.detach().numpy(), requires_grad=True)
target_tinygrad = Tensor(target_torch.numpy())
print(input_tinygrad.numpy())
print(target_tinygrad.numpy())
print("tinygrad log softmax")
print(input_tinygrad.log_softmax(axis=1).numpy())
print(input_tinygrad.log_softmax(axis=1).mul(target_tinygrad).sum(axis=1).numpy())

print(input_tinygrad.log_softmax(axis=1).mul(target_tinygrad).sum(axis=1).numpy())
print(input_tinygrad.log_softmax(axis=1).mul(target_tinygrad).sum(axis=1).mean().numpy())
print(input_tinygrad.log_softmax(axis=1).mul(target_tinygrad).mean().numpy())


#loss   = torch.nn.CrossEntropyLoss(reduction='none')
#output = loss(input, target)
#print(output)
