
from tinygrad.tensor import Tensor
import numpy as np
import torch


# x = Tensor.randn(1, 1, 4, 4)
x = Tensor.arange(9).reshape(shape=(3, 3))
x = Tensor(np.expand_dims(np.expand_dims(x.cpu().data, axis=0), axis=0))

print(x.shape)
print("in:")
print(x.cpu().data)
print("out:")
print(x.maxpool2d(kernel_size=(2,2), stride=1).cpu().data)
print("Should be:")
print([4, 5])
print([7, 8])

exit()

def withTorch(x, shape, kernel_size, stride):
  x = torch.from_numpy(x).reshape(shape)
  print("Torch:")
  print(torch.nn.functional.max_pool2d(x, kernel_size, (stride, stride)).data[1])

def withTinygrad(x, shape, kernel_size, stride):
  x = Tensor(x).reshape(shape=shape)
  print("Tinygrad:")
  print(x.max_pool2d(kernel_size=kernel_size, stride=stride).cpu().cpu().data[1])

#shapes = [(1, 1, 4, 4), (1, 1, 24, 24), (1, 1, 12, 12), (1, 1, 13, 13), (2, 2, 64, 64)]
#kernel_sizes = [(2,2), (3,3), (3,2), (5,5), (5,1)]
#strides = [2, 1, 3, 4, 5]

#
shapes = [(2, 2, 4, 4)]
kernel_sizes = [(2,2)]
strides = [1]
#

for i, ksz in enumerate(kernel_sizes):
  x = np.random.randn(*shapes[i]).astype(np.float32)
  withTorch(x, shapes[i], ksz, strides[i])
  withTinygrad(x, shapes[i], ksz, strides[i])


x = torch.randn(25).reshape((5, 5))
x = x.unsqueeze(0).unsqueeze(0)

print("---------------------- Torch --------------------------")
print("x")
print(x.shape)
print("maxpool2d strided")
print(torch.nn.functional.max_pool2d(x, (2, 2), (1, 1)).shape)

print("---------------------- Tinygrad -----------------------")

x = Tensor.arange(25).reshape(shape=(5, 5))
x = Tensor(np.expand_dims(np.expand_dims(x.cpu().data, axis=0), axis=0))

print("x")
# print(x.cpu().data)
print(x.shape)

print("maxpool2d strided")
# print(x.max_pool2d(kernel_size=(2, 2), stride=1).cpu().data)
print(x.max_pool2d(kernel_size=(2, 2), stride=1).shape)
