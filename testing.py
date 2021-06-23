
from tinygrad.tensor import Tensor
import numpy as np
import torch


# x = Tensor.randn(1, 1, 4, 4)
x = Tensor.arange(9).reshape(shape=(3, 3))
x = Tensor(np.expand_dims(np.expand_dims(x.cpu().data, axis=0), axis=0))

tx = torch.from_numpy(x.cpu().data).reshape((1, 1, 3, 3))
print("Torch:")
print(tx.shape)
tt = torch.nn.functional.max_pool2d(tx, (2,2), (1, 1))
print("Torch maxpool output shape: ", tt.shape)
print(tt)

print(x.shape)
print("in:")
print(x.shape)
print(x.cpu().data)
print("out:")

# print(x.max_pool2d(kernel_size=(2,2), stride=(1,1)).cpu().data)
my = x.maxpool2d(kernel_size=(2, 2), stride=(1,1))
print(my.shape)
print(my.cpu().data)
print("Should be:")
print([4, 5])
print([7, 8])

# print([1, 2])
# print([4, 5])
# print([7, 8])

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


"""


class MaxPool2d(Function):
  def forward(ctx, x, kernel_size=(2,2), stride=None):
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None: stride = kernel_size
    elif isinstance(stride, tuple): raise Exception("MaxPool2d doesn't support asymmetrical strides yet.")
    output_shape = ((x.shape[2] - kernel_size[0])//stride + 1, (x.shape[3] - kernel_size[1])//stride + 1)
    print("Output shape", output_shape)
    ret = np.ndarray(shape=(output_shape[0] * output_shape[1]))
    # for i in range(output_shape[0]):
    for i in range(x.shape[2]):
      # for j in range(x.shape[3]):
      for j in range(x.shape[3]):
        max = x[0][0][i][j]
        max_coeff = i * x.shape[2] * j

        for k in range(1, kernel_size[0]):
          if (i + stride * k) >= x.shape[3]: continue
          m = x[0][0][i + stride * k, j]
          if m > max:
            max = m
            max_coeff = i + stride * k + x.shape[2] * j
        
        print("Index: ", (i + output_shape[1] * j - j), " is: ", max_coeff, " j: ", j, " i: ", i)
        # print("i: ", i)

        if (i + output_shape[1] * j - j >= ret.shape[0]): continue
        if (ret[i + output_shape[1] * j - j] > max_coeff): continue
        ret[i + output_shape[1] * j - j] = max_coeff
    # print(ret.reshape(2, 2).shape)
    ret = ret.reshape(output_shape[1], output_shape[0])
    print("RET")
    print(ret)
    return ret
    # return strided_pool2d(x, kernel_size, stride, 'max')

  def backward(ctx, grad_output):
    raise Exception("Not implemented yet")
"""
