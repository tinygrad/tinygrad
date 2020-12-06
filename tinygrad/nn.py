import numpy as np
from tinygrad.tensor import Tensor

class BatchNorm2D:
  def __init__(self, sz, eps=0.001):
    self.eps = Tensor([eps], requires_grad=False)
    self.two = Tensor([2], requires_grad=False)
    self.weight = Tensor.ones(sz)
    self.bias = Tensor.zeros(sz)

    self.running_mean = Tensor.zeros(sz, requires_grad=False)
    self.running_var = Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)
    self.sz = sz

  def __call__(self, x):
    sz = x.shape[1]
    BS = x.shape[0]
    m = x.shape[2]*x.shape[3]
    print(x.shape,BS,m)

    A =  Tensor(np.ones([1,BS], dtype=np.float32)/BS,  gpu=x.gpu, requires_grad=False)
    B =  Tensor(np.ones([m,1], dtype=np.float32)/m,  gpu=x.gpu, requires_grad=False)
    self.running_mean = A.dot(x.reshape(shape=[BS,x.shape[1]*m])).reshape(shape=[x.shape[1],m]).dot(B).reshape(shape=[x.shape[1]])
    y = (x - self.running_mean.reshape(shape=[1, -1, 1, 1]))*(x - self.running_mean.reshape(shape=[1, -1, 1, 1]))
    self.running_var = A.dot(y.reshape(shape=[BS,x.shape[1]*m])).reshape(shape=[y.shape[1],m]).dot(B).reshape(shape=[x.shape[1]])


    # this work at inference?
    x = x.sub(self.running_mean.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.div(self.running_var.add(self.eps).reshape(shape=[1, -1, 1, 1]).sqrt())
    x = x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
    return x

