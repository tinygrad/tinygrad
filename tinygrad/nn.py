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

  def __call__(self, x, training = False):
    if training: #how to determine?
      [bs, sz], m = x.shape[:2], x.shape[2]*x.shape[3]
      div =  Tensor(np.array([1/bs/m], dtype=np.float32),  gpu=x.gpu, requires_grad=False)
      crow =  Tensor.ones(1,bs,  gpu=x.gpu, requires_grad=False)
      ccol =  Tensor.ones(m,1, gpu=x.gpu, requires_grad=False)
      self.running_mean = crow.dot(x.reshape(shape=[bs,sz*m])).reshape(shape=[sz,m]).dot(ccol).reshape(shape=[sz]).mul(div)
      y = (x - self.running_mean.reshape(shape=[1, -1, 1, 1])).mul(x - self.running_mean.reshape(shape=[1, -1, 1, 1]))
      self.running_var = crow.dot(y.reshape(shape=[bs,sz*m])).reshape(shape=[sz,m]).dot(ccol).reshape(shape=[sz]).mul(div)

    x = x.sub(self.running_mean.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.div(self.running_var.add(self.eps).reshape(shape=[1, -1, 1, 1]).sqrt())
    x = x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
    return x

