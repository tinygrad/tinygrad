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

  def __call__(self, x):
    # TODO: use tinyops for this
    # mean op needs to support the axis argument before we can do this
    #self.running_mean.data = x.data.mean(axis=(0,2,3))
    #self.running_var.data = ((x - self.running_mean.reshape(shape=[1, -1, 1, 1]))**self.two).data.mean(axis=(0,2,3))

    # this work at inference?
    x = x.sub(self.running_mean.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.div(self.running_var.add(self.eps).reshape(shape=[1, -1, 1, 1]).sqrt())
    x = x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
    return x

