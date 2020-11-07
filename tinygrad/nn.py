from tinygrad.tensor import Tensor

def swish(x):
  return x.mul(x.sigmoid())

class BatchNorm2D:
  def __init__(self, sz, eps=0.001):
    self.eps = eps
    self.weight = Tensor.zeros(sz)
    self.bias = Tensor.zeros(sz)

    # TODO: need running_mean and running_var
    self.running_mean = Tensor.zeros(sz)
    self.running_var = Tensor.zeros(sz)
    self.num_batches_tracked = Tensor.zeros(1)

  def __call__(self, x):
    # this work at inference?
    x = x.sub(self.running_mean.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.div(self.running_var.add(Tensor([self.eps], gpu=x.gpu)).reshape(shape=[1, -1, 1, 1]).sqrt())
    x = x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
    return x

