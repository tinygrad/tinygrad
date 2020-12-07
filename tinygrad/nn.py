from tinygrad.tensor import Tensor

class BatchNorm2D:
  def __init__(self, sz, eps=1e-5, track_running_stats=False, training=True):
    self.eps = Tensor([eps], requires_grad=False)
    self.two = Tensor([2], requires_grad=False)
    self.weight = Tensor.ones(sz)
    self.bias = Tensor.zeros(sz)
    self.track_running_stats = track_running_stats
    self.training = training

    self.running_mean = Tensor.zeros(sz, requires_grad=False)
    self.running_var = Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x):
    [bs, sz], m = x.shape[:2], x.shape[2]*x.shape[3]
    div =  Tensor([1/bs/m],  gpu=x.gpu, requires_grad=False)   #we are reimplementing means by contracting with 1's
    crow =  Tensor.ones(1,bs,  gpu=x.gpu, requires_grad=False) #we are reimplementing means by contracting with 1's
    ccol =  Tensor.ones(m,1, gpu=x.gpu, requires_grad=False)   #we are reimplementing means by contracting with 1's
    batch_mean = crow.dot(x.reshape(shape=[bs,sz*m])).reshape(shape=[sz,m]).dot(ccol).reshape(shape=[sz]).mul(div)
    y = pow((x - self.running_mean.reshape(shape=[1, -1, 1, 1])).reshape(shape=[-1]), self.two)
    #y = (x - self.running_mean.reshape(shape=[1, -1, 1, 1])).mul(x - self.running_mean.reshape(shape=[1, -1, 1, 1]))#, self.two) 
    batch_var = crow.dot(y.reshape(shape=[bs,sz*m])).reshape(shape=[sz,m]).dot(ccol).reshape(shape=[sz]).mul(div)
    if self.track_running_stats: #needs momentum
        self.running_mean = self.running_mean.mul(self.num_batches_tracked).add(batch_mean)
        self.running_var = self.running_var.mul(self.num_batches_tracked).add(batch_var)
        self.num_batches_tracked = self.num_batches_tracked.add(Tensor.ones(1, requires_grad=False))
        self.running_mean = self.running_mean.mul(self.num_batches_tracked).div(self.num_batches_tracked)
        self.running_var = self.running_var.mul(self.num_batches_tracked).div(self.num_batches_tracked)
    elif self.training: #just use the batch mean and variance  
        self.running_mean = batch_mean
        self.running_var = batch_var

    x = x.sub(self.running_mean.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.div(self.running_var.add(self.eps).reshape(shape=[1, -1, 1, 1]).sqrt())
    x = x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
    return x

