from tinygrad import Tensor, nn
from tinygrad.helpers import prod
from tinygrad.tensor import Function
from tinygrad.multi import MultiLazyBuffer

# Function to gather all the shards before a fwd and bwd, and scatter the gradients after bwd
class AllGather(Function):
  def forward(self, x:MultiLazyBuffer) -> MultiLazyBuffer:
    self.input_axis = x.axis
    self.bounds = x.bounds
    return x.all_gather()
  def backward(self, grad_output:MultiLazyBuffer) -> MultiLazyBuffer: return grad_output.scatter(self.input_axis, self.bounds)


# (temporary) Layer overrides to apply the AllGather in forward to the parameters that are going to be sharded
# this behaviour is similar to Pytorch parametrization
class FSDPConv2d(nn.Conv2d):
  def __call__(self, x:Tensor) -> Tensor: return x.conv2d(
    AllGather.apply(self.weight),
    AllGather.apply(self.bias),
    self.groups, self.stride, self.dilation, self.padding
  )

class FSDPBatchNorm2d(nn.BatchNorm):
  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = self.calc_stats(x)
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * x.numel()/(x.numel()-x.shape[1]) * batch_var.detach())
      self.num_batches_tracked += 1
    return x.batchnorm(AllGather.apply(self.weight), AllGather.apply(self.bias), batch_mean, batch_var.add(self.eps).rsqrt())

class FSDPLinear(nn.Linear):
  def __call__(self, x:Tensor) -> Tensor: return x.linear(
    AllGather.apply(self.weight.transpose()),
    AllGather.apply(self.bias),
  )

# Function to shard the parameters of the optimizer (including the model itself)
def fsdp(obj, devices: tuple[str]):
  for param in nn.state.get_parameters(obj):
    if(param.shape[0] == 1 or prod(param.shape) <= 1):
       param.to_(devices)
    else:
        param.shard_(devices, axis=0)
  return obj