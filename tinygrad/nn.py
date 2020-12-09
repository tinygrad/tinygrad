import collections
import numpy as np
from abc import abstractmethod, ABCMeta

from tinygrad.tensor import Tensor
from tinygrad.utils import get_parameters

class BatchNorm2D:
  def __init__(self, sz, eps=0.001):
    self.eps = eps #Tensor([eps], requires_grad=False)
    self.two = 2.0 #Tensor([2], requires_grad=False)
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

class Model(metaclass=ABCMeta):
    def __init__(self, *args, **kargs):
        self._total_params = 0

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _modules(self):
        _modules = []
        for _, layer in self.__dict__.items():
            if isinstance(layer, Tensor):
                _modules.append(layer)
            elif isinstance(layer, (tuple,list)):
                if all(isinstance(l, Tensor) for l in layer):
                    [_modules.append(l) for l in layer]
        return _modules

    def init_backward(self):
        for p in self.parameters():
            p.grad = 0.0
        return

    @property
    def modules(self):
        return self._modules()

    @property
    def  total_params(self):
        return len(self.parameters())

    def parameters(self):
        return get_parameters(self)

    def summary(self):
        # bad coding :(
        format_summary = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Shape", "Param #")
        format_summary += "\n"
        for key, mod in enumerate(self._modules()):
            params = self.total_params
            name = mod.name if mod.name else "Layer" + f" {key}"
            in_shape = mod.shape
            params = params
            format_summary += "|{:>20}  {:>25} {:>15}|".format(
            name,
            str(in_shape),
            "{0:,}".format(params),
            )
            format_summary += "\n"
        print(format_summary)
