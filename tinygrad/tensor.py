# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from functools import partialmethod
from inspect import signature
import numpy as np

# **** start with two base classes ****

class Tensor:
  def __init__(self, data):
    #print(type(data), data)
    if type(data) == list:
      data = np.array(data, dtype=np.float32)
    if type(data) != np.ndarray:
      print("error constructing tensor with %r" % data)
      assert(False)
    if data.dtype == np.float64:
      #print("are you sure you want float64 in %r?" % data)
      pass
    self.data = data
    self.grad = None

    # internal variables used for autograd graph construction
    self._ctx = None

  def __repr__(self):
    return "Tensor %r with grad %r" % (self.data, self.grad)

  @property
  def shape(self):
    return self.data.shape

  @staticmethod
  def zeros(*shape):
    return Tensor(np.zeros(shape, dtype=np.float32))

  @staticmethod
  def randn(*shape):
    return Tensor(np.random.randn(*shape).astype(np.float32))

  @staticmethod
  def eye(dim):
    return Tensor(np.eye(dim).astype(np.float32))

  def backward(self, allow_fill=True):
    #print("running backward on", self)
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # fill in the first grad with one
      # this is "implicit gradient creation"
      assert self.data.size == 1
      self.grad = np.ones_like(self.data)

    assert(self.grad is not None)

    grads = self._ctx.backward(self._ctx, self.grad)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    for t,g in zip(self._ctx.parents, grads):
      if g is None:
        continue
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" %
          (self._ctx, g.shape, t.data.shape))
        assert(False)
      t.grad = g
      t.backward(False)

  def mean(self):
    div = Tensor(np.array([1/self.data.size], dtype=self.data.dtype))
    return self.sum().mul(div)

# An instantiation of the Function is the Context
class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  # note that due to how partialmethod works, self and arg are switched
  def apply(self, arg, *x, **kwargs):
    # support the args in both orders
    if type(arg) == Tensor:
      op = self
      x = [arg]+list(x)
    else:
      op = arg
      x = [self]+list(x)
    ctx = op(*x)
    # use default params
    params = signature(op.forward).parameters
    for p in params.values():
      if p.default is not p.empty:
        setattr(ctx, p.name, p.default)
    # overwrite with passed params
    for k, v in kwargs.items():
      setattr(ctx, k, v)
    ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))

# this registers all the operations
import tinygrad.ops

