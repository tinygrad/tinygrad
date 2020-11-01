# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from functools import partial, partialmethod
from inspect import signature
import numpy as np
try:
  import pyopencl as cl
  cl_ctx = cl.create_some_context(answers=[0,2])  # change if you don't have mac
  cl_queue = cl.CommandQueue(cl_ctx)
  GPU = True
except ImportError:
  # no GPU support
  cl_ctx, cl_queue = None, None
  GPU = False

# **** start with two base classes ****

class Tensor:
  did_float_warning = False

  def __init__(self, data):
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
    elif isinstance(data, cl._cl.Buffer):
      self.gpu = True
    elif not isinstance(data, np.ndarray):
      raise TypeError("Error constructing tensor with %r" % data)

    if isinstance(data, np.ndarray):
      shape = data.shape
      if data.dtype != np.float32 and not Tensor.did_float_warning:
        # warning? float64 is actually needed for numerical jacobian
        print("warning, %r isn't float32" % (data.shape,))
        Tensor.did_float_warning = True
      self.gpu = False

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
  def ones(*shape):
    return Tensor(np.ones(shape, dtype=np.float32))

  @staticmethod
  def randn(*shape):
    return Tensor(np.random.randn(*shape).astype(np.float32))

  @staticmethod
  def eye(dim):
    return Tensor(np.eye(dim).astype(np.float32))

  def cpu(self):
    if self.gpu:
      data = np.empty(self.shape, dtype=np.float32)
      cl.enqueue_copy(cl_queue, data, self.data)
      return Tensor(data)
    else:
      return self

  def cuda(self):
    if not GPU:
      raise Exception("No GPU Support")
    if not self.gpu:
      assert self.data.dtype == np.float32   # only float32 on GPU
      data = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
      data.shape = self.shape
      data.dtype = self.data.dtype
      return Tensor(data)
    else:
      return self
    
  def backward(self, allow_fill=True):
    #print("running backward on", self)
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # fill in the first grad with one
      # this is "implicit gradient creation"
      assert self.data.shape == (1,)
      self.grad = Tensor(np.ones(self.data.shape, dtype=self.data.dtype))
      if self.gpu:
        self.grad = self.grad.cuda()

    assert(self.grad is not None)

    grads = self._ctx.backward(self._ctx, self.grad.data)
    if len(self._ctx.parents) == 1:
      grads = [grads]
    grads = [Tensor(x) for x in grads]
    for t,g in zip(self._ctx.parents, grads):
      if g is None:
        continue
      if g.shape != t.data.shape:
        print("grad shape must match tensor shape in %r, %r != %r" %
          (self._ctx, g.shape, t.data.shape))
        assert(False)
      t.grad = g
      t.backward(False)

  # ***** put ops in these dicts *****

  ops = {}
  opsgpu = {}

  # ***** non first class ops *****

  def mean(self):
    div = Tensor(np.array([1/np.prod(self.data.shape)], dtype=self.data.dtype))
    if self.gpu:
      div = div.cuda()
    return self.sum().mul(div)

  def sqrt(self):
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)+0.5)
    return self.pow(root)

  def div(self, y):
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)-1)
    return self.mul(y.pow(root))

# An instantiation of the Function is the Context
class Function:
  cl_ctx = cl_ctx
  cl_queue = cl_queue

  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, *x, **kwargs):
    op = self
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

def register(name, fxn, gpu=False):
  if gpu:
    Tensor.opsgpu[name] = fxn
  else:
    Tensor.ops[name] = fxn
  def dispatch(self, name, *x, **kwargs):
    if self.gpu:
      f = Tensor.opsgpu[name]
    else:
      f = Tensor.ops[name]
    return f.apply(f, self, *x, **kwargs)
  setattr(Tensor, name, partialmethod(dispatch, name))

# this registers all the operations
import tinygrad.ops
if GPU:
  import tinygrad.opsgpu

