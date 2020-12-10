# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from inspect import signature
import numpy as np
import os

# **** profiler ****

DEBUG = os.getenv("DEBUG", None) is not None
if DEBUG:
  import collections, atexit, time
  debug_counts = collections.defaultdict(int)
  debug_times = collections.defaultdict(float)
  def print_debug_exit():
    for name, _ in sorted(debug_times.items(), key=lambda x: -x[1]):
      print(f"{name:>20} : {debug_counts[name]:>6} {debug_times[name]:>10.2f} ms")
  atexit.register(print_debug_exit)

class ProfileOp:
  def __init__(self, name, x, backward=False):
    self.name = ("back_" if backward else "")+name
    self.x = x
  def __enter__(self):
    if DEBUG: self.st = time.time()
  def __exit__(self, *junk):
    if DEBUG:
      if cl_queue is not None:
        cl_queue.finish()
      et = (time.time()-self.st)*1000.
      debug_counts[self.name] += 1
      debug_times[self.name] += et
      print(f"{self.name:>20} : {et:>7.2f} ms {[y.shape for y in self.x]}")

# **** GPU functions ****

cl_ctx, cl_queue = None, None
def require_init_gpu():
  global cl_ctx, cl_queue
  if cl_queue is None:
    cl_ctx = cl.create_some_context(interactive=False)
    # this is an in-order command queue
    cl_queue = cl.CommandQueue(cl_ctx)

class GPUBuffer:
  def __init__(self, shape, hostbuf=None):
    self.shape, self.dtype = tuple(shape), np.float32
    self.cl = hostbuf.cl if isinstance(hostbuf, GPUBuffer) else \
      cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE | (cl.mem_flags.COPY_HOST_PTR if hostbuf is not None else 0), 4*np.prod(shape),
                hostbuf=hostbuf.astype(np.float32).ravel() if hostbuf is not None else None)

  def __repr__(self):
    return f"<GPUBuffer with shape {self.shape!r}>"

# **** start with two base classes, Tensor and Function ****

class Tensor:
  did_float_warning = False
  default_gpu = False
  ops_cpu, ops_gpu = {}, {}

  def __init__(self, data, gpu=None, requires_grad=True):
    if gpu is None:
      gpu = Tensor.default_gpu
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
    elif GPU and isinstance(data, GPUBuffer):
      self.gpu = True
    elif not isinstance(data, np.ndarray):
      raise TypeError(f"Error constructing tensor with {data!r}")

    if isinstance(data, np.ndarray):
      if data.dtype != np.float32 and not Tensor.did_float_warning:
        # warning? float64 is actually needed for numerical jacobian
        print(f"warning, {data.shape!r} isn't float32")
        Tensor.did_float_warning = True
      self.gpu = False

    self.data = data
    self.grad = None
    self.requires_grad = requires_grad

    if gpu:
      self.cuda_()

    # internal variables used for autograd graph construction
    self._ctx = None

  def __repr__(self):
    return f"Tensor {self.data!r} with grad {(self.grad.data if self.grad else None)!r}"

  def assign(self, x):
    self.data = x.data

  @property
  def shape(self):
    return self.data.shape

  @property
  def dtype(self):
    return self.data.dtype

  # ***** creation helper functions *****

  @classmethod
  def zeros(cls, *shape, **kwargs):
    return cls(np.zeros(shape, dtype=np.float32), **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs):
    return cls(np.ones(shape, dtype=np.float32), **kwargs)

  @classmethod
  def randn(cls, *shape, **kwargs):
    return cls(np.random.randn(*shape).astype(np.float32), **kwargs)

  @classmethod
  def uniform(cls, *shape, **kwargs):
    return cls((np.random.uniform(-1., 1., size=shape)/np.sqrt(np.prod(shape))).astype(np.float32), **kwargs)

  @classmethod
  def eye(cls, dim, **kwargs):
    return cls(np.eye(dim).astype(np.float32), **kwargs)

  # ***** toposort and backward pass *****

  def deepwalk(self, visited: set, nodes: list):
    visited.add(self)
    if self._ctx:
      [i.deepwalk(visited, nodes) for i in self._ctx.parents if i not in visited]
      nodes.append(self)
    return nodes

  def backward(self):
    assert self.shape == (1,)

    # fill in the first grad with one
    # this is "implicit gradient creation"
    self.grad = Tensor(np.ones(self.shape, dtype=self.dtype), gpu=self.gpu, requires_grad=False)

    for t0 in reversed(self.deepwalk(set(), [])):
      assert (t0.grad is not None)
      with ProfileOp(t0._ctx.__class__.__name__, [t0.grad], backward=True):
        grads = t0._ctx.backward(t0._ctx, t0.grad.data)
      if len(t0._ctx.parents) == 1:
        grads = [grads]
      for t,g in zip(t0._ctx.parents, grads):
        if g is None:
          continue
        assert g.shape == t.shape, \
          f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
        gt = Tensor(g, requires_grad=False)
        t.grad = gt if t.grad is None else (t.grad + gt)

  # ***** tinygrad supports CPU and GPU *****

  def cpu(self):
    if self.gpu:
      ret = Tensor(np.empty(self.shape, dtype=np.float32), gpu=False)
      cl.enqueue_copy(cl_queue, ret.data, self.data.cl, is_blocking=True)
      if self.grad:
        ret.grad = self.grad.cpu()
      return ret
    else:
      return self

  def cuda_(self):
    self.data = self.cuda().data
    self.gpu = True

  def cuda(self):
    if not GPU:
      raise Exception("No GPU Support, install pyopencl")
    if not self.gpu:
      require_init_gpu()
      ret = Tensor(GPUBuffer(self.shape, self.data))
      if self.grad:
        ret.grad = self.grad.cuda()
      return ret
    else:
      return self

  def detach(self):
    return Tensor(self.data, self.gpu)

  # ***** non first class ops *****

  def mean(self, axis=None):
    out = self.sum(axis=axis)
    coeff = np.prod(out.shape)/np.prod(self.shape)
    return out * coeff

  def sqrt(self):
    return self.pow(0.5)

  def div(self, y):
    return self * (y ** -1.0)

  def swish(self):
    return self * self.sigmoid()

  def tanh(self):
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def leakyrelu(self, neg_slope=0.01):
    return self.relu() + (-neg_slope*self).relu()

  def abs(self):
    return self.relu() + (-1.0*self).relu()

# An instantiation of the Function is the Context
class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, *x, **kwargs):
    ctx = self(*x) # self - operation i.e 'add', 'sub', etc.
    # use default params
    params = signature(self.forward).parameters
    for p in params.values():
      if p.default is not p.empty:
        setattr(ctx, p.name, p.default)
    # overwrite with passed params
    for k, v in kwargs.items():
      setattr(ctx, k, v)
    with ProfileOp(ctx.__class__.__name__, x):
      ret = Tensor(self.forward(ctx, *[t.data for t in x], **kwargs),
                   requires_grad=any([t.requires_grad for t in x]))
    if ret.requires_grad:
      ret._ctx = ctx
    return ret

def register(name, fxn, gpu=False):
  if gpu:
    Tensor.ops_gpu[name] = fxn
  else:
    Tensor.ops_cpu[name] = fxn
  def dispatch(*x, **kwargs):
    tt = [arg for arg in x if isinstance(arg, Tensor)][0]
    x = [Tensor(np.array([arg], dtype=tt.dtype), gpu=tt.gpu, requires_grad=False) if not isinstance(arg, Tensor) else arg for arg in x]
    f = (Tensor.ops_gpu if tt.gpu else Tensor.ops_cpu)[name]
    f.cl_ctx, f.cl_queue = cl_ctx, cl_queue
    return f.apply(f, *x, **kwargs)
  setattr(Tensor, name, dispatch)
  # TODO: div is a second class op, so it doesn't work here
  if name in ['add', 'sub', 'mul', 'pow']:
    setattr(Tensor, f"__{name}__", dispatch)
    setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(dispatch(self,x)))
    setattr(Tensor, f"__r{name}__", lambda self,x: dispatch(x,self))

# this registers all the operations
import tinygrad.ops_cpu
try:
  import pyopencl as cl
  import tinygrad.ops_gpu
  GPU = True
except ImportError:
  # no GPU support
  GPU = False

