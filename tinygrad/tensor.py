# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
import os, atexit, time, inspect, functools, importlib
from collections import defaultdict
import numpy as np
from tinygrad.helpers import prod
from typing import List

# **** enumerate supported devices ****

class Device:
  _ops = sorted(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "llops")))
  imports = dict(enumerate([os.path.splitext(x)[0] for x in _ops if x.startswith("ops_")]))
  DEFAULT = None
  buffers, llops = {}, {}
  for i,op in imports.items():
    name = op[len("ops_"):].upper()
    vars()[name] = name 
    DEFAULT = name if os.environ.get(name, 0) == "1" else DEFAULT
    try:
      llops[name] = importlib.import_module('tinygrad.llops.'+op)
      def find_buffer(llo, name): return [cls for cname, cls in inspect.getmembers(llo, inspect.isclass) if (cname.upper() == name + "BUFFER")][0]
      buffers[name] = find_buffer(llops[name], name)
    except ImportError as e:
      print(op, "not available", e)
  DEFAULT = CPU if DEFAULT is None else DEFAULT

# **** start with two base classes, Tensor and Function ****

class Tensor:
  did_float_warning = False
  training = False

  def __init__(self, data, device=Device.DEFAULT, requires_grad=True):
    self.device, self.data = device, self._move_data(data, device)

    self.grad, self.requires_grad = None, requires_grad

    # internal variables used for autograd graph construction
    self._ctx = None

  def __repr__(self):
    return f"<Tensor {self.data!r} with grad {(self.grad.data if self.grad else None)!r}>"

  def realize(self):
    if getattr(self.data, 'realize', None) is not None:
      self.data.realize()

  def assign(self, x):
    if not isinstance(x, Tensor):
      x = Tensor(x)
    assert self.shape == x.shape
    self.data = x.data
    return x

  @property
  def shape(self):
    return self.data.shape

  @staticmethod
  def _get_data_dtype(data):
    return data.getdtype() if getattr(data, 'getdtype', None) else (data.dtype if getattr(data, 'dtype', None) else np.float32)

  @property
  def dtype(self):
    return Tensor._get_data_dtype(self.data)

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
  def arange(cls, stop, start=0, **kwargs):
    return cls(np.arange(start=start, stop=stop).astype(np.float32), **kwargs)

  @classmethod
  def uniform(cls, *shape, **kwargs):
    return cls((np.random.uniform(-1., 1., size=shape)/np.sqrt(prod(shape))).astype(np.float32), **kwargs)

  @classmethod
  def eye(cls, dim, **kwargs):
    return cls(np.eye(dim).astype(np.float32), **kwargs)

  # ***** toposort and backward pass *****

  def deepwalk(self):
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node._ctx:
        [_deepwalk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])

  def backward(self):
    assert self.shape == (1,)

    # fill in the first grad with one
    # this is "implicit gradient creation"
    self.grad = Tensor.ones(*self.shape, device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      if not any(x.requires_grad for x in t0._ctx.parents):
        continue
      assert (t0.grad is not None)
      grads = t0._ctx.backward(t0._ctx, t0.grad.data)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, \
            f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)

  # ***** tinygrad supports many devices *****

  @staticmethod
  def _move_data(data, device):
    if isinstance(data, Device.buffers[device]):
      return data
    if isinstance(data, list):
      # TODO: don't use np.array here, support Tensor creation direct to device
      data = np.array(data, dtype=np.float32)
    if isinstance(data, np.ndarray):
      data = data.view(Device.buffers[Device.CPU])

    if Tensor._get_data_dtype(data) != np.float32 and not Tensor.did_float_warning:
      # warning? float64 is actually needed for numerical jacobian
      print(f"warning, {data.shape!r} isn't float32, it's {data.dtype}")
      Tensor.did_float_warning = True

    data = data.toCPU().view(Device.buffers[Device.CPU])
    return Device.buffers[device].fromCPU(data)

  def to_(self, device):
    self.data, self.device = self._move_data(self.data, device), device
    if self.grad: self.grad.to_(device)

  def to(self, device):
    ret = Tensor(self.data, device)
    if self.grad: ret.grad = self.grad.to(device)
    return ret

  def detach(self):
    return Tensor(self.data, device=self.device, requires_grad=False)

  def numpy(self):
    return np.array(self.cpu().data)

  # ***** non first class ops *****
  
  def __getitem__(self, val):
    arg = []
    new_shape = []
    if val is not None:
      for i, s in enumerate(val if isinstance(val, (list, tuple)) else [val]):
        if isinstance(s, int):
          arg.append((s, s + 1))
        else:
          arg.append((s.start if s.start is not None else 0,
            (s.stop if s.stop >=0 else self.shape[i]+s.stop) if s.stop is not None else self.shape[i]))
          new_shape.append(arg[-1][1] - arg[-1][0])
          assert s.step is None or s.step == 1
    new_shape += self.shape[len(arg):]
    ret = self.slice(arg = arg + [(0,self.shape[i]) for i in range(len(arg), len(self.shape))])
    return ret.reshape(shape=new_shape) if tuple(ret.shape) != tuple(new_shape) else ret

  def cat(self, *args, dim=0):
    dim = (dim + len(self.shape)) if dim < 0 else dim
    for y in args: assert len(self.shape) == len(y.shape)
    args = [self] + list(args)
    s = [[] for _ in range(len(args))]
    for i in range(len(self.shape)):
      if i != dim:
        assert self.shape[i] == y.shape[i]
        for j in range(len(args)):
          s[j].append((0, self.shape[i]))
      else:
        shape_sum = 0
        for y in args: shape_sum += y.shape[i]
        k = 0
        for j,y in enumerate(args):
          s[j].append((-k, shape_sum-k))
          k += y.shape[i]
    ret = self.slice(arg=s[0])
    for ts,y in zip(s[1:], args[1:]):
      ret += y.slice(arg=ts)
    return ret

  def pad2d(self, padding):
    return self[:, :, -padding[2]:self.shape[2]+padding[3], -padding[0]:self.shape[3]+padding[1]]

  def matmul(x, w):
    bs, groups = prod(x.shape[0:-2]), prod(w.shape[0:-2])
    cin, cout = w.shape[-2], w.shape[-1]
    out_shape_t = tuple(list(x.shape[0:-2])+[cout,-1])
    if len(x.shape) == 1: order, out_shape_t = (0,), (cout, )
    else: order = tuple(list(range(len(x.shape)-2))+[len(x.shape)-1, len(x.shape)-2])
    worder = tuple(list(range(len(w.shape)-2))+[len(w.shape)-1, len(w.shape)-2])

    # NOTE: with NHWC we can remove the transposes
    # bs x groups*cin x H x W
    cx = x.transpose(order=order).reshape(shape=(bs//groups, groups*cin, -1, 1))
    # groups*cout x cin x H, W
    cw = w.transpose(order=worder).reshape(shape=(groups*cout, cin, 1, 1))
    return cx.conv2d(cw, groups=groups).reshape(shape=out_shape_t).transpose(order=order)

  dot = matmul

  def transpose(self, order=(1,0)):
    return self.permute(order=order)

  def flatten(self, start_dim=0):
    return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))

  def _canonicalize_reduce_axis(self, axis):
    if axis is None: axis = range(len(self.shape))
    if isinstance(axis, int): axis = [axis]
    axis = tuple([x if x >= 0 else x+len(self.shape) for x in axis])
    shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis]
    shape = [1] if shape == [] else shape
    return axis, shape

  def sum(self, axis=None, keepdim=False):
    axis, out_shape = self._canonicalize_reduce_axis(axis)
    ret = self._sum(axis=axis)
    return ret if keepdim or ret.shape == out_shape else ret.reshape(shape=out_shape)

  def max(self, axis=None, keepdim=False):
    axis, out_shape = self._canonicalize_reduce_axis(axis)
    ret = self._max(axis=axis)
    return ret if keepdim or ret.shape == out_shape else ret.reshape(shape=out_shape)

  def mean(self, axis=None, keepdim=False):
    out = self.sum(axis=axis, keepdim=keepdim)
    return out * (prod(out.shape)/prod(self.shape))

  def sqrt(self):
    return self.pow(0.5)

  def div(self, y):
    return self * (y ** -1.0)
  __truediv__ = div

  def sigmoid(self):
    #e = self.exp(); return e.div(1 + e)
    return (1.0 + (0.0-self).exp()) ** -1.0

  def elu(self, alpha=1.0):
    return self.relu() - (-alpha*(self.exp() - 1)).relu()

  def swish(self):
    return self * self.sigmoid()

  def relu6(self):
    return self.relu() - (self-6).relu()

  def clip(self, min, max):
    return ((self-min).relu()+min) - (self-max).relu()

  def hardswish(self):
    return self * (self+3).relu6() * (1/6)

  def tanh(self):
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0

  def gelu(x):
    # https://github.com/huggingface/transformers/blob/master/src/transformers/activations.py
    #import torch; return Tensor(torch.nn.functional.gelu(torch.tensor(x.data)).numpy())
    return 0.5 * x * (1 + (x * 0.7978845608 * (1 + 0.044715 * x * x)).tanh())

  def leakyrelu(self, neg_slope=0.01):
    return self.relu() - (-neg_slope*self).relu()

  def _softmax(self):
    m = self - self.max(axis=len(self.shape)-1, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=len(self.shape)-1, keepdim=True)

  def softmax(self):
    _, e, ss = self._softmax()
    return e.div(ss)

  def logsoftmax(self):
    m, _, ss = self._softmax()
    return m - ss.log()

  def dropout(self, p=0.5):
    if not Tensor.training: return self
    _mask = np.asarray(np.random.binomial(1, 1.0-p, size=self.shape), dtype=self.dtype)
    return self * Tensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))

  def softplus(self, limit=20, beta=1):
    # safe softplus - 1/beta*log(1 + exp(beta*x)) (PyTorch)
    eb = (self*beta).exp()
    ret = (1 + eb).log()
    return (1/beta)*ret

  def mish(self):
    return self * (self.softplus().tanh()) # x*tanh(softplus(x))

  def abs(self):
    return self.relu() + (-1.0*self).relu()

  def sign(self):
    return self / (self.abs() + 1e-10)

  def _pool2d(self, py, px):
    xup = self[:, :, :self.shape[2]-self.shape[2]%py, :self.shape[3]-self.shape[3]%px] if (self.shape[2]%py != 0) or (self.shape[3]%px != 0) else self
    return xup.reshape(shape=(xup.shape[0], xup.shape[1], xup.shape[2]//py, py, xup.shape[3]//px, px))

  def avg_pool2d(self, kernel_size=(2,2)):
    return self._pool2d(*kernel_size).mean(axis=(3,5))

  def max_pool2d(self, kernel_size=(2,2)):
    return self._pool2d(*kernel_size).max(axis=(3,5))

  def conv2d(self, weight, bias=None, **kwargs):
    ret = self._conv2d(weight, **kwargs)
    return ret if bias is None else ret.add(bias.reshape(shape=[1, -1, 1, 1]))

  # ***** broadcasted binary ops *****

  @staticmethod
  def broadcasted(fxn, x, y):
    tt = [arg for arg in [x,y] if isinstance(arg, Tensor)][0]  # this is the prototype tensor
    if not isinstance(x, Tensor): x = Tensor([x], device=tt.device, requires_grad=False) 
    if not isinstance(y, Tensor): y = Tensor([y], device=tt.device, requires_grad=False) 

    n_dims = max(len(x.shape), len(y.shape))
    if len(x.shape) != n_dims: x = x.reshape(list(x.shape) + [1]*(n_dims-len(x.shape)))
    if len(y.shape) != n_dims: y = y.reshape(list(y.shape) + [1]*(n_dims-len(y.shape)))

    shape_ret = tuple([max(sx, sy) for sx,sy in zip(x.shape, y.shape)])
    if x.shape != shape_ret: x = x.expand(shape_ret)
    if y.shape != shape_ret: y = y.expand(shape_ret)
    return fxn(x, y)

  # TODO: are these the only ones that can take number arguments?
  def add(self, x): return Tensor.broadcasted(Tensor._add, self, x)
  def sub(self, x): return Tensor.broadcasted(Tensor._sub, self, x)
  def mul(self, x): return Tensor.broadcasted(Tensor._mul, self, x)
  def pow(self, x): return Tensor.broadcasted(Tensor._pow, self, x)

  # ***** functional nn ops *****

  # TODO: fix the kwargs problem
  def reshape(self, shape): return self._reshape(shape=shape)
  def expand(self, shape): return self._expand(shape=shape)

  def linear(self, weight, bias):
    shp = [1] * (len(self.shape)-1) + [-1]
    ret = self.mul(weight.reshape(shape=shp)) if len(weight.shape) == 1 else self.dot(weight)
    return ret.add(bias.reshape(shape=shp))

  def sequential(self, ll):
    for l in ll: self = l(self)
    return self

  def layernorm(x, eps=1e-5):
    y = (x - x.mean(axis=-1, keepdim=True))
    return y.div((y*y).mean(axis=-1, keepdim=True).add(eps).sqrt())

# An instantiation of the Function is the Context
from tinygrad.ops import Ops
class Function(Ops):
  def __new__(cls, *args, **kwargs):
    cls.forward = staticmethod(cls.forward)
    cls.backward = staticmethod(cls.backward)
    return super().__new__(cls)

  def __init__(self, device, *tensors):
    self.device = device
    self.parents = tensors
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = any(self.needs_input_grad)
    self.saved_tensors = []

  buffer = property(lambda self: Device.buffers[self.device])
  op = property(lambda self: Device.llops[self.device])

  def save_for_backward(self, *x):
    if self.requires_grad:
      self.saved_tensors.extend(x)

  @classmethod
  def apply(cls, *x:List[Tensor], **kwargs):
    ctx = cls(x[0].device, *x)
    ret = Tensor(cls.forward(ctx, *[t.data for t in x], **kwargs),
                 device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
      ret._ctx = ctx    # used by autograd engine
    return ret

# register functions to move between devices
for device in [device for device in Device.__dict__.keys() if device[0] != "_"]:
  setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, Device.__dict__[device]))
  setattr(Tensor, f"{device.lower()}_", functools.partialmethod(Tensor.to_, Device.__dict__[device]))

# register all the mlops "math" operations
def register(name, fxn):
  def dispatch(*x, **kwargs): return fxn.apply(*x, **kwargs)   # TODO: there's probably a very pythonic thing to replace this with
  setattr(Tensor, "_"+name if (getattr(Tensor, name, None) is not None) else name, dispatch)
for name, cls in inspect.getmembers(importlib.import_module('tinygrad.mlops'), inspect.isclass):
  if name[0] != "_" and name != "Function" and not name.endswith("Ops"): register(name.lower(), cls)

# register the operators
# TODO: add div
def register_op(name, fxn):
  setattr(Tensor, f"__{name}__", fxn)
  setattr(Tensor, f"__i{name}__", lambda self,x: self.assign(fxn(self,x)))
  setattr(Tensor, f"__r{name}__", lambda self,x: fxn(x,self))
for name in ['add', 'sub', 'mul', 'pow', 'matmul']: register_op(name, getattr(Tensor, name))