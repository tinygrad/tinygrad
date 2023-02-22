# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import functools, itertools
import numpy as np
from tinygrad.helpers import prod, argfix, make_pair, getenv
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union
from tinygrad.lazy import Device, LazyBuffer
from tinygrad.ops import DEBUG

# An instantiation of the Function is the Context
class Function:
  def __init__(self, device:str, *tensors:Tensor):
    self.device, self.parents = device, tensors
    self.needs_input_grad = [t.requires_grad for t in self.parents]
    self.requires_grad = True if any(self.needs_input_grad) else (None if any(x is None for x in self.needs_input_grad) else False)
    self.saved_tensors : List[LazyBuffer] = []

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise NotImplementedError(f"backward not implemented for {type(self)}")

  # NOTE: it doesn't hurt to save this since the ctx will be freed fast without grad
  def save_for_backward(self, *x): self.saved_tensors.extend(x)

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs):
    ctx = fxn(x[0].device, *x)
    ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad and not Tensor.no_grad:
      ret._ctx = ctx    # used by autograd engine
    return ret

import tinygrad.mlops as mlops

# **** start with two base classes, Tensor and Function ****

class Tensor:
  __deletable__ = ('_ctx',)
  training : ClassVar[bool] = False
  no_grad : ClassVar[bool] = False

  def __init__(self, data, device=Device.DEFAULT, requires_grad:Optional[bool]=None):
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
    elif isinstance(data, LazyBuffer) and data.device != device:
      # TODO: this has to realize, it shouldn't have to
      data = data.realize().toCPU()

    if isinstance(data, np.ndarray):
      data = data if data.shape else data.reshape((1,))
      self.lazydata = LazyBuffer.fromCPU(data.astype(np.float32), device)
    elif isinstance(data, LazyBuffer):
      self.lazydata = data
    else:
      raise RuntimeError(f"can't create Tensor from {data}")

    # tensors have gradients, buffers do not
    self.grad : Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad : Optional[bool] = requires_grad

    # internal variables used for autograd graph construction
    self._ctx : Optional[Function] = None

  def __repr__(self):
    return f"<Tensor {self.lazydata if self.lazydata.realized is None else self.lazydata.realized!r} with grad {(self.grad.lazydata if self.grad else None)!r}>"

  @property
  def shape(self) -> Tuple[int, ...]: return self.lazydata.shape

  # dtype handling was very broken. it's always float32 now
  @property
  def dtype(self) -> type: return np.float32

  @property
  def device(self) -> str: return self.lazydata.device

  # ***** data handlers ****

  def realize(self) -> Tensor:
    self.lazydata.realize()
    return self

  def assign(self, x) -> Tensor:
    if not isinstance(x, Tensor): x = Tensor(x)
    assert self.shape == x.shape
    assert not x.requires_grad  # self requires_grad is okay?
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.lazydata.realized is not None and not getenv("DISALLOW_ASSIGN"): x.lazydata.output_buffer = self.lazydata.realized
    self.lazydata = x.lazydata
    return self

  def detach(self): return Tensor(self.lazydata, device=self.device, requires_grad=False)
  def numpy(self) -> np.ndarray: return np.array(self.lazydata.toCPU())

  # TODO: if things are realized this won't work
  def to_(self, device:str):
    assert self.lazydata.realized is None
    self.lazydata.device = device
    if self.grad:
      self.grad.lazydata.device = device

  def to(self, device:str):
    ret = Tensor(self.lazydata, device)
    if self.grad:
      ret.grad = self.grad.to(device)
    return ret

  # ***** creation helper functions *****

  # TODO: remove use of numpy here

  @classmethod
  def zeros_like(cls, tensor, **kwargs): return cls.zeros(*tensor.shape, **kwargs)

  @classmethod
  def zeros(cls, *shape, **kwargs): return cls(np.zeros(shape, dtype=np.float32), **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs): return cls(np.ones(shape, dtype=np.float32), **kwargs)

  @classmethod
  def empty(cls, *shape, **kwargs): return cls(np.empty(shape, dtype=np.float32), **kwargs)

  @classmethod
  def randn(cls, *shape, **kwargs): return cls(np.random.standard_normal(size=shape).astype(np.float32), **kwargs)

  @classmethod
  def arange(cls, stop, start=0, **kwargs): return cls(np.arange(start=start, stop=stop, dtype=np.float32), **kwargs)

  # TODO: uniform should be a late binding thing
  # Return random number between -1 and 1
  # NOTE: this behavior changed from depending on the shape to not
  @classmethod
  def uniform(cls, *shape, **kwargs): return cls((np.random.random(size=shape).astype(np.float32) * 2 - 1), **kwargs)

  @classmethod
  def scaled_uniform(cls, *shape, **kwargs): return cls((np.random.random(size=shape).astype(np.float32) * 2 - 1) * (prod(shape)**-0.5), **kwargs)

  @classmethod
  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  def glorot_uniform(cls, *shape, **kwargs): return cls((np.random.random(size=shape).astype(np.float32) * 2 - 1) * ((6/(shape[0]+prod(shape[1:])))**0.5), **kwargs)

  @classmethod
  def eye(cls, dim, **kwargs): return cls(np.eye(dim, dtype=np.float32), **kwargs)

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
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx

  # ***** non first class ops (hlops) *****

  # Tensors mostly follow the normal python indexing / slicing behavior for sequences
  # - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  # - A slice i:j returns the elements with indices in [i, j)
  #   - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
  #   - Negative values for i and j are taken relative to the end of the sequence
  #   - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
  # - Indexing with np.newaxis or None on a given axis will add a new dimension of size one before that axis
  # - Empty slices are not allowed
  # - Strides other than 1 are not allowedå
  def __getitem__(self, val):
    def slcfix(i, sz, default): return default if i is None else max(0, min(sz, sz+i if i < 0 else i))  # Fix negative idxs, clamp to [0,N]
    new_slice, new_shape = [], []
    val = [val] if not isinstance(val, (list, tuple)) else val
    assert sum(s is not None for s in val) <= len(self.shape)
    assert all(s.step is None or s.step == 1 for s in val if isinstance(s, slice))
    for i,(sz,s) in enumerate(zip(self.shape, [v for v in val if v is not None])):  # Slicing only depends on ints + slices
      if isinstance(s, int) and not (-sz <= s < sz):
        raise IndexError(f"index {s} is out of bounds for dimension {i} with size {sz}")
      new_slice.append((s%sz, s%sz+1) if isinstance(s, int) else (slcfix(s.start, sz, 0), slcfix(s.stop, sz, sz)))
    for s,sz in zip(val, [self.shape[i-1] for i in itertools.accumulate([s is not None for s in val])]):  # Shape depends on slices + positions of Nones
      if not isinstance(s, int):
        new_shape.append(1 if s is None else slcfix(s.stop, sz, sz) - slcfix(s.start, sz, 0))
    new_shape += [self.shape[i] for i in range(len(new_slice), len(self.shape))]
    new_slice += [(0,self.shape[i]) for i in range(len(new_slice), len(self.shape))]
    return self.slice(arg = new_slice).reshape(new_shape if len(new_shape) else (1,))

  def cat(self, *args, dim=0):
    dim = (dim + len(self.shape)) if dim < 0 else dim
    for y in args:
      assert len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim)
    catargs = [self] + list(args)
    shape_cumsum = [0, *itertools.accumulate([y.shape[dim] for y in catargs])]
    slc = [[(0, s) for s in self.shape] for _ in catargs]
    for s,k in zip(slc, shape_cumsum):
      s[dim] = (-k, shape_cumsum[-1]-k)
    return functools.reduce(Tensor.__add__, [arg.slice(arg=s) for arg,s in zip(catargs, slc)])

  # TODO: make this nicer with syntactic sugar in slice
  def chunk(self, num, dim):
    slice_params = [[(0, s) for s in self.shape] for _ in range(num)]
    for i,k in enumerate(range(0, self.shape[dim], self.shape[dim]//num)):
      slice_params[i][dim] = (k, min(self.shape[dim], k+self.shape[dim]//num))
    return [self.slice(arg=p) for p in slice_params]

  # TODO: what's the difference between dot and matmul?
  def dot(self:Tensor, w:Tensor):
    # NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
    bs, groups = prod(self.shape[0:-2]), prod(w.shape[0:-2])
    cin, cout = w.shape[-2], w.shape[-1]
    out_shape_t = tuple(list(self.shape[0:-2])+[cout,-1])
    if len(self.shape) > 1:
      order = tuple(list(range(len(self.shape)-2))+[len(self.shape)-1, len(self.shape)-2])
    else:
      order, out_shape_t = (0,), (cout, )
    worder = tuple(list(range(len(w.shape)-2))+[len(w.shape)-1, len(w.shape)-2])

    # NOTE: with NHWC we can remove the transposes
    # bs x groups*cin x H x W
    cx = self.transpose(order=order).reshape(shape=(bs//groups, groups*cin, -1, 1))
    # groups*cout x cin x H, W
    cw = w.transpose(order=worder).reshape(shape=(groups*cout, cin, 1, 1))
    return cx.conv2d(cw, groups=groups).reshape(shape=out_shape_t).transpose(order=order)

  # (padding_left, padding_right, padding_top, padding_bottom)
  def pad2d(self, padding:Tuple[int, ...]): return self.slice(arg = [(0,self.shape[0]), (0,self.shape[1]), (-padding[2],self.shape[2]+padding[3]), (-padding[0],self.shape[3]+padding[1])])
  # TODO: this is totally not transpose
  def transpose(self, order=(1,0)): return self.permute(order=order)
  def flatten(self, start_dim=0): return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))

  def _reduce(self, fxn:Type[Function], axis=None, keepdim=False):
    if axis is None:
      axis = range(len(self.shape))
    if isinstance(axis, int):
      axis = [axis]
    axis = tuple([x if x >= 0 else x+len(self.shape) for x in axis])
    shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis]
    ret = fxn.apply(self, axis=axis)
    return ret if keepdim else ret.reshape(shape=[1] if shape == [] else shape)

  def sum(self, axis=None, keepdim=False): return self._reduce(mlops.Sum, axis, keepdim)
  def max(self, axis=None, keepdim=False): return self._reduce(mlops.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self).max(axis=axis, keepdim=keepdim))

  def mean(self, axis=None, keepdim=False):
    out = self.sum(axis=axis, keepdim=keepdim)
    return out * (prod(out.shape)/prod(self.shape))

  def _softmax(self):
    m = self - self.max(axis=len(self.shape)-1, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=len(self.shape)-1, keepdim=True)

  def softmax(self):
    _, e, ss = self._softmax()
    return e.div(ss)

  # TODO: logsoftmax -> log_softmax and add dim param
  def logsoftmax(self):
    m, _, ss = self._softmax()
    return m - ss.log()

  def dropout(self, p=0.5) -> Tensor:
    if not Tensor.training:
      return self
    _mask : np.ndarray = np.asarray(np.random.binomial(1, 1.0-p, size=self.shape), dtype=self.dtype)
    return self * Tensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))

  # TODO: support arbitrary strides
  def _pool2d(self, py, px):
    xup = self[:, :, :self.shape[2]-self.shape[2]%py, :self.shape[3]-self.shape[3]%px] if (self.shape[2]%py != 0) or (self.shape[3]%px != 0) else self
    return xup.reshape(shape=(xup.shape[0], xup.shape[1], xup.shape[2]//py, py, xup.shape[3]//px, px))

  def avg_pool2d(self, kernel_size=(2,2)): return self._pool2d(*make_pair(kernel_size)).mean(axis=(3,5))
  def max_pool2d(self, kernel_size=(2,2)): return self._pool2d(*make_pair(kernel_size)).max(axis=(3,5))

  def conv2d(self, weight, bias=None, **kwargs):
    ret = mlops.Conv2D.apply(self, weight, **kwargs)
    return ret if bias is None else ret.add(bias.reshape(shape=[1, -1, 1, 1]))

  # ***** math functions (unary) *****

  def __neg__(self): return 0.0-self
  def sqrt(self): return self.pow(0.5)
  def square(self): return self*self
  def clip(self, min_, max_): return ((self-min_).relu()+min_) - (self-max_).relu()
  def abs(self): return self.relu() + (-self).relu()
  def sign(self): return self / (self.abs() + 1e-10)

  # ***** activation functions (unary) *****

  def sigmoid(self): return (1.0 + (-self).exp()).reciprocal()
  def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
  def swish(self): return self * self.sigmoid()
  def silu(self): return self.swish()   # The SiLU function is also known as the swish function.
  def relu6(self): return self.relu() - (self-6).relu()
  def hardswish(self): return self * (self+3).relu6() * (1/6)
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
  def quick_gelu(self): return self * (self * 1.702).sigmoid()
  def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()
  def mish(self): return self * self.softplus().tanh()
  def softplus(self, limit=20, beta=1): return (1/beta) * (1 + (self*beta).exp()).log()

  # ***** broadcasted binary ops *****

  @staticmethod
  def broadcasted(fxn:Type[Function], tx:Union[Tensor, float], ty:Union[Tensor, float]):
    tt = [arg for arg in [tx,ty] if isinstance(arg, Tensor)][0]  # this is the prototype tensor
    x,y = [Tensor([t], device=tt.device, requires_grad=False) if not isinstance(t, Tensor) else t for t in [tx,ty]]
    x,y = [t.reshape([1]*(max(len(x.shape), len(y.shape))-len(t.shape)) + list(t.shape)) for t in [x,y]]
    shape_ret = tuple(max(sx, sy) for sx,sy in zip(x.shape, y.shape))
    return fxn.apply(x.expand(shape_ret), y.expand(shape_ret))

  # ***** first class ops (mlops) *****

  def contiguous(self): return mlops.Contiguous.apply(self)
  def relu(self): return mlops.ReLU.apply(self)
  def log(self): return mlops.Log.apply(self)
  def exp(self): return mlops.Exp.apply(self)
  def reciprocal(self): return mlops.Reciprocal.apply(self)

  # NOTE: __pow__ and friends are broken in mypyc with the ** operator
  def __add__(self, x): return Tensor.broadcasted(mlops.Add, self, x)
  def __radd__(self, x): return Tensor.broadcasted(mlops.Add, x, self)
  def __sub__(self, x): return Tensor.broadcasted(mlops.Sub, self, x)
  def __rsub__(self, x): return Tensor.broadcasted(mlops.Sub, x, self)
  def __mul__(self, x): return Tensor.broadcasted(mlops.Mul, self, x)
  def __rmul__(self, x): return Tensor.broadcasted(mlops.Mul, x, self)
  def __pow__(self, x): return Tensor.broadcasted(mlops.Pow, self, x)
  def __rpow__(self, x): return Tensor.broadcasted(mlops.Pow, x, self)

  # non broadcasted ops
  def __truediv__(self, x): return self * (x.reciprocal() if isinstance(x, Tensor) else (1/x))
  def __rtruediv__(self, x): return self.reciprocal() * x
  def __matmul__(self, x): return self.dot(x)
  def __rmatmul__(self, x:Tensor): return x.dot(self)

  # assignment, any way to make this automatic?
  def __iadd__(self, x): return self.assign(self.__add__(x))
  def __isub__(self, x): return self.assign(self.__sub__(x))
  def __imul__(self, x): return self.assign(self.__mul__(x))
  def __ipow__(self, x): return self.assign(self.__pow__(x))
  def __itruediv__(self, x): return self.assign(self.__truediv__(x))
  def __imatmul__(self, x): self.assign(self.__matmul__(x))

  # simple tensor math API
  def add(self, x): return self.__add__(x)
  def sub(self, x): return self.__sub__(x)
  def mul(self, x): return self.__mul__(x)
  def pow(self, x): return self.__pow__(x)
  def div(self, x): return self.__truediv__(x)
  def matmul(self, x): return self.__matmul__(x)

  # ***** functional nn ops *****

  def reshape(self, shape, *args): return mlops.Reshape.apply(self, shape=argfix(shape, *args))
  def expand(self, shape, *args): return mlops.Expand.apply(self, shape=tuple(x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))))
  def permute(self, order, *args): return mlops.Permute.apply(self, order=argfix(order, *args))
  def flip(self, axis, *args): return mlops.Flip.apply(self, axis=argfix(axis, *args))
  def slice(self, arg): return mlops.Slice.apply(self, arg=arg)
  def unsqueeze(self, dim):
    if dim < 0: dim = len(self.shape) + dim + 1
    return mlops.Reshape.apply(self, shape=self.shape[:dim] + (1,) + self.shape[dim:])

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)  # type: ignore
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return functools.reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis=-1, eps=1e-5):
    y = (self - self.mean(axis=axis, keepdim=True))
    return y.div((y*y).mean(axis=axis, keepdim=True).add(eps).sqrt())

  def batchnorm(self, weight:Tensor, bias:Tensor, mean:Tensor, invstd:Tensor):
    x = (self - mean.reshape(shape=[1, -1, 1, 1])) * weight.reshape(shape=[1, -1, 1, 1])
    return x.mul(invstd.reshape(shape=[1, -1, 1, 1])) + bias.reshape(shape=[1, -1, 1, 1])

# register functions to move between devices
for device in [device for device in Device._buffers.keys() if device[0] != "_"]:
  setattr(Tensor, f"{device.lower()}", functools.partialmethod(Tensor.to, device))
  setattr(Tensor, f"{device.lower()}_", functools.partialmethod(Tensor.to_, device))
