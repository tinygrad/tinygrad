# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import functools, itertools
import numpy as np
from tinygrad.helpers import prod, argfix, make_pair, getenv
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union
from tinygrad.lazy import Device, LazyBuffer
from tinygrad.helpers import DEBUG

IMAGE = getenv("IMAGE", 0)

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
  _rng : ClassVar[np.random.Generator] = np.random.default_rng()
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

  @staticmethod
  def manual_seed(seed=None):
    Tensor._rng = np.random.default_rng(seed=seed)

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
  def randn(cls, *shape, **kwargs): return cls(Tensor._rng.standard_normal(size=shape, dtype=np.float32), **kwargs)

  @classmethod
  def arange(cls, stop, start=0, **kwargs): return cls(np.arange(start=start, stop=stop, dtype=np.float32), **kwargs)

  # TODO: uniform should be a late binding thing
  # Return random number between -1 and 1
  # NOTE: this behavior changed from depending on the shape to not
  @classmethod
  def uniform(cls, *shape, **kwargs): return cls((Tensor._rng.random(size=shape, dtype=np.float32) * 2 - 1), **kwargs)

  @classmethod
  def scaled_uniform(cls, *shape, **kwargs): return cls((Tensor._rng.random(size=shape, dtype=np.float32) * 2 - 1) * (prod(shape)**-0.5), **kwargs)

  @classmethod
  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  def glorot_uniform(cls, *shape, **kwargs): return cls((Tensor._rng.random(size=shape, dtype=np.float32) * 2 - 1) * ((6/(shape[0]+prod(shape[1:])))**0.5), **kwargs)

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
    _mask : np.ndarray = np.asarray(Tensor._rng.binomial(1, 1.0-p, size=self.shape), dtype=self.dtype)
    return self * Tensor(_mask, requires_grad=False, device=self.device) * (1/(1.0 - p))

  def _pool2d(self, ky, kx, sy, sx, dy=1, dx=1):
    if ky > sy or kx > sx or dy != 1 or dx != 1:
      bs,c,iy,ix = self.shape
      oy = (iy - dy * (ky-1) - 1)//sy + 1
      ox = (ix - dx * (kx-1) - 1)//sx + 1
      # duplicate the inputs for each of the kernels
      xup = self.reshape(bs, c, 1, iy, 1, ix).expand(bs, c, ky, iy, kx, ix).reshape(bs, c, ky*iy, kx*ix)
      # slide by dilation
      xup = xup.slice(((0,bs), (0,c), (0,ky*(iy+dy)), (0,kx*(ix+dx))))
      xup = xup.reshape(bs, c, ky, iy+dy, kx, ix+dx)
      xup = xup.slice(((0,bs), (0,c), (0,ky), (0,oy*sy), (0,kx), (0,ox*sx)))
      # handle stride, and permute to move reduce to the end
      return xup.reshape(bs, c, ky, oy, sy, kx, ox, sx)[:, :, :, :, 0, :, :, 0]
    else:
      # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
      xup = self.slice(((0, self.shape[0]), (0, self.shape[1]), (0, (self.shape[2]+(sy-ky))//sy*sy), (0, (self.shape[3]+(sx-kx))//sx*sx)))
      return xup.reshape(shape=(xup.shape[0], xup.shape[1], xup.shape[2]//sy, sy, xup.shape[3]//sx, sx))[:, :, :, :ky, :, :kx].permute(0, 1, 3, 2, 5, 4)

  def avg_pool2d(self, kernel_size=(2,2), stride=None): return self._pool2d(*make_pair(kernel_size), *make_pair(stride if stride is not None else kernel_size)).mean(axis=(2,4))
  def max_pool2d(self, kernel_size=(2,2), stride=None): return self._pool2d(*make_pair(kernel_size), *make_pair(stride if stride is not None else kernel_size)).max(axis=(2,4))

  def _conv2d(self, weight:Tensor, groups, padding, sy, sx, dy, dx) -> Tensor:
    (bs,cin_,_,_), (cout,cin,H,W) = self.shape, weight.shape
    assert cin*groups == cin_, f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({cin*groups} vs. {cin_})"

    # conv2d is a pooling op (with padding)
    x = self.pad2d(padding)._pool2d(H,W,sy,sx,dy,dx)
    oy, ox, rcout = x.shape[3], x.shape[5], cout//groups
    # NOTE: we do this expand explicitly so the permute isn't pushed in the binop
    x = x.reshape(bs, groups, 1, cin, H, oy, W, ox).expand(bs, groups, rcout, cin, H, oy, W, ox).permute(0,1,2,5,7,3,4,6)

    # conv! broadcasted to (bs, groups, rcout, oy, ox, cin, H, W)
    return (x * weight.reshape(1, groups, rcout, 1, 1, cin, H, W)).sum((-3, -2, -1)).reshape(bs, cout, oy, ox)

  # TODO: does this belong in Tensor?
  def _image_conv2d(self, weight:Tensor, groups, padding, sy, sx, dy, dx) -> Tensor:
    (bs,_,iy,ix), (cout,cin,H,W) = self.shape, weight.shape
    rcout = cout//groups
    x, w = self, weight.reshape(groups, rcout, cin, H, W)

    # hack for non multiples of 4 on cin
    if cin % 4 != 0 and not (cin == 1 and groups%4 == 0):
      x = x.reshape(bs, groups, cin, iy, ix)   # do this always?
      added_input_channels = 4 - (cin % 4)
      cin = cin + added_input_channels
      w = w.slice(tuple((0, cin) if i == 2 else (0, w.shape[i]) for i in range(len(w.shape))))
      x = x.slice(tuple((0, cin) if i == 2 else (0, x.shape[i]) for i in range(len(x.shape))))
      x = x.reshape(bs, groups*cin, iy, ix)

    # hack for non multiples of 4 on rcout
    added_output_channels = 0
    if rcout % 4 != 0 and not (rcout == 1 and groups%4 == 0):
      added_output_channels = 4 - (rcout % 4)
      rcout += added_output_channels
      cout = groups * rcout
      w = w.slice(tuple((0, rcout) if i == 1 else (0, w.shape[i]) for i in range(len(w.shape))))

    # packed
    x = x.permute(0,2,3,1).reshape(bs * iy, ix * groups * cin//4, 4)
    if cin == 1: w = w.reshape(cout//4,4,H*W).permute(0,2,1)
    else: w = w.reshape(cout//4,4,cin//4,4,H,W).permute(0,4,2,5,1,3).reshape(cout//4, H*cin*W, 4)

    # contiguous creates the image, and early realize static weights (TODO: don't always realize)
    x, w = x.contiguous(), w.contiguous().realize()

    # put it back for normal conv
    x = x.reshape(bs, iy, ix, groups*cin).permute(0,3,1,2)
    w = w.reshape(cout//4, H, cin//4 if cin >= 4 else 1, W, 4, 4 if cin >= 4 else 1).permute(0,4,2,5,1,3).reshape(cout, cin, H, W)

    # run normal conv
    ret = x._conv2d(w, groups, padding, sy, sx, dy, dx)

    # make image sized
    oy, ox = ret.shape[2:]
    ret = ret.permute(0,2,3,1).reshape(bs*oy, ox*cout//4, 4).contiguous()

    # undo hack for non multiples of 4 on C.rcout
    if added_output_channels != 0:
      ret = ret.reshape(bs, oy, ox, groups, rcout)[:, :, :, :, :-added_output_channels]
      rcout -= added_output_channels
      cout = groups * rcout

    # NCHW output
    return ret.reshape(bs, oy, ox, cout).permute(0,3,1,2)

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, stride=1, groups=1, dilation=1, padding=0):
    ret = mlops.Conv2D.apply(self, weight, stride=stride, groups=groups, dilation=dilation, padding=padding)
    #padding_ = [padding]*4 if isinstance(padding, int) else (padding if len(padding) == 4 else [padding[1], padding[1], padding[0], padding[0]])
    #ret = (self._image_conv2d if IMAGE >= 1 else self._conv2d)(weight, groups, padding_, *make_pair(stride), *make_pair(dilation))
    return ret if bias is None else ret.add(bias.reshape(1, -1, 1, 1))

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
